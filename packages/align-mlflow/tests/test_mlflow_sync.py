"""align-mlflow sync, run as a command against a temporary MLflow store."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.tracing.client import TracingClient
from open_world_fixtures import SCENARIO, completion_line, episode_records, write_run

from align_mlflow.store import connect
from align_mlflow.sync import sync_run

EXPERIMENT = "open world test"
MERIT_LOW = "ADEPT-June2025-merit-0.0"
MERIT_HIGH = "ADEPT-June2025-merit-1.0"
ADM = "phase2_pipeline_direct_regression TEST_OPENWORLD"
RUN = f"{ADM} 2026-08-31__09-17-11"


@pytest.fixture
def local_store(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    MlflowClient(tracking_uri=tracking_uri).create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    return tracking_uri, connect(tracking_uri, EXPERIMENT)


def sync(path, tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT) is None:
        database = Path(tracking_uri.removeprefix("sqlite:///"))
        client.create_experiment(
            EXPERIMENT, artifact_location=str(database.parent / "artifacts")
        )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "align_mlflow",
            "sync",
            str(path),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            EXPERIMENT,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )


def logged_traces(tracking_uri):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    traces = client.search_traces(
        locations=[experiment.experiment_id], max_results=1000
    )
    return sorted(traces, key=lambda trace: trace.info.request_time)


def root_span(trace):
    return next(span for span in trace.data.spans if span.parent_id is None)


def child_names(trace):
    children = [span for span in trace.data.spans if span.parent_id is not None]
    return [span.name for span in sorted(children, key=lambda span: span.start_time_ns)]


def session(trace):
    return trace.info.trace_metadata["mlflow.trace.session"]


def test_sync_logs_each_step_as_a_trace_in_its_episode_session(tmp_path):
    write_run(
        tmp_path / "results" / "direct_regression" / "2026-08-31__09-17-11",
        episode_records(MERIT_LOW) + episode_records(MERIT_HIGH),
        [completion_line(MERIT_LOW, 0.83), completion_line(MERIT_HIGH, 0.31)],
    )
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = sync(tmp_path / "results", tracking_uri)

    assert result.returncode == 0, result.stderr
    assert f"{RUN}: 14 new steps, 2 new episode scores" in result.stdout
    traces = logged_traces(tracking_uri)
    key = traces[0].info.tags["align.run_key"]
    first_session = f"{SCENARIO} | {MERIT_LOW} | {RUN} | episode 1 | {key}"
    second_session = f"{SCENARIO} | {MERIT_HIGH} | {RUN} | episode 2 | {key}"
    assert [session(trace) for trace in traces] == [first_session] * 7 + [
        second_session
    ] * 7
    assert [trace.info.tags["mlflow.traceName"] for trace in traces[:7]] == [
        "01 CHECK_VITALS Patient 1",
        "02 TAG_CHARACTER Patient 1",
        "03 TREAT_PATIENT Patient 1",
        "04 MOVE_TO Patient 2",
        "05 TAG_CHARACTER Patient 2",
        "06 END_SCENE",
        "07 MOVE_TO_EVAC Patient 1",
    ]

    tag_step = traces[1]
    action = root_span(tag_step).outputs["output"]["action"]
    assert action["action_type"] == "TAG_CHARACTER"
    assert action["parameters"] == {"category": "IMMEDIATE"}
    assert root_span(tag_step).outputs["choice_info"]["alignment_info"]["votes"] == {
        "0": 1.0,
        "1": 0.0,
    }
    assert child_names(tag_step) == [
        "WorldStateTrackerADMComponent",
        "OWDirectRegressionADMComponent",
        "MultinomialWeightedMidpointAlignmentADMComponent",
    ]
    assert tag_step.info.tags["action_detail"] == "IMMEDIATE"

    driver_step = traces[5]
    assert driver_step.info.tags["chosen_by"] == "driver"
    assert child_names(driver_step) == []
    assert "choice_info" in root_span(driver_step).outputs
    assert (
        root_span(driver_step).attributes["align.choice_info_applies_to_action"]
        is False
    )

    assert {
        session(trace): [(a.name, a.value) for a in trace.info.assessments]
        for trace in traces
        if trace.info.assessments
    } == {
        first_session: [("ta3_session_alignment_score", 0.83)],
        second_session: [("ta3_session_alignment_score", 0.31)],
    }


def test_sync_again_adds_nothing(tmp_path):
    write_run(
        tmp_path / "2026-08-31__09-17-11",
        episode_records(MERIT_LOW),
        [completion_line(MERIT_LOW, 0.83)],
    )
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    sync(tmp_path, tracking_uri)

    again = sync(tmp_path, tracking_uri)

    assert f"{RUN}: 0 new steps, 0 new episode scores" in again.stdout
    assert len(logged_traces(tracking_uri)) == 7
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert len(client.search_runs([experiment.experiment_id])) == 1


def test_sync_follows_a_run_while_it_is_written(tmp_path):
    run_dir = tmp_path / "2026-08-31__09-17-11"
    records = episode_records(MERIT_LOW) + episode_records(MERIT_HIGH)
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    write_run(run_dir, records[:3])
    assert (
        f"{RUN}: 3 new steps, 0 new episode scores"
        in sync(tmp_path, tracking_uri).stdout
    )

    write_run(run_dir, records[:9], [completion_line(MERIT_LOW, 0.83)])
    assert (
        f"{RUN}: 6 new steps, 1 new episode scores"
        in sync(tmp_path, tracking_uri).stdout
    )

    text = json.dumps(records, indent=2)
    (run_dir / "input_output.json").write_text(text[: len(text) // 2])
    mid_write = sync(tmp_path, tracking_uri)
    assert mid_write.returncode == 1
    assert "not synced" in mid_write.stdout
    assert len(logged_traces(tracking_uri)) == 9


def test_unaligned_episode_scores_name_the_target_ta3_used(tmp_path):
    write_run(
        tmp_path / "2026-08-28__15-52-05",
        episode_records(None) + episode_records(None),
        [completion_line(MERIT_LOW, 0.4), completion_line(MERIT_HIGH, 0.6)],
    )
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    sync(tmp_path, tracking_uri)

    scored = [trace for trace in logged_traces(tracking_uri) if trace.info.assessments]
    run = f"{ADM} 2026-08-28__15-52-05"
    key = scored[0].info.tags["align.run_key"]
    assert [
        (session(trace), trace.info.assessments[0].metadata["alignment_target_id"])
        for trace in scored
    ] == [
        (f"{SCENARIO} | unaligned | {run} | episode 1 | {key}", MERIT_LOW),
        (f"{SCENARIO} | unaligned | {run} | episode 2 | {key}", MERIT_HIGH),
    ]


def test_copies_of_a_run_are_logged_once(tmp_path):
    original = write_run(
        tmp_path / "a" / "2026-08-31__09-17-11", episode_records(MERIT_LOW)
    )
    shutil.copytree(original, tmp_path / "b" / "2026-08-31__09-17-11")
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = sync(tmp_path, tracking_uri)

    assert result.stdout.count(f"{RUN}: 7 new steps") == 1
    assert result.stdout.count(f"{RUN}: 0 new steps") == 1
    assert len(logged_traces(tracking_uri)) == 7


def test_same_run_label_with_different_usernames_has_separate_sessions(tmp_path):
    for username, score in [("session-a", 0.25), ("session-b", 0.75)]:
        run_dir = write_run(
            tmp_path / username / "2026-08-31__09-17-11",
            episode_records(MERIT_LOW)[:1],
            [completion_line(MERIT_LOW, score)],
        )
        (run_dir / "meta.json").write_text(json.dumps({"username": username}))
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = sync(tmp_path, tracking_uri)

    assert result.returncode == 0, result.stderr
    traces = logged_traces(tracking_uri)
    assert len(traces) == 2
    assert len({session(trace) for trace in traces}) == 2
    assert {trace.info.assessments[0].value for trace in traces} == {0.25, 0.75}
    for trace in traces:
        assert trace.info.tags["run"] == RUN
        assert trace.info.tags["align.run_key"] in session(trace)
        assert trace.info.assessments[0].metadata["mlflow.trace.session"] == session(
            trace
        )


@pytest.mark.parametrize("failure", ["all", "child", "metadata", "confirmation"])
def test_failed_trace_export_is_reported_and_recovered(
    tmp_path, monkeypatch, local_store, failure
):
    run_dir = write_run(
        tmp_path / "2026-08-31__09-17-11", episode_records(MERIT_LOW)[:1]
    )
    tracking_uri, experiment_id = local_store
    log_spans = TracingClient.log_spans
    span_calls = 0

    def fail_spans(client, *args, **kwargs):
        nonlocal span_calls
        span_calls += 1
        if failure == "all" or span_calls == 1:
            raise MlflowException("span export unavailable")
        return log_spans(client, *args, **kwargs)

    def fail_write(*args, **kwargs):
        raise MlflowException("trace write unavailable")

    with monkeypatch.context() as patch:
        if failure == "confirmation":
            patch.setattr(TracingClient, "set_trace_tag", fail_write)
        elif failure != "metadata":
            patch.setattr(TracingClient, "log_spans", fail_spans)
        if failure in {"all", "metadata"}:
            patch.setattr(TracingClient, "start_trace", fail_write)

        failed = sync_run(experiment_id, run_dir)

    assert failed.error is not None
    assert failed.new_steps == 0
    stored = MlflowClient().search_traces(
        locations=[experiment_id], include_spans=False
    )
    assert len(stored) == (0 if failure in {"all", "metadata"} else 1)
    if stored:
        partial = TracingClient().store.get_trace(
            stored[0].info.trace_id, allow_partial=True
        )
        assert len(partial.data.spans) == (3 if failure == "child" else 4)
        assert "align.import_complete" not in partial.info.tags

    recovered = sync_run(experiment_id, run_dir)

    assert recovered.error is None
    assert recovered.new_steps == 1
    traces = logged_traces(tracking_uri)
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 4
    assert traces[0].info.tags["align.import_complete"] == "source-record-v1"
    assert sync_run(experiment_id, run_dir).new_steps == 0
    assert logged_traces(tracking_uri)[0].info.trace_id == traces[0].info.trace_id


def test_recovery_requires_cleanup_and_preserves_confirmed_steps(
    tmp_path, monkeypatch, local_store
):
    records = episode_records(MERIT_LOW)[:2]
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", records[:1])
    tracking_uri, experiment_id = local_store
    assert sync_run(experiment_id, run_dir).error is None
    first_trace_id = logged_traces(tracking_uri)[0].info.trace_id
    write_run(run_dir, records)

    def fail_confirmation(*args, **kwargs):
        raise MlflowException("confirmation unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(TracingClient, "set_trace_tag", fail_confirmation)
        assert sync_run(experiment_id, run_dir).error is not None

    with monkeypatch.context() as patch:
        patch.setattr(MlflowClient, "delete_traces", lambda *args, **kwargs: 0)
        failed_cleanup = sync_run(experiment_id, run_dir)

    assert "Could not remove incomplete imports" in failed_cleanup.error
    assert len(logged_traces(tracking_uri)) == 2

    recovered = sync_run(experiment_id, run_dir)

    assert recovered.error is None
    assert recovered.new_steps == 1
    traces = logged_traces(tracking_uri)
    assert len(traces) == 2
    assert traces[0].info.trace_id == first_trace_id
    assert all(
        trace.info.tags["align.import_complete"] == "source-record-v1"
        for trace in traces
    )


def test_cli_preserves_original_records_and_launch_files(tmp_path):
    records = episode_records(MERIT_LOW)
    records[0]["source_timestamp"] = "2026-08-31T09:17:12Z"
    records[0]["unknown_top_level"] = {"null": None, "unicode": "café"}
    records[0]["input"]["unknown_input"] = [None, False, 1]
    records[0]["input"]["full_state"]["elapsed_time"] = "0"
    records[0]["input"]["full_state"]["events"] = [{"description": "arrival"}]
    records[0]["output"]["extra_output"] = None
    records[0]["output"]["action"]["future_action_field"] = {"value": 2}
    records[0]["choice_info"]["future_choice_field"] = ["a", None]
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", records)
    (run_dir / ".hydra/overrides.yaml").write_bytes(b"- seed=42\r\n")
    (run_dir / ".hydra/hydra.yaml").write_bytes(b"hydra:\n  job:\n    name: test\n")
    files = {
        name: (run_dir / name).read_bytes()
        for name in (
            ".hydra/config.yaml",
            ".hydra/overrides.yaml",
            ".hydra/hydra.yaml",
            "meta.json",
        )
    }
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"

    result = sync(run_dir, tracking_uri)

    assert result.returncode == 0, result.stderr
    traces = logged_traces(tracking_uri)
    assert [
        {"input": root_span(trace).inputs, **root_span(trace).outputs}
        for trace in traces
    ] == records
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    source = runs[0]
    assert source.info.status == "FINISHED"
    assert source.data.tags["align.source_version"] == "0.5.11"
    assert source.data.tags["align.source_path"] == str(run_dir)
    assert {trace.info.trace_metadata["mlflow.sourceRun"] for trace in traces} == {
        source.info.run_id
    }
    destination = tmp_path / "downloaded"
    destination.mkdir()
    for name, content in files.items():
        saved = client.download_artifacts(
            source.info.run_id, f"source/{name}", str(destination)
        )
        assert Path(saved).read_bytes() == content
        assert (run_dir / name).read_bytes() == content


def test_provenance_upload_resumes_and_copies_share_one_run(
    tmp_path, monkeypatch, local_store
):
    run_dir = write_run(
        tmp_path / "a/2026-08-31__09-17-11", episode_records(MERIT_LOW)[:1]
    )
    tracking_uri, experiment_id = local_store
    client = MlflowClient()
    upload = MlflowClient.log_artifact

    def fail_metadata(self, run_id, path, **kwargs):
        if Path(path).name == "meta.json":
            raise MlflowException("artifact store unavailable")
        return upload(self, run_id, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MlflowClient, "log_artifact", fail_metadata)
        assert "artifact store unavailable" in sync_run(experiment_id, run_dir).error
    assert logged_traces(tracking_uri) == []
    assert len(client.search_runs([experiment_id])) == 1
    assert sync_run(experiment_id, run_dir).error is None
    trace_id = logged_traces(tracking_uri)[0].info.trace_id
    copy = tmp_path / "b/2026-08-31__09-17-11"
    shutil.copytree(run_dir, copy)

    def unexpected_upload(*args, **kwargs):
        pytest.fail("Unchanged provenance must not be uploaded again")

    with monkeypatch.context() as patch:
        patch.setattr(MlflowClient, "log_artifact", unexpected_upload)
        assert sync_run(experiment_id, copy).error is None
    assert len(client.search_runs([experiment_id])) == 1
    assert logged_traces(tracking_uri)[0].info.trace_id == trace_id


def test_changed_provenance_does_not_overwrite_archived_files(tmp_path, local_store):
    run_dir = write_run(
        tmp_path / "2026-08-31__09-17-11", episode_records(MERIT_LOW)[:1]
    )
    overrides = run_dir / ".hydra/overrides.yaml"
    overrides.write_text("- seed=1\n")
    tracking_uri, experiment_id = local_store
    assert sync_run(experiment_id, run_dir).error is None
    trace_id = logged_traces(tracking_uri)[0].info.trace_id
    overrides.write_text("- seed=2\n")

    result = sync_run(experiment_id, run_dir)

    assert "Source provenance changed: .hydra/overrides.yaml" in result.error
    assert logged_traces(tracking_uri)[0].info.trace_id == trace_id
    client = MlflowClient()
    source = client.search_runs([experiment_id])[0]
    archived = client.download_artifacts(
        source.info.run_id, "source/.hydra/overrides.yaml"
    )
    assert Path(archived).read_text() == "- seed=1\n"


def test_older_trace_format_is_reimported_with_source_records(tmp_path, local_store):
    records = episode_records(MERIT_LOW)[:1]
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", records)
    tracking_uri, experiment_id = local_store
    assert sync_run(experiment_id, run_dir).error is None
    previous = logged_traces(tracking_uri)[0]
    MlflowClient().set_trace_tag(
        previous.info.trace_id, "align.import_complete", "true"
    )

    result = sync_run(experiment_id, run_dir)

    assert result.error is None
    assert result.new_steps == 1
    traces = logged_traces(tracking_uri)
    assert len(traces) == 1
    assert traces[0].info.trace_id != previous.info.trace_id
    assert {
        "input": root_span(traces[0]).inputs,
        **root_span(traces[0]).outputs,
    } == records[0]
