"""Exercise generic Open World imports through the CLI and persisted MLflow data."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource
from open_world_fixtures import completion_line, episode_records, write_run

from align_mlflow.provenance import preserve_run, read_provenance
from align_mlflow.store import connect, log_step
from align_mlflow.traces import step_traces
from align_utils.open_world import load_run

EXPERIMENT = "generic open world"
TARGET = "ADEPT-June2025-merit-0.0"
DRIVER = "align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver"
MODEL = "Qwen/Qwen3.8-27B-FP8"
UNKNOWN_EVIDENCE = {
    "note": "unrecognized decision evidence",
    "nested": [None, False, 1, {"unicode": "café"}],
}


def generic_records():
    records = episode_records(TARGET)[:2]
    for record in records:
        record["choice_info"] = {
            "decision": {"reasoning": "Review the situation before choosing."},
            "unknown_nested": copy.deepcopy(UNKNOWN_EVIDENCE),
            "optional_value": None,
        }
    records[1]["choice_info"]["reviewer_notes"] = [
        {"reasoning": "Keep the proposed action.", "accepted": True}
    ]
    return records


def generic_run(path, records, architecture=None):
    run = write_run(path, records, [completion_line(TARGET, 0.83)], driver=DRIVER)
    config = {
        "name": "action_based",
        "driver": {
            "_target_": DRIVER,
            "model": None,
            "chat_model": {"model": MODEL},
            "extra_tools": {
                "kaleido": {
                    "_target_": "example.tools.KaleidoTool",
                    "auto_assess": False,
                    "explain": False,
                }
            },
        },
        "interface": {"adm_profile": "TEST_OPENWORLD"},
    }
    metadata = {
        "version": "0.5.11",
        "username": "generic-open-world-session",
        "driver": "langchain_agent",
    }
    if architecture is not None:
        metadata["architecture"] = architecture
        config["driver"]["reviewer"] = {
            "_target_": "example.review.DecisionReviewer",
            "chat_model": None,
            "weighting": {"influence": 0.5, "swap_threshold": 0.15},
        }
    (run / ".hydra/config.yaml").write_text(yaml.safe_dump(config))
    (run / "meta.json").write_text(json.dumps(metadata))
    return run


def write_timing(run, durations):
    (run / "timing.json").write_text(
        json.dumps({"scenarios": [], "raw_times_s": durations})
    )


def create_store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return uri, client, experiment_id


def command(run, tracking_uri):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "align_mlflow",
            "sync",
            str(run),
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            EXPERIMENT,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )


def read_traces(client, experiment_id):
    return sorted(
        client.search_traces(locations=[experiment_id]),
        key=lambda trace: int(trace.info.tags["align.record_index"]),
    )


def root_span(trace):
    return next(span for span in trace.data.spans if span.parent_id is None)


def read_timing_plan(client, source_run, destination):
    reference = json.loads(source_run.data.tags["align.timeline"])
    destination.mkdir()
    artifact = Path(
        client.download_artifacts(
            source_run.info.run_id, reference["artifact_path"], str(destination)
        )
    )
    data = artifact.read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference["sha256"]
    return json.loads(data)


@pytest.mark.parametrize(
    ("architecture", "expected_adm"),
    [
        (None, "langchain_agent + kaleido"),
        ("decision_agent_reviewer", "decision_agent_reviewer"),
    ],
)
def test_cli_imports_generic_metadata_timings_and_inspectable_evidence(
    tmp_path, architecture, expected_adm
):
    records = generic_records()
    run = generic_run(tmp_path / "2026-09-24__16-05-25", records, architecture)
    write_timing(run, [1.25, 2.5])
    original_files = {
        name: (run / name).read_bytes()
        for name in (
            ".hydra/config.yaml",
            "meta.json",
            "input_output.json",
            "timing.json",
        )
    }
    uri, client, experiment_id = create_store(tmp_path)

    result = command(run, uri)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 new traces, 1 new episode scores" in result.stdout
    traces = read_traces(client, experiment_id)
    assert len(traces) == 2
    source_runs = client.search_runs([experiment_id])
    assert len(source_runs) == 1
    source_run = source_runs[0]
    for tags in [source_run.data.tags, *(trace.info.tags for trace in traces)]:
        assert tags["adm"] == expected_adm
        assert tags["llm"] == MODEL
        assert tags["align.driver"] == DRIVER
    assert {trace.info.trace_metadata["mlflow.sourceRun"] for trace in traces} == {
        source_run.info.run_id
    }
    assert traces[0].info.assessments[0].value == 0.83
    for trace, record, duration in zip(traces, records, [1.25, 2.5]):
        root = root_span(trace)
        assert len(trace.data.spans) == 1
        assert (root.end_time_ns - root.start_time_ns) / 1e9 == duration
        assert root.attributes["align.timing_source"] == "timing.json.raw_times_s"
        assert root.attributes["align.timeline_type"] == "synthetic"
        assert {"input": root.inputs["source"], **root.outputs["source"]} == record
        evidence = root.outputs["evidence"]["choice_info"]
        assert evidence["source"] == "source.choice_info"
        assert evidence["applies_to_action"] is True
        assert evidence["fields"]["unknown_nested"]["type"] == "object"
        assert (
            json.loads(evidence["fields"]["unknown_nested"]["preview"])
            == UNKNOWN_EVIDENCE
        )
        assert evidence["fields"]["optional_value"]["type"] == "null"
    assert root_span(traces[1]).start_time_ns > root_span(traces[0]).end_time_ns
    plan = read_timing_plan(client, source_run, tmp_path / "timing-download")
    assert plan["basis"] == "raw_times_s"
    assert plan["raw_times_s"] == [1.25, 2.5]

    mlflow.set_tracking_uri(uri)
    mlflow.log_feedback(
        trace_id=traces[0].info.trace_id,
        name="human_review",
        value=True,
        source=AssessmentSource(source_type="HUMAN", source_id="reviewer"),
        rationale="Retain the reviewed decision.",
        metadata={
            "mlflow.trace.session": traces[0].info.trace_metadata[
                "mlflow.trace.session"
            ]
        },
    )
    before = [trace.to_dict() for trace in read_traces(client, experiment_id)]
    copied = tmp_path / "copied" / run.name
    shutil.copytree(run, copied)
    write_timing(copied, [50.0, 60.0])

    repeated = command(copied, uri)

    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert "0 new traces, 0 new episode scores" in repeated.stdout
    assert [trace.to_dict() for trace in read_traces(client, experiment_id)] == before
    repeated_runs = client.search_runs([experiment_id])
    assert len(repeated_runs) == 1
    assert repeated_runs[0].info.run_id == source_run.info.run_id
    assert (
        repeated_runs[0].data.tags["align.timeline"]
        == source_run.data.tags["align.timeline"]
    )
    for name, content in original_files.items():
        assert (run / name).read_bytes() == content


def test_late_timing_file_preserves_imported_trace_and_append_timeline(tmp_path):
    records = generic_records()
    run = generic_run(tmp_path / "2026-09-24__16-05-25", records[:1])
    uri, client, experiment_id = create_store(tmp_path)
    first_import = command(run, uri)
    assert first_import.returncode == 0, first_import.stdout + first_import.stderr
    before = read_traces(client, experiment_id)[0]
    source_run = client.search_runs([experiment_id])[0]

    (run / "input_output.json").write_text(json.dumps(records))
    write_timing(run, [1.25, 2.5])
    appended = command(run, uri)

    assert appended.returncode == 0, appended.stdout + appended.stderr
    assert "1 new traces, 0 new episode scores" in appended.stdout
    traces = read_traces(client, experiment_id)
    assert len(traces) == 2
    assert traces[0].to_dict() == before.to_dict()
    assert {
        root_span(trace).end_time_ns - root_span(trace).start_time_ns
        for trace in traces
    } == {1_000}
    assert {root_span(trace).attributes["align.timing_source"] for trace in traces} == {
        "synthetic_minimum"
    }
    assert root_span(traces[1]).start_time_ns > root_span(traces[0]).end_time_ns
    source_after = client.search_runs([experiment_id])[0]
    assert source_after.info.run_id == source_run.info.run_id
    assert (
        source_after.data.tags["align.timeline"]
        == source_run.data.tags["align.timeline"]
    )
    plan = read_timing_plan(client, source_after, tmp_path / "timing-download")
    assert plan["basis"] == "component_durations"


@pytest.mark.parametrize("name", ["3813", "2026-09-24__16-05-25"])
def test_recorded_start_date_survives_download_and_copied_rerun(tmp_path, name):
    run = generic_run(tmp_path / name, generic_records())
    write_timing(run, [1.25, 2.5])
    log = run / "align_system.log"
    log.write_text(
        "Undated diagnostic\n"
        "[99/99/99 14:55:28] invalid date\n"
        "[09/18/26 14:55:28] INFO Recorded startup\n"
        "[09/19/26 12:00:00] INFO Later message\n"
    )
    copied_mtime = int(datetime(2026, 10, 19, 20, 0).timestamp()) * 1_000_000_000
    config = run / ".hydra/config.yaml"
    os.utime(config, ns=(copied_mtime, copied_mtime))
    original_files = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }
    expected_start = (
        int(
            (
                datetime(2026, 9, 18, 14, 55, 28)
                if name == "3813"
                else datetime(2026, 9, 24, 16, 5, 25)
            ).timestamp()
        )
        * 1_000_000_000
    )
    uri, client, experiment_id = create_store(tmp_path)

    imported = command(run, uri)

    assert imported.returncode == 0, imported.stdout + imported.stderr
    traces = read_traces(client, experiment_id)
    assert [root_span(trace).start_time_ns for trace in traces] == [
        expected_start,
        expected_start + 1_251_000_000,
    ]
    assert all(
        root_span(trace).attributes["align.timeline_type"] == "synthetic"
        for trace in traces
    )
    source = client.search_runs([experiment_id])[0]
    plan = read_timing_plan(client, source, tmp_path / "timing-before")
    assert plan["origin_ns"] == expected_start
    assert config.stat().st_mtime_ns == copied_mtime
    assert {
        path: (run / path).read_bytes() for path in original_files
    } == original_files

    copied = tmp_path / "copied" / name
    shutil.copytree(run, copied)
    (copied / "align_system.log").write_text(
        "[10/19/27 20:00:00] INFO Different copied log\n"
    )
    later_mtime = copied_mtime + 86_400_000_000_000
    os.utime(copied / ".hydra/config.yaml", ns=(later_mtime, later_mtime))

    repeated = command(copied, uri)

    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert "0 new traces, 0 new episode scores" in repeated.stdout
    assert [trace.to_dict() for trace in read_traces(client, experiment_id)] == [
        trace.to_dict() for trace in traces
    ]
    sources = client.search_runs([experiment_id])
    assert len(sources) == 1
    assert sources[0].info.run_id == source.info.run_id
    assert read_timing_plan(client, sources[0], tmp_path / "timing-after") == plan


@pytest.mark.parametrize("log_kind", ["missing", "unrecognized", "unreadable"])
def test_optional_start_log_uses_config_time_when_unavailable(tmp_path, log_kind):
    run = generic_run(tmp_path / "3813", generic_records())
    log = run / "align_system.log"
    if log_kind == "unrecognized":
        log.write_text("Unrecognized startup diagnostic\n")
    elif log_kind == "unreadable":
        log.mkdir()
    origin = int(datetime(2026, 9, 18, 14, 55, 28).timestamp()) * 1_000_000_000
    os.utime(run / ".hydra/config.yaml", ns=(origin, origin))
    uri, client, experiment_id = create_store(tmp_path)

    imported = command(run, uri)

    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert root_span(read_traces(client, experiment_id)[0]).start_time_ns == origin


def test_legacy_numeric_run_without_plan_keeps_config_origin_when_appending(tmp_path):
    records = generic_records()
    run = generic_run(tmp_path / "3813", records[:1])
    (run / "align_system.log").write_text("[09/18/26 14:55:28] INFO Recorded startup\n")
    origin = int(datetime(2026, 10, 19, 20, 0).timestamp()) * 1_000_000_000
    os.utime(run / ".hydra/config.yaml", ns=(origin, origin))
    uri, client, experiment_id = create_store(tmp_path)
    connect(uri, EXPERIMENT)
    loaded = load_run(run)
    source_id = preserve_run(experiment_id, loaded, read_provenance(run))
    first_id = log_step(experiment_id, source_id, step_traces(loaded, origin)[0])
    first_root = root_span(client.get_trace(first_id)).to_dict()
    assert "align.timeline" not in client.get_run(source_id).data.tags
    (run / "input_output.json").write_text(json.dumps(records))
    write_timing(run, [1.25, 2.5])

    appended = command(run, uri)

    assert appended.returncode == 0, appended.stdout + appended.stderr
    assert "1 new traces" in appended.stdout
    traces = read_traces(client, experiment_id)
    assert len(traces) == 2
    assert [root_span(trace).start_time_ns for trace in traces] == [
        origin,
        origin + 1_001_000,
    ]
    assert root_span(traces[0]).to_dict() == first_root
    assert "align.timeline" not in client.get_run(source_id).data.tags
