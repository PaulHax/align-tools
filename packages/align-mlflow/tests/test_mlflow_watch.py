"""Detecting runs that align-system has written to, and retrying failed syncs."""

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.tracing.client import TracingClient
from open_world_fixtures import SCENARIO, completion_line, episode_records, write_run

import align_mlflow.watch as watch_module
from align_mlflow.store import connect
from align_mlflow.sync import SyncResult, sync_run

TARGET = "ADEPT-June2025-merit-0.0"


@pytest.fixture
def experiment_id(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    MlflowClient(tracking_uri=tracking_uri).create_experiment(
        "watch test", artifact_location=str(tmp_path / "artifacts")
    )
    return connect(tracking_uri, "watch test")


def test_changed_runs_reports_runs_whose_files_changed(tmp_path):
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", episode_records(TARGET)[:3])

    first = watch_module.changed_runs(tmp_path, {})
    assert list(first) == [run_dir]
    assert watch_module.changed_runs(tmp_path, first) == {}

    write_run(run_dir, episode_records(TARGET)[:4])
    grown = watch_module.changed_runs(tmp_path, first)
    assert list(grown) == [run_dir]

    (run_dir / f"{SCENARIO}.{TARGET}.final_state_unstructured.json").write_text(
        "done\n"
    )
    assert list(watch_module.changed_runs(tmp_path, grown)) == [run_dir]


def test_watch_retries_failed_syncs_and_reports_each_error_once(tmp_path, monkeypatch):
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", episode_records(TARGET))
    results = iter(
        [
            SyncResult(run_dir, error="mid-write"),
            SyncResult(run_dir, error="mid-write"),
            SyncResult(run_dir, "run", new_steps=7),
        ]
    )
    remaining_sleeps = iter(range(2))

    def sleep(_seconds):
        if next(remaining_sleeps, None) is None:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_module.time, "sleep", sleep)
    synced, reported = [], []

    def sync(path):
        synced.append(path)
        return next(results)

    with pytest.raises(KeyboardInterrupt):
        watch_module.watch(tmp_path, sync, reported.append, interval_s=0)

    assert synced == [run_dir] * 3
    assert [result.error for result in reported] == ["mid-write", None]


def test_watch_logs_score_when_only_the_raw_log_changes(
    tmp_path, monkeypatch, experiment_id
):
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", episode_records(TARGET)[:1])
    input_stat = (run_dir / "input_output.json").stat()
    reported = []
    polls = iter(range(2))

    def next_poll(_seconds):
        if next(polls) == 0:
            with (run_dir / "raw_align_system.log").open("a") as log:
                log.write(completion_line(TARGET, 0.83))
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch_module.time, "sleep", next_poll)

    with pytest.raises(KeyboardInterrupt):
        watch_module.watch(
            tmp_path,
            lambda path: sync_run(experiment_id, path),
            reported.append,
            interval_s=0,
        )

    assert [(r.new_steps, r.new_scores, r.error) for r in reported] == [
        (1, 0, None),
        (0, 1, None),
    ]
    final_stat = (run_dir / "input_output.json").stat()
    assert (final_stat.st_mtime_ns, final_stat.st_size) == (
        input_stat.st_mtime_ns,
        input_stat.st_size,
    )
    assert not (run_dir / "timing.json").exists()
    assert not list(run_dir.glob("*.final_state_unstructured.json"))
    traces = MlflowClient().search_traces(locations=[experiment_id])
    assert len(traces) == 1
    assert traces[0].info.assessments[0].value == 0.83


def test_watch_retries_failed_export_without_a_source_change(
    tmp_path, monkeypatch, experiment_id
):
    run_dir = write_run(tmp_path / "2026-08-31__09-17-11", episode_records(TARGET)[:1])
    signature = watch_module.run_signature(run_dir)
    log_spans, start_trace = TracingClient.log_spans, TracingClient.start_trace
    reported = []
    polls = iter(range(2))

    def failed_spans(*args, **kwargs):
        raise MlflowException("span export unavailable")

    def failed_trace(*args, **kwargs):
        raise MlflowException("trace export unavailable")

    def next_poll(_seconds):
        if next(polls) == 0:
            monkeypatch.setattr(TracingClient, "log_spans", log_spans)
            monkeypatch.setattr(TracingClient, "start_trace", start_trace)
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(TracingClient, "log_spans", failed_spans)
    monkeypatch.setattr(TracingClient, "start_trace", failed_trace)
    monkeypatch.setattr(watch_module.time, "sleep", next_poll)

    with pytest.raises(KeyboardInterrupt):
        watch_module.watch(
            tmp_path,
            lambda path: sync_run(experiment_id, path),
            reported.append,
            interval_s=0,
        )

    assert len(reported) == 2
    assert reported[0].error is not None
    assert reported[1].error is None
    assert reported[1].new_steps == 1
    assert watch_module.run_signature(run_dir) == signature
    traces = MlflowClient().search_traces(locations=[experiment_id])
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 4
