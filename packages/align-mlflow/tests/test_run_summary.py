"""Run summaries from persisted episode scores in a real MLflow store."""

import json
import math

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource
from open_world_fixtures import completion_line, episode_records, write_run

from align_mlflow.store import connect
from align_mlflow.summaries import refresh_run_summary
from align_mlflow.sync import sync_run
from align_mlflow.traces import SCORE_ASSESSMENT, SESSION_METADATA_KEY

LOW = "ADEPT-June2025-merit-0.0"
HIGH = "ADEPT-June2025-merit-1.0"
MIDDLE = "ADEPT-June2025-merit-0.4"


@pytest.fixture
def store(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("summary", artifact_location=str(tmp_path / "artifacts"))
    return client, connect(uri, "summary")


def imported_run(client, experiment_id):
    runs = client.search_runs([experiment_id])
    assert len(runs) == 1
    return runs[0]


def test_summary_weights_episodes_equally_and_retains_scores_without_log(
    tmp_path, store
):
    client, eid = store
    source = write_run(
        tmp_path / "source",
        episode_records(LOW)[:1] + episode_records(HIGH) + episode_records(MIDDLE)[:1],
        [completion_line(LOW, 0.0), completion_line(HIGH, 1.0)],
    )
    result = sync_run(eid, source)
    assert result.error is None
    run = imported_run(client, eid)
    assert run.data.metrics == {
        "episode_count": 3.0,
        "scored_episode_count": 2.0,
        "missing_score_count": 1.0,
        "target_mismatch_count": 0.0,
        "ta3_alignment_mean": 0.5,
    }
    histories = {
        key: client.get_metric_history(run.info.run_id, key) for key in run.data.metrics
    }

    (source / "raw_align_system.log").unlink()
    assert sync_run(eid, source).error is None
    assert imported_run(client, eid).data.metrics == run.data.metrics
    for key, history in histories.items():
        assert client.get_metric_history(run.info.run_id, key) == history


def test_summary_adds_late_scores_and_appended_episodes(tmp_path, store):
    client, eid = store
    source = write_run(tmp_path / "source", episode_records(LOW)[:1])
    assert sync_run(eid, source).error is None
    run = imported_run(client, eid)
    assert run.data.metrics == {
        "episode_count": 1.0,
        "scored_episode_count": 0.0,
        "missing_score_count": 1.0,
        "target_mismatch_count": 0.0,
    }

    (source / "raw_align_system.log").write_text(completion_line(LOW, 0.0))
    assert sync_run(eid, source).error is None
    assert imported_run(client, eid).data.metrics["ta3_alignment_mean"] == 0.0

    (source / "input_output.json").write_text(
        json.dumps(episode_records(LOW)[:1] + episode_records(HIGH)[:1])
    )
    (source / "raw_align_system.log").write_text(
        completion_line(LOW, 0.0) + completion_line(HIGH, 1.0)
    )
    assert sync_run(eid, source).error is None
    metrics = imported_run(client, eid).data.metrics
    assert metrics["ta3_alignment_mean"] == 0.5
    assert metrics["episode_count"] == metrics["scored_episode_count"] == 2.0
    assert metrics["missing_score_count"] == 0.0
    history = client.get_metric_history(run.info.run_id, "ta3_alignment_mean")
    assert len(history) == 2
    assert sync_run(eid, source).error is None
    assert client.get_metric_history(run.info.run_id, "ta3_alignment_mean") == history


def test_summary_exposes_evaluated_target_mismatch_but_not_unaligned(tmp_path, store):
    client, eid = store
    source = write_run(
        tmp_path / "source", episode_records(LOW)[:1] + episode_records(None)[:1]
    )
    assert sync_run(eid, source).error is None
    run = imported_run(client, eid)
    traces = client.search_traces(locations=[eid], include_spans=False)
    for trace in traces:
        mlflow.log_feedback(
            trace_id=trace.info.trace_id,
            name=SCORE_ASSESSMENT,
            value=0.25,
            source=AssessmentSource(
                source_type="CODE", source_id="raw_align_system.log"
            ),
            metadata={
                SESSION_METADATA_KEY: trace.info.trace_metadata[SESSION_METADATA_KEY],
                "alignment_target_id": HIGH,
            },
        )
        mlflow.log_feedback(
            trace_id=trace.info.trace_id,
            name=SCORE_ASSESSMENT,
            value=1.0,
            source=AssessmentSource(source_type="HUMAN", source_id="reviewer"),
        )
    metrics = refresh_run_summary(client, run.info.run_id)
    assert metrics["target_mismatch_count"] == 1.0
    assert metrics["scored_episode_count"] == 2.0
    assert metrics["ta3_alignment_mean"] == 0.25


def test_removing_last_score_clears_previous_mean_without_repeating_history(
    tmp_path, store
):
    client, eid = store
    source = write_run(
        tmp_path / "source", episode_records(LOW)[:1], [completion_line(LOW, 0.75)]
    )
    assert sync_run(eid, source).error is None
    run = imported_run(client, eid)
    trace = client.search_traces(locations=[eid], include_spans=False)[0]
    assessment = trace.info.assessments[0]
    mlflow.delete_assessment(trace.info.trace_id, assessment.assessment_id)

    metrics = refresh_run_summary(client, run.info.run_id)
    assert metrics["scored_episode_count"] == 0.0
    assert metrics["missing_score_count"] == 1.0
    assert math.isnan(metrics["ta3_alignment_mean"])
    assert math.isnan(imported_run(client, eid).data.metrics["ta3_alignment_mean"])
    history = client.get_metric_history(run.info.run_id, "ta3_alignment_mean")
    assert len(history) == 2
    refresh_run_summary(client, run.info.run_id)
    assert len(client.get_metric_history(run.info.run_id, "ta3_alignment_mean")) == 2
