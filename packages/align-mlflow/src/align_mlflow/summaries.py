"""Summarize recorded episode outcomes on their source MLflow run."""

import math
import time
from typing import Dict, Iterable, cast

from mlflow import MlflowClient
from mlflow.entities import Metric, Trace

from .assessments import is_recorded_session_score
from .traces import SESSION_METADATA_KEY


def episode_metrics(traces: Iterable[Trace]) -> Dict[str, float]:
    """Weight each episode once, retaining missing and mismatched outcomes."""
    episodes = {}
    for trace in traces:
        session = trace.info.trace_metadata.get(SESSION_METADATA_KEY)
        if not session:
            raise ValueError(f"Episode trace {trace.info.trace_id} has no session")
        if session in episodes:
            raise ValueError(f"Multiple first-action traces for session {session}")
        episodes[session] = trace

    values = []
    mismatches = 0
    for session, trace in sorted(episodes.items()):
        scores = [
            assessment
            for assessment in trace.info.assessments or []
            if is_recorded_session_score(assessment, session)
            and assessment.valid is not False
        ]
        if not scores:
            continue
        score = max(
            scores,
            key=lambda assessment: (
                assessment.last_update_time_ms,
                assessment.assessment_id or "",
            ),
        )
        values.append(float(cast(float, score.value)))
        requested = trace.info.tags.get("alignment_target_id")
        evaluated = (score.metadata or {}).get("alignment_target_id")
        mismatches += bool(requested and evaluated and requested != evaluated)

    metrics = {
        "episode_count": float(len(episodes)),
        "scored_episode_count": float(len(values)),
        "missing_score_count": float(len(episodes) - len(values)),
        "target_mismatch_count": float(mismatches),
    }
    if values:
        metrics["ta3_alignment_mean"] = math.fsum(values) / len(values)
    return metrics


def refresh_run_summary(client: MlflowClient, source_run_id: str) -> Dict[str, float]:
    """Refresh metrics from persisted scores, including when source logs disappear.

    The latest valid TA3 assessment supplies one score per episode. Trace-level
    reviews and feedback from other sources do not change this summary.
    """
    source = client.get_run(source_run_id)
    traces = []
    token = None
    while True:
        page = client.search_traces(
            locations=[source.info.experiment_id],
            filter_string=(
                f"run_id = '{source_run_id}' AND tags.step = '1' "
                "AND tags.`align.import_complete` = 'source-record-v1'"
            ),
            include_spans=False,
            max_results=500,
            page_token=token,
        )
        traces.extend(page)
        token = page.token
        if not token:
            break
    metrics = episode_metrics(traces)
    if (
        "ta3_alignment_mean" not in metrics
        and "ta3_alignment_mean" in source.data.metrics
    ):
        # Metric history cannot be deleted through the tracking API. NaN clears
        # the current mean when every previously scored outcome is removed.
        metrics["ta3_alignment_mean"] = math.nan
    changed = {
        key: value
        for key, value in metrics.items()
        if source.data.metrics.get(key) != value
        and not (math.isnan(value) and math.isnan(source.data.metrics.get(key, 0.0)))
    }
    if changed:
        previous_times = [
            metric.timestamp
            for key in changed
            if key in source.data.metrics
            for metric in client.get_metric_history(source_run_id, key)
        ]
        timestamp = max(time.time_ns() // 1_000_000, max(previous_times, default=0) + 1)
        client.log_batch(
            source_run_id,
            metrics=[
                Metric(key, value, timestamp, 0) for key, value in changed.items()
            ],
        )
    return metrics
