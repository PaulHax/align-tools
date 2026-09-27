"""MLflow side effects: find what is already logged, log steps and scores."""

import os
from typing import Dict, Iterator, List, Optional

import mlflow
from align_utils.open_world import RAW_LOG_FILE
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource, Trace
from mlflow.exceptions import MlflowException

from .traces import (
    RECORD_INDEX_TAG,
    RUN_KEY_TAG,
    SCORE_ASSESSMENT,
    SESSION_METADATA_KEY,
    SessionScore,
    StepTrace,
)

_PAGE_SIZE = 500
_COMPLETE_TAG = "align.import_complete"


def connect(tracking_uri: str, experiment_name: str) -> str:
    """Point MLflow at tracking_uri and return the experiment id.

    Trace export is synchronous so log_step can verify the persisted result.
    Call this before creating any trace, since MLflow reads the setting once.
    """
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment_name).experiment_id


def _run_traces(experiment_id: str, run_key: str) -> Iterator[Trace]:
    client = MlflowClient()
    page_token: Optional[str] = None
    while True:
        page = client.search_traces(
            locations=[experiment_id],
            filter_string=f"tags.`{RUN_KEY_TAG}` = '{run_key}'",
            max_results=_PAGE_SIZE,
            page_token=page_token,
            include_spans=False,
        )
        yield from page
        page_token = page.token
        if not page_token:
            return


def logged_steps(experiment_id: str, run_key: str) -> Dict[int, Trace]:
    """Confirmed imports by record index, removing interrupted import attempts."""
    traces = list(_run_traces(experiment_id, run_key))
    incomplete = [
        trace.info.trace_id
        for trace in traces
        if trace.info.tags.get(_COMPLETE_TAG) != "true"
    ]
    if incomplete:
        _delete_incomplete(experiment_id, incomplete)
    return {
        int(trace.info.tags[RECORD_INDEX_TAG]): trace
        for trace in traces
        if trace.info.tags.get(_COMPLETE_TAG) == "true"
    }


def _delete_incomplete(experiment_id: str, trace_ids: List[str]) -> None:
    deleted = MlflowClient().delete_traces(
        experiment_id=experiment_id, trace_ids=trace_ids
    )
    if deleted != len(trace_ids):
        raise MlflowException(
            f"Could not remove incomplete imports {trace_ids}; retry sync"
        )


def has_session_score(trace: Trace) -> bool:
    return any(a.name == SCORE_ASSESSMENT for a in trace.info.assessments or [])


def log_step(experiment_id: str, step: StepTrace) -> str:
    root = mlflow.start_span_no_context(
        name=step.root.name,
        span_type=step.root.span_type,
        inputs=step.root.inputs,
        attributes=dict(step.root.attributes),
        tags=dict(step.tags),
        metadata={SESSION_METADATA_KEY: step.session_id},
        experiment_id=experiment_id,
        start_time_ns=step.root.start_ns,
    )
    span_ids = {root.span_id}
    for child in step.children:
        span = mlflow.start_span_no_context(
            name=child.name,
            span_type=child.span_type,
            parent_span=root,
            inputs=child.inputs,
            attributes=dict(child.attributes),
            start_time_ns=child.start_ns,
        )
        span.end(outputs=child.outputs, end_time_ns=child.end_ns)
        span_ids.add(span.span_id)
    root.end(outputs=step.root.outputs, end_time_ns=step.root.end_ns)

    # MLflow suppresses exporter exceptions, even with synchronous export.
    # Search loads persisted spans instead of consulting the active trace cache.
    client = MlflowClient()
    query = {
        "locations": [experiment_id],
        "filter_string": f"request_id = '{root.trace_id}'",
        "max_results": 1,
    }
    persisted = client.search_traces(**query, include_spans=True)
    if (
        not persisted
        or {span.span_id for span in persisted[0].data.spans} != span_ids
        or any(
            persisted[0].info.tags.get(key) != value for key, value in step.tags.items()
        )
        or persisted[0].info.trace_metadata.get(SESSION_METADATA_KEY) != step.session_id
    ):
        unconfirmed = client.search_traces(**query, include_spans=False)
        if unconfirmed and (
            unconfirmed[0].info.tags.get(RUN_KEY_TAG) != step.tags[RUN_KEY_TAG]
        ):
            # Without the run tag, the next sync cannot discover this attempt.
            _delete_incomplete(experiment_id, [root.trace_id])
        raise MlflowException(
            f"Trace {root.trace_id} was not fully persisted; retry sync"
        )
    client.set_trace_tag(root.trace_id, _COMPLETE_TAG, "true")
    confirmed = client.search_traces(**query, include_spans=False)
    if not confirmed or confirmed[0].info.tags.get(_COMPLETE_TAG) != "true":
        raise MlflowException(
            f"Trace {root.trace_id} import was not confirmed; retry sync"
        )
    return root.trace_id


def log_session_score(trace_id: str, score: SessionScore) -> None:
    """Log TA3's episode score the way MLflow stores session-level feedback:
    on the session's first trace, with the session id in its metadata."""
    metadata = {SESSION_METADATA_KEY: score.session_id}
    if score.alignment_target_id:
        metadata["alignment_target_id"] = score.alignment_target_id
    mlflow.log_feedback(
        trace_id=trace_id,
        name=SCORE_ASSESSMENT,
        value=score.value,
        source=AssessmentSource(source_type="CODE", source_id=RAW_LOG_FILE),
        rationale=score.rationale,
        metadata=metadata,
    )
