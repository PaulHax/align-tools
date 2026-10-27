"""Captured timing and available log evidence, separate from launch provenance."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Sequence

from align_utils.open_world import RAW_LOG_FILE, OpenWorldRun
from mlflow import MlflowClient
from mlflow.entities import Run, Trace
from mlflow.exceptions import MlflowException

from .evidence import DecisionLog
from .timing import STEP_GAP_NS, TIMELINE_TAG, TimingPlan, resolve_timing_plan

LOG_SNAPSHOT_TAG = "align.log_snapshot"


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def timing_manifest(plan: TimingPlan) -> dict:
    digest = text_digest(plan.to_json())
    return {
        "artifact_path": f"evidence/timing/{digest}.json",
        "sha256": digest,
        "basis": plan.basis,
        "origin_ns": plan.origin_ns,
    }


def _upload_text(client: MlflowClient, run_id: str, path: str, text: str) -> None:
    _upload_bytes(client, run_id, path, text.encode("utf-8"))


def _upload_bytes(client: MlflowClient, run_id: str, path: str, content: bytes) -> None:
    with TemporaryDirectory() as staging:
        artifact = Path(path)
        local = Path(staging) / artifact.name
        local.write_bytes(content)
        client.log_artifact(
            run_id, str(local), artifact_path=artifact.parent.as_posix()
        )


def captured_timing(
    client: MlflowClient,
    source: Run,
    run: OpenWorldRun,
    origin_ns: int,
    logged: Mapping[int, Trace],
    recovery_origin_ns: int,
) -> tuple[TimingPlan, Optional[str]]:
    """Reuse the captured plan; an older source run retains its prior timeline."""
    value = source.data.tags.get(TIMELINE_TAG)
    if value is None:
        return resolve_timing_plan(run, origin_ns, legacy=True), None
    captured: Optional[TimingPlan] = None
    try:
        captured = _read_timing(client, source.info.run_id, run, value)
    except (KeyError, TypeError, ValueError, OSError, MlflowException):
        resume_index = next(
            (index for index in range(len(run.records)) if index not in logged),
            len(run.records),
        )
        origin = recovery_origin_ns
        if logged:
            latest = max(logged.values(), key=lambda trace: trace.info.timestamp_ms)
            trace = client.get_trace(latest.info.trace_id)
            root = next(span for span in trace.data.spans if span.parent_id is None)
            origin = root.end_time_ns + STEP_GAP_NS
        plan = TimingPlan(
            "raw_times_s", origin, (None,) * len(run.records), resume_index
        )
        warning: Optional[str] = (
            "Captured timing was unavailable; earlier traces were retained and "
            "new action durations are unknown."
        )
    else:
        plan = resolve_timing_plan(run, origin_ns, existing=captured)
        warning = None
    if plan != captured:
        preserve_timing(client, source.info.run_id, plan)
        client.set_tag(
            source.info.run_id,
            TIMELINE_TAG,
            json.dumps(timing_manifest(plan), sort_keys=True),
        )
    return plan, warning


def _read_timing(
    client: MlflowClient, run_id: str, run: OpenWorldRun, value: str
) -> TimingPlan:
    manifest = json.loads(value)
    digest, path = manifest["sha256"], manifest["artifact_path"]
    if path != f"evidence/timing/{digest}.json":
        raise ValueError("Invalid timing artifact path")
    available = client.list_artifacts(run_id, str(Path(path).parent))
    if any(item.path == path for item in available):
        with TemporaryDirectory() as destination:
            local = client.download_artifacts(run_id, path, destination)
            text = Path(local).read_text(encoding="utf-8")
        if text_digest(text) == digest:
            return TimingPlan.from_json(text)
    candidate = resolve_timing_plan(
        run,
        manifest["origin_ns"],
        legacy=manifest["basis"] == "component_durations",
    )
    if timing_manifest(candidate) != manifest:
        raise ValueError("Original captured timing is unavailable")
    preserve_timing(client, run_id, candidate)
    return candidate


def preserve_timing(client: MlflowClient, run_id: str, plan: TimingPlan) -> None:
    manifest = timing_manifest(plan)
    _upload_text(client, run_id, manifest["artifact_path"], plan.to_json())


def preserve_log(
    client: MlflowClient,
    source: Run,
    content: Optional[bytes],
    logs: Sequence[DecisionLog],
) -> None:
    """Retain a log snapshot once, with its saved-action association results."""
    if content is None:
        return
    digest = hashlib.sha256(content).hexdigest()
    prefix = f"evidence/log/{digest}"
    index = {
        "snapshot_sha256": digest,
        "scope": "saved_action_records",
        "records": [
            {
                "record_index": i,
                "status": log.status,
                **(
                    {"skipped_rejected_attempts": log.skipped_rejected_attempts}
                    if log.skipped_rejected_attempts
                    else {}
                ),
                **({"source": log.excerpt.reference()} if log.excerpt else {}),
            }
            for i, log in enumerate(logs)
        ],
    }
    index_text = json.dumps(index, sort_keys=True)
    manifest = {
        "artifact_path": f"{prefix}/{RAW_LOG_FILE}",
        "index_path": f"{prefix}/index-{text_digest(index_text)}.json",
        "sha256": digest,
    }
    previous = json.loads(source.data.tags.get(LOG_SNAPSHOT_TAG, "{}"))
    if previous == manifest:
        return
    if previous.get("sha256") != digest:
        _upload_bytes(client, source.info.run_id, manifest["artifact_path"], content)
    _upload_text(client, source.info.run_id, manifest["index_path"], index_text)
    client.set_tag(
        source.info.run_id, LOG_SNAPSHOT_TAG, json.dumps(manifest, sort_keys=True)
    )
