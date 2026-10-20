"""Bring MLflow up to date with open-world run directories."""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from align_utils.open_world import (
    CONFIG_FILE,
    RAW_LOG_FILE,
    find_open_world_runs,
    load_run,
)
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .artifacts import captured_timing, preserve_log
from .evidence import read_decision_logs
from .provenance import preserve_run, read_provenance
from .store import (
    has_session_score,
    log_session_score,
    log_step,
    logged_steps,
    sync_source_version,
)
from .timing import UNKNOWN_TIMING_SOURCE
from .traces import run_key, run_label, session_scores, step_traces

# align-system's Hydra run directories are named by start time.
_RUN_DIR_TIME_FORMAT = "%Y-%m-%d__%H-%M-%S"
_RUN_DIR_NAME = re.compile(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}")
_LOG_TIME = re.compile(r"^\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\]")
_LOG_TIME_FORMAT = "%m/%d/%y %H:%M:%S"


@dataclass(frozen=True)
class SyncResult:
    run_dir: Path
    run_label: Optional[str] = None
    new_steps: int = 0
    new_scores: int = 0
    error: Optional[str] = None
    warning: Optional[str] = None


def run_start_ns(run_dir: Path) -> int:
    """When the run started: its directory name, else its Hydra config's mtime."""
    if _RUN_DIR_NAME.fullmatch(run_dir.name):
        started = datetime.strptime(run_dir.name, _RUN_DIR_TIME_FORMAT)
        return int(started.timestamp()) * 1_000_000_000
    return (run_dir / CONFIG_FILE).stat().st_mtime_ns


def _new_run_start_ns(run_dir: Path, fallback_ns: int) -> int:
    """Prefer a recorded log date over a copied file's modification time."""
    if _RUN_DIR_NAME.fullmatch(run_dir.name):
        return fallback_ns
    try:
        with (run_dir / "align_system.log").open(
            encoding="utf-8", errors="replace"
        ) as source:
            for line in source:
                match = _LOG_TIME.match(line)
                if match is not None:
                    try:
                        started = datetime.strptime(match[1], _LOG_TIME_FORMAT)
                    except ValueError:
                        continue
                    return int(started.timestamp()) * 1_000_000_000
    except OSError:
        pass
    return fallback_ns


def sync_run(
    experiment_id: str, run_dir: Path, *, include_log: bool = False
) -> SyncResult:
    """Log the run's steps and episode scores that MLflow does not have yet.

    Runs that cannot be read yet, for example while input_output.json is
    being rewritten, are reported in ``error`` and change nothing.
    """
    log_warning = None
    try:
        log_content = (run_dir / RAW_LOG_FILE).read_bytes()
    except FileNotFoundError:
        log_content = None
    except OSError as error:
        log_content = None
        log_warning = f"Optional log could not be read: {error}"
    log_text = (
        log_content.decode("utf-8", errors="replace")
        if log_content is not None
        else None
    )
    try:
        run = load_run(run_dir, log_text=log_text)
        provenance = read_provenance(run_dir)
        logs = read_decision_logs(
            run_dir,
            run.records,
            log_text=log_text,
            snapshot_sha256=hashlib.sha256(log_content).hexdigest()
            if log_content is not None
            else None,
        )
        origin_ns = run_start_ns(run_dir)
    except (OSError, ValueError) as error:
        return SyncResult(
            run_dir, error=f"Could not read run; no new data imported: {error}"
        )

    try:
        recovery_origin_ns = _new_run_start_ns(run_dir, origin_ns)
        source_run_id = preserve_run(
            experiment_id,
            run,
            provenance,
            origin_ns=recovery_origin_ns,
        )
        client = MlflowClient()
        source = client.get_run(source_run_id)
        logged = logged_steps(experiment_id, run_key(run))
        timeline, recovery_warning = captured_timing(
            client, source, run, origin_ns, logged, recovery_origin_ns
        )
        if include_log:
            preserve_log(client, source, log_content, logs)
        sync_source_version(logged.values(), run.version)
        trace_ids = {index: trace.info.trace_id for index, trace in logged.items()}
        new_steps = [
            step
            for step in step_traces(run, origin_ns, logs, timeline)
            if step.record_index not in logged
        ]
        for step in new_steps:
            trace_ids[step.record_index] = log_step(experiment_id, source_run_id, step)

        scored = {index for index, trace in logged.items() if has_session_score(trace)}
        new_scores = [
            score
            for score in session_scores(run)
            if score.first_record_index not in scored
        ]
        for score in new_scores:
            log_session_score(trace_ids[score.first_record_index], score)
        MlflowClient().set_terminated(source_run_id)
    except (MlflowException, OSError, ValueError) as error:
        return SyncResult(
            run_dir,
            run_label(run),
            error=f"Import interrupted; partial data may be present: {error}",
        )

    timing_warning = (
        "Some action durations are unknown; synthetic minimum spans are shown."
        if any(
            step.tags["align.timing_source"] == UNKNOWN_TIMING_SOURCE
            for step in new_steps
        )
        else None
    )
    return SyncResult(
        run_dir,
        run_label(run),
        len(new_steps),
        len(new_scores),
        warning="; ".join(
            filter(
                None, [run.score_warning, log_warning, timing_warning, recovery_warning]
            )
        )
        or None,
    )


def sync_tree(
    experiment_id: str, root: Path, *, include_log: bool = False
) -> Tuple[SyncResult, ...]:
    return tuple(
        sync_run(experiment_id, run_dir, include_log=include_log)
        for run_dir in find_open_world_runs(root)
    )
