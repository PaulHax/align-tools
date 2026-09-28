"""Bring MLflow up to date with open-world run directories."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from align_utils.open_world import CONFIG_FILE, find_open_world_runs, load_run
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .evidence import read_decision_logs
from .provenance import preserve_run, read_provenance
from .store import (
    has_session_score,
    log_session_score,
    log_step,
    logged_steps,
    sync_source_version,
)
from .traces import run_key, run_label, session_scores, step_traces

# align-system's Hydra run directories are named by start time.
_RUN_DIR_TIME_FORMAT = "%Y-%m-%d__%H-%M-%S"
_RUN_DIR_NAME = re.compile(r"\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2}")


@dataclass(frozen=True)
class SyncResult:
    run_dir: Path
    run_label: Optional[str] = None
    new_steps: int = 0
    new_scores: int = 0
    error: Optional[str] = None


def run_start_ns(run_dir: Path) -> int:
    """When the run started: its directory name, else its Hydra config's mtime."""
    if _RUN_DIR_NAME.fullmatch(run_dir.name):
        started = datetime.strptime(run_dir.name, _RUN_DIR_TIME_FORMAT)
        return int(started.timestamp()) * 1_000_000_000
    return (run_dir / CONFIG_FILE).stat().st_mtime_ns


def sync_run(experiment_id: str, run_dir: Path) -> SyncResult:
    """Log the run's steps and episode scores that MLflow does not have yet.

    Runs that cannot be read yet, for example while input_output.json is
    being rewritten, are reported in ``error`` and change nothing.
    """
    try:
        run = load_run(run_dir)
        provenance = read_provenance(run_dir)
        logs = read_decision_logs(run_dir, run.records)
    except (OSError, ValueError) as error:
        return SyncResult(run_dir, error=str(error))

    try:
        source_run_id = preserve_run(experiment_id, run, provenance)
        logged = logged_steps(experiment_id, run_key(run))
        sync_source_version(logged.values(), run.version)
        trace_ids = {index: trace.info.trace_id for index, trace in logged.items()}
        new_steps = [
            step
            for step in step_traces(run, run_start_ns(run_dir), logs)
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
        return SyncResult(run_dir, run_label(run), error=str(error))

    return SyncResult(run_dir, run_label(run), len(new_steps), len(new_scores))


def sync_tree(experiment_id: str, root: Path) -> Tuple[SyncResult, ...]:
    return tuple(
        sync_run(experiment_id, run_dir) for run_dir in find_open_world_runs(root)
    )
