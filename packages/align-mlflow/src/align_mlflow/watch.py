"""Keep MLflow in sync while align-system writes open-world runs."""

import time
from pathlib import Path
from typing import Callable, Dict, Tuple

from align_utils.open_world import (
    FINAL_STATE_GLOB,
    INPUT_OUTPUT_FILE,
    RAW_LOG_FILE,
    TIMING_FILE,
    find_open_world_runs,
)

from .provenance import PROVENANCE_FILES
from .sync import SyncResult

Signature = Tuple[Tuple[str, int, int], ...]


def run_signature(run_dir: Path) -> Signature:
    """Path, mtime and size of files that supply traces, scores or provenance."""
    paths = [
        run_dir / INPUT_OUTPUT_FILE,
        run_dir / RAW_LOG_FILE,
        run_dir / TIMING_FILE,
        *(run_dir / path for path in PROVENANCE_FILES),
        *sorted(run_dir.glob(FINAL_STATE_GLOB)),
    ]
    stats = [
        (path.relative_to(run_dir).as_posix(), path.stat())
        for path in paths
        if path.is_file()
    ]
    return tuple((name, stat.st_mtime_ns, stat.st_size) for name, stat in stats)


def changed_runs(root: Path, seen: Dict[Path, Signature]) -> Dict[Path, Signature]:
    """Runs under root whose signature differs from ``seen``, with the new one."""
    current = {
        run_dir: run_signature(run_dir) for run_dir in find_open_world_runs(root)
    }
    return {
        run_dir: signature
        for run_dir, signature in current.items()
        if seen.get(run_dir) != signature
    }


def watch(
    root: Path,
    sync: Callable[[Path], SyncResult],
    report: Callable[[SyncResult], None],
    interval_s: float,
) -> None:
    """Sync changed runs every ``interval_s`` seconds until interrupted.

    A run that fails to sync, for example because input_output.json was caught
    mid-write, is retried on the next poll; each distinct error is reported once.
    """
    seen: Dict[Path, Signature] = {}
    errors: Dict[Path, str] = {}
    while True:
        for run_dir, signature in changed_runs(root, seen).items():
            result = sync(run_dir)
            if result.error is None:
                seen[run_dir] = signature
                errors.pop(run_dir, None)
                report(result)
            elif errors.get(run_dir) != result.error:
                errors[run_dir] = result.error
                report(result)
        time.sleep(interval_s)
