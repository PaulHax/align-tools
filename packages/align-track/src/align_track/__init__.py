"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main
from .sqlite_tracker import SQLiteTracker, track_run, track_hydra_run


__all__ = [
    "list_runs_main",
    "__version__",
    # sqlite tracking (zero dependencies)
    "SQLiteTracker",
    "track_run",
    "track_hydra_run",
]
