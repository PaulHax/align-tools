"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main


# Lazy imports for optional tracking backends
def __getattr__(name: str):
    """Lazy import for optional tracking modules."""
    if name == "AimTracker":
        from .aim_tracker import AimTracker
        return AimTracker
    if name == "track_hydra_run":
        from .aim_tracker import track_hydra_run
        return track_hydra_run
    if name == "query_runs":
        from .aim_tracker import query_runs
        return query_runs
    if name == "query_metrics":
        from .aim_tracker import query_metrics
        return query_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "list_runs_main",
    "__version__",
    # aim tracking (optional)
    "AimTracker",
    "track_hydra_run",
    "query_runs",
    "query_metrics",
]
