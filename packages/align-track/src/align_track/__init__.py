"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main


# Lazy imports for optional tracking backends
def __getattr__(name: str):
    """Lazy import for optional tracking modules."""
    if name == "MLflowTracker":
        from .mlflow_tracker import MLflowTracker
        return MLflowTracker
    if name == "get_best_run":
        from .mlflow_tracker import get_best_run
        return get_best_run
    if name == "track_hydra_run":
        from .mlflow_tracker import track_hydra_run
        return track_hydra_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "list_runs_main",
    "__version__",
    # mlflow tracking (optional)
    "MLflowTracker",
    "get_best_run",
    "track_hydra_run",
]
