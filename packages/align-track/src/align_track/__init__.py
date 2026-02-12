"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main


# Lazy imports for optional tracking backends
def __getattr__(name: str):
    """Lazy import for optional tracking modules."""
    if name == "ClearMLTracker":
        from .clearml_tracker import ClearMLTracker
        return ClearMLTracker
    if name == "track_hydra_task":
        from .clearml_tracker import track_hydra_task
        return track_hydra_task
    if name == "init_clearml_task":
        from .clearml_tracker import init_clearml_task
        return init_clearml_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "list_runs_main",
    "__version__",
    # clearml tracking (optional)
    "ClearMLTracker",
    "track_hydra_task",
    "init_clearml_task",
]
