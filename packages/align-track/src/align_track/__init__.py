"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main

# Lazy imports for optional tracking backends
def __getattr__(name: str):
    """Lazy import for optional tracking modules."""
    if name == "WandbTracker":
        from .wandb_tracker import WandbTracker
        return WandbTracker
    if name == "track_hydra_run":
        from .wandb_tracker import track_hydra_run
        return track_hydra_run
    if name == "init_wandb_tracking":
        from .wandb_tracker import init_wandb_tracking
        return init_wandb_tracking
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "list_runs_main",
    "__version__",
    # wandb tracking (optional)
    "WandbTracker",
    "track_hydra_run",
    "init_wandb_tracking",
]
