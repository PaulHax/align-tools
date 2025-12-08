"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main

__all__ = ["list_runs_main", "__version__"]
