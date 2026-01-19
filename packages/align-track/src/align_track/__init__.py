"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .list_runs import main as list_runs_main
from .ingest import ingest_experiments_directory, IngestSummary
from .mlflow_client import setup_mlflow_tracking, experiment_exists
from .aim_ingest import ingest_experiments_directory_aim
from .aim_client import setup_aim_repo, scene_run_exists
from .cli import cli

__all__ = [
    "list_runs_main",
    "ingest_experiments_directory",
    "ingest_experiments_directory_aim",
    "IngestSummary",
    "setup_mlflow_tracking",
    "experiment_exists",
    "setup_aim_repo",
    "scene_run_exists",
    "cli",
    "__version__",
]
