"""Tracking and monitoring utilities for align-system experiments."""

from importlib.metadata import version

__version__ = version("align-track")

from .ingest import ingest_experiments_directory, IngestSummary
from .mlflow_client import setup_mlflow_tracking, experiment_exists
from .cli import cli

__all__ = [
    "ingest_experiments_directory",
    "IngestSummary",
    "setup_mlflow_tracking",
    "experiment_exists",
    "cli",
    "__version__",
]
