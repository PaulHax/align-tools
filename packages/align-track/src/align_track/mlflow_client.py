"""MLflow client wrapper utilities."""

from typing import Optional

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


def setup_mlflow_tracking(tracking_uri: str, experiment_name: str) -> str:
    """Setup MLflow tracking URI and experiment.

    Args:
        tracking_uri: MLflow tracking URI (file path or server URL)
        experiment_name: Name of the MLflow experiment

    Returns:
        Experiment ID
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def get_run_by_tag(experiment_id: str, tag_key: str, tag_value: str) -> Optional[Run]:
    """Find a run by tag value.

    Args:
        experiment_id: MLflow experiment ID
        tag_key: Tag key to search for
        tag_value: Tag value to match

    Returns:
        Run object if found, None otherwise
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{tag_key} = '{tag_value}'",
        max_results=1
    )
    return runs[0] if runs else None


def experiment_exists(experiment_id: str, experiment_path: str) -> bool:
    """Check if experiment already exists in MLflow by experiment_path tag.

    Args:
        experiment_id: MLflow experiment ID
        experiment_path: Path to the experiment directory (used as unique identifier)

    Returns:
        True if experiment already tracked in MLflow, False otherwise
    """
    return get_run_by_tag(experiment_id, "experiment_path", experiment_path) is not None
