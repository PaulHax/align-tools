"""MLflow experiment tracking for align-system with Hydra integration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

try:
    import mlflow
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    Run = Any  # type: ignore
    MlflowClient = Any  # type: ignore


def _check_mlflow_available() -> None:
    """Check if mlflow is installed."""
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "mlflow is not installed. Install with: pip install mlflow"
        )


def _omegaconf_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to plain dict.

    Args:
        cfg: Hydra/OmegaConf configuration object

    Returns:
        Plain dictionary with resolved values
    """
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except ImportError:
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        return dict(cfg) if cfg else {}


def _flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten nested dictionary for MLflow parameters.

    MLflow parameters must be flat key-value pairs. This function
    flattens nested dicts using dot notation.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys

    Returns:
        Flattened dictionary with string values
    """
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            # MLflow params must be strings and <= 500 chars
            str_val = str(v) if v is not None else ""
            if len(str_val) > 500:
                str_val = str_val[:497] + "..."
            items.append((new_key, str_val))

    return dict(items)


class MLflowTracker:
    """Experiment tracker using MLflow.

    Provides integration between Hydra configuration management
    and MLflow experiment tracking for align-system experiments.

    Example:
        >>> from align_track.mlflow_tracker import MLflowTracker
        >>>
        >>> tracker = MLflowTracker(
        ...     experiment_name="align-experiments",
        ...     tracking_uri="http://localhost:5000"
        ... )
        >>> with tracker.start_run(hydra_cfg) as run:
        ...     for epoch in range(100):
        ...         tracker.log_metrics({"loss": loss_value})
        ...     tracker.log_artifact("model.pt")
    """

    def __init__(
        self,
        experiment_name: str = "align-experiments",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """Initialize the MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file store)
            artifact_location: Base location for artifacts
        """
        _check_mlflow_available()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "mlruns"
        )
        self.artifact_location = artifact_location
        self._run_id: Optional[str] = None

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self._experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(
        self,
        config: Optional[Any] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
        description: Optional[str] = None,
    ) -> Generator["MLflowTracker", None, None]:
        """Start a new MLflow run with Hydra config support.

        Args:
            config: Hydra DictConfig or plain dict of parameters
            run_name: Name for this run
            tags: Dictionary of tags
            nested: Whether this is a nested run
            description: Optional description for the run

        Yields:
            Self (MLflowTracker) for method chaining
        """
        # Convert OmegaConf to dict if needed
        config_dict = _omegaconf_to_dict(config) if config else {}

        # Auto-detect run name from Hydra output dir if available
        if run_name is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_dir = Path(HydraConfig.get().runtime.output_dir)
                run_name = hydra_dir.name
            except Exception:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare tags
        run_tags = {"framework": "align-track"}
        if tags:
            run_tags.update(tags)

        # Add git info if available
        git_info = self._get_git_info()
        if git_info:
            run_tags.update(git_info)

        # Start MLflow run
        with mlflow.start_run(
            run_name=run_name,
            nested=nested,
            tags=run_tags,
            description=description,
        ) as run:
            self._run_id = run.info.run_id

            # Log flattened parameters
            if config_dict:
                flat_params = _flatten_dict(config_dict)
                # MLflow has a limit on number of params, log in batches
                param_items = list(flat_params.items())
                batch_size = 100
                for i in range(0, len(param_items), batch_size):
                    batch = dict(param_items[i:i + batch_size])
                    mlflow.log_params(batch)

            # Log Hydra config file as artifact
            self._log_hydra_config_artifact()

            try:
                yield self
            finally:
                self._run_id = None

    def _get_git_info(self) -> Dict[str, str]:
        """Get git commit and branch info."""
        import subprocess

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return {"git.commit": commit[:8], "git.branch": branch}
        except Exception:
            return {}

    def _log_hydra_config_artifact(self) -> None:
        """Log the Hydra config.yaml as an artifact if available."""
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_dir = Path(HydraConfig.get().runtime.output_dir)
            config_dir = hydra_dir / ".hydra"

            if config_dir.exists():
                mlflow.log_artifacts(str(config_dir), "hydra_config")
        except Exception:
            pass

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run_id

    @property
    def experiment_id(self) -> str:
        """Get the experiment ID."""
        return self._experiment_id

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self._run_id is None:
            raise RuntimeError("No active run. Use start_run() first.")
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if self._run_id is None:
            raise RuntimeError("No active run. Use start_run() first.")
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact file or directory.

        Args:
            local_path: Path to file or directory to log
            artifact_path: Destination path within artifact store
        """
        if self._run_id is None:
            raise RuntimeError("No active run. Use start_run() first.")

        local_path = Path(local_path)
        if local_path.is_dir():
            mlflow.log_artifacts(str(local_path), artifact_path)
        else:
            mlflow.log_artifact(str(local_path), artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: str = "sklearn",
        **kwargs,
    ) -> None:
        """Log a model using MLflow's model logging.

        Args:
            model: Model object to log
            artifact_path: Destination path for model artifacts
            flavor: MLflow flavor (sklearn, pytorch, tensorflow, etc.)
            **kwargs: Additional arguments for the flavor's log_model
        """
        if self._run_id is None:
            raise RuntimeError("No active run. Use start_run() first.")

        flavor_module = getattr(mlflow, flavor, None)
        if flavor_module is None:
            raise ValueError(f"Unknown MLflow flavor: {flavor}")

        flavor_module.log_model(model, artifact_path, **kwargs)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        if self._run_id is None:
            raise RuntimeError("No active run. Use start_run() first.")
        mlflow.set_tag(key, value)


def get_best_run(
    experiment_name: str,
    metric_name: str,
    ascending: bool = False,
    tracking_uri: Optional[str] = None,
) -> Run:
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric_name: Metric to optimize
        ascending: If True, minimize; if False, maximize
        tracking_uri: MLflow tracking URI

    Returns:
        Best MLflow Run object

    Raises:
        ValueError: If experiment or runs not found
    """
    _check_mlflow_available()

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} {order}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    return runs[0]


def track_hydra_run(
    experiment_name: str = "align-experiments",
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Decorator to automatically track Hydra experiments with MLflow.

    Example:
        >>> @hydra.main(config_path="conf", config_name="config")
        ... @track_hydra_run(experiment_name="my-experiment")
        ... def train(cfg):
        ...     # Your training code
        ...     pass

    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI
        tags: Additional tags for the run

    Returns:
        Decorated function with automatic MLflow tracking
    """
    def decorator(func):
        def wrapper(cfg, *args, **kwargs):
            tracker = MLflowTracker(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
            )

            with tracker.start_run(config=cfg, tags=tags) as run:
                # Inject tracker into kwargs for optional access
                kwargs["_mlflow_tracker"] = run

                try:
                    result = func(cfg, *args, **kwargs)
                    return result
                except Exception as e:
                    mlflow.set_tag("error", str(e)[:250])
                    raise

        return wrapper
    return decorator


def start_mlflow_server(
    port: int = 5000,
    host: str = "127.0.0.1",
    backend_store_uri: str = "sqlite:///mlflow.db",
    artifact_root: str = "./mlflow-artifacts",
) -> None:
    """Print command to start MLflow tracking server.

    Args:
        port: Server port
        host: Server host
        backend_store_uri: Database URI for metadata
        artifact_root: Root directory for artifacts
    """
    cmd = f"""mlflow server \\
    --backend-store-uri {backend_store_uri} \\
    --default-artifact-root {artifact_root} \\
    --host {host} \\
    --port {port}"""

    print("Start MLflow tracking server with:")
    print(cmd)
    print(f"\nThen access the UI at: http://{host}:{port}")
