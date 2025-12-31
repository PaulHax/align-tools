"""ClearML experiment tracking for align-system with native Hydra integration.

ClearML provides native OmegaConf/Hydra support, automatically capturing
configurations without manual conversion.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

try:
    from clearml import Task, Logger

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    Task = Any  # type: ignore
    Logger = Any  # type: ignore


def _check_clearml_available() -> None:
    """Check if clearml is installed."""
    if not CLEARML_AVAILABLE:
        raise ImportError(
            "clearml is not installed. Install with: pip install clearml"
        )


class ClearMLTracker:
    """Experiment tracker using ClearML.

    ClearML has native OmegaConf support, making it excellent for Hydra
    experiments. It automatically logs Hydra configurations without manual
    conversion.

    Example:
        >>> from align_track.clearml_tracker import ClearMLTracker
        >>>
        >>> # ClearML automatically captures Hydra config!
        >>> tracker = ClearMLTracker(project="align-experiments")
        >>> with tracker.start_task("training") as task:
        ...     for epoch in range(100):
        ...         tracker.log_metric("loss", loss_value, epoch)
        ...     tracker.log_artifact("model.pt")

    Setup:
        1. Create a free account at https://app.clear.ml
        2. Run `clearml-init` to configure credentials
        3. Or self-host with `docker run clearml`
    """

    def __init__(
        self,
        project_name: str = "align-experiments",
        auto_connect_frameworks: bool = True,
        reuse_last_task_id: bool = False,
    ):
        """Initialize the ClearML tracker.

        Args:
            project_name: ClearML project name
            auto_connect_frameworks: Auto-log popular ML frameworks
            reuse_last_task_id: Reuse existing task if matches
        """
        _check_clearml_available()
        self.project_name = project_name
        self.auto_connect_frameworks = auto_connect_frameworks
        self.reuse_last_task_id = reuse_last_task_id
        self._task: Optional[Task] = None
        self._logger: Optional[Logger] = None

    @contextmanager
    def start_task(
        self,
        task_name: str,
        task_type: str = "training",
        tags: Optional[list[str]] = None,
        connect_config: bool = True,
    ) -> Generator[Task, None, None]:
        """Start a new ClearML task with automatic Hydra config capture.

        ClearML will automatically detect and log OmegaConf configurations
        used by Hydra. No manual conversion needed!

        Args:
            task_name: Name for this task/run
            task_type: Type of task (training, testing, inference, etc.)
            tags: List of tags for categorization
            connect_config: Auto-connect Hydra config (recommended True)

        Yields:
            ClearML Task object
        """
        # Map common task type names
        task_type_map = {
            "train": Task.TaskTypes.training,
            "training": Task.TaskTypes.training,
            "eval": Task.TaskTypes.testing,
            "test": Task.TaskTypes.testing,
            "testing": Task.TaskTypes.testing,
            "inference": Task.TaskTypes.inference,
            "data": Task.TaskTypes.data_processing,
            "data_processing": Task.TaskTypes.data_processing,
        }
        clearml_task_type = task_type_map.get(
            task_type.lower(), Task.TaskTypes.training
        )

        # Initialize ClearML task
        # ClearML automatically captures OmegaConf/Hydra configs!
        self._task = Task.init(
            project_name=self.project_name,
            task_name=task_name,
            task_type=clearml_task_type,
            tags=tags,
            auto_connect_frameworks=self.auto_connect_frameworks,
            reuse_last_task_id=self.reuse_last_task_id,
        )

        self._logger = self._task.get_logger()

        try:
            yield self._task
        finally:
            self._task.close()
            self._task = None
            self._logger = None

    @property
    def task(self) -> Optional[Task]:
        """Get the current active task."""
        return self._task

    @property
    def logger(self) -> Optional[Logger]:
        """Get the current logger."""
        return self._logger

    def log_metric(
        self,
        title: str,
        value: float,
        iteration: Optional[int] = None,
        series: Optional[str] = None,
    ) -> None:
        """Log a scalar metric.

        Args:
            title: Metric name/title
            value: Metric value
            iteration: Step/iteration number
            series: Series name within the title (for grouping)
        """
        if self._logger is None:
            raise RuntimeError("No active task. Use start_task() first.")

        self._logger.report_scalar(
            title=title,
            series=series or title,
            value=value,
            iteration=iteration or 0,
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        iteration: Optional[int] = None,
        title: str = "metrics",
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            iteration: Step/iteration number
            title: Title for grouping metrics
        """
        if self._logger is None:
            raise RuntimeError("No active task. Use start_task() first.")

        for series, value in metrics.items():
            self._logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration or 0,
            )

    def log_artifact(
        self,
        name: str,
        artifact_object: Union[str, Path, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (file, directory, or Python object).

        Args:
            name: Artifact name
            artifact_object: Path to file/directory or Python object
            metadata: Optional metadata dict
        """
        if self._task is None:
            raise RuntimeError("No active task. Use start_task() first.")

        artifact_path = Path(artifact_object) if isinstance(
            artifact_object, (str, Path)
        ) else None

        if artifact_path and artifact_path.exists():
            self._task.upload_artifact(
                name=name,
                artifact_object=str(artifact_path),
                metadata=metadata,
            )
        else:
            # Log Python object directly
            self._task.upload_artifact(
                name=name,
                artifact_object=artifact_object,
                metadata=metadata,
            )

    def log_model(
        self,
        name: str,
        model_path: Union[str, Path],
        framework: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a model file or directory.

        Args:
            name: Model name
            model_path: Path to model file or directory
            framework: ML framework (pytorch, tensorflow, sklearn, etc.)
            metadata: Optional metadata dict
        """
        if self._task is None:
            raise RuntimeError("No active task. Use start_task() first.")

        output_model = self._task.connect_output_model(name=name)
        output_model.update_weights(
            weights_filename=str(model_path),
            framework=framework,
            auto_delete_file=False,
        )

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: Optional[int] = None,
    ) -> None:
        """Log a matplotlib figure.

        Args:
            title: Plot title
            series: Series name
            figure: Matplotlib figure object
            iteration: Step/iteration number
        """
        if self._logger is None:
            raise RuntimeError("No active task. Use start_task() first.")

        self._logger.report_matplotlib_figure(
            title=title,
            series=series,
            figure=figure,
            iteration=iteration or 0,
        )

    def log_table(
        self,
        title: str,
        series: str,
        table: Any,
        iteration: Optional[int] = None,
    ) -> None:
        """Log a pandas DataFrame or table.

        Args:
            title: Table title
            series: Series name
            table: Pandas DataFrame or dict
            iteration: Step/iteration number
        """
        if self._logger is None:
            raise RuntimeError("No active task. Use start_task() first.")

        self._logger.report_table(
            title=title,
            series=series,
            table_plot=table,
            iteration=iteration or 0,
        )

    def connect_config(self, config: Dict[str, Any], name: str = "config") -> Dict:
        """Manually connect a configuration dict.

        Note: ClearML auto-connects OmegaConf, so this is usually not needed
        for Hydra configs. Use for additional configs.

        Args:
            config: Configuration dictionary
            name: Config section name

        Returns:
            Connected config (may be modified by UI)
        """
        if self._task is None:
            raise RuntimeError("No active task. Use start_task() first.")

        return self._task.connect(config, name=name)


def track_hydra_task(
    project_name: str = "align-experiments",
    task_type: str = "training",
    tags: Optional[list[str]] = None,
):
    """Decorator to automatically track Hydra experiments with ClearML.

    ClearML has native OmegaConf support, so Hydra configurations are
    automatically captured and logged.

    Example:
        >>> @hydra.main(config_path="conf", config_name="config")
        ... @track_hydra_task(project_name="my-project")
        ... def train(cfg):
        ...     # ClearML automatically captures cfg!
        ...     pass

    Args:
        project_name: ClearML project name
        task_type: Type of task (training, testing, etc.)
        tags: List of tags

    Returns:
        Decorated function with automatic ClearML tracking
    """
    def decorator(func):
        def wrapper(cfg, *args, **kwargs):
            # Get task name from Hydra output dir if possible
            task_name = "experiment"
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_dir = Path(HydraConfig.get().runtime.output_dir)
                task_name = hydra_dir.name
            except Exception:
                pass

            tracker = ClearMLTracker(project_name=project_name)

            with tracker.start_task(
                task_name=task_name,
                task_type=task_type,
                tags=tags,
            ) as task:
                # ClearML auto-captures the Hydra OmegaConf config!
                # Inject tracker for optional access
                kwargs["_clearml_task"] = task
                kwargs["_clearml_tracker"] = tracker

                try:
                    result = func(cfg, *args, **kwargs)
                    return result
                except Exception as e:
                    tracker.log_metric("error", 1)
                    task.add_tags(["failed"])
                    raise

        return wrapper
    return decorator


# Convenience function for initialization
def init_clearml_task(
    project_name: str = "align-experiments",
    task_name: str = "experiment",
    task_type: str = "training",
    tags: Optional[list[str]] = None,
) -> Task:
    """Quick initialization of ClearML task.

    This is a convenience function that directly returns a ClearML Task.
    The Task will automatically capture Hydra/OmegaConf configurations.

    Args:
        project_name: ClearML project name
        task_name: Name for this task
        task_type: Type of task
        tags: Optional tags

    Returns:
        ClearML Task object
    """
    _check_clearml_available()

    task_type_map = {
        "train": Task.TaskTypes.training,
        "training": Task.TaskTypes.training,
        "eval": Task.TaskTypes.testing,
        "test": Task.TaskTypes.testing,
    }

    return Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type_map.get(task_type, Task.TaskTypes.training),
        tags=tags,
        auto_connect_frameworks=True,
    )


def setup_clearml_credentials() -> None:
    """Print instructions for setting up ClearML credentials."""
    print("""
ClearML Setup Instructions
==========================

Option 1: Free cloud (recommended for getting started)
-------------------------------------------------------
1. Create account at https://app.clear.ml
2. Go to Settings > Workspace > Create new credentials
3. Run: clearml-init
4. Paste your credentials when prompted

Option 2: Self-hosted server
----------------------------
1. Run ClearML server:
   docker run -d --name clearml-server -p 8080:8080 \\
       allegroai/clearml:latest

2. Access UI at http://localhost:8080
3. Create credentials in Settings
4. Run: clearml-init --host http://localhost:8080

The self-hosted option is 100% free with unlimited users.
""")
