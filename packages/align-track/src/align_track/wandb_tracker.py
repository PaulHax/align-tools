"""Weights & Biases experiment tracking for align-system with Hydra integration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

try:
    import wandb
    from wandb.sdk.wandb_run import Run

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    Run = Any  # type: ignore


def _check_wandb_available() -> None:
    """Check if wandb is installed."""
    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is not installed. Install with: pip install wandb"
        )


def _omegaconf_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf DictConfig to plain dict for wandb.

    Args:
        cfg: Hydra/OmegaConf configuration object

    Returns:
        Plain dictionary with resolved values
    """
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except ImportError:
        # If OmegaConf not available, assume it's already a dict
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        return dict(cfg) if cfg else {}


class WandbTracker:
    """Experiment tracker using Weights & Biases.

    Provides seamless integration between Hydra configuration management
    and wandb experiment tracking for align-system experiments.

    Example:
        >>> from align_track.wandb_tracker import WandbTracker
        >>>
        >>> tracker = WandbTracker(project="align-experiments")
        >>> with tracker.start_run(hydra_cfg) as run:
        ...     for epoch in range(100):
        ...         run.log({"loss": loss_value, "accuracy": acc_value})
        ...     run.log_artifact("model.pt", type="model")
    """

    def __init__(
        self,
        project: str = "align-experiments",
        entity: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ):
        """Initialize the wandb tracker.

        Args:
            project: wandb project name
            entity: wandb entity (team or username)
            tracking_uri: Optional self-hosted wandb server URL
        """
        _check_wandb_available()
        self.project = project
        self.entity = entity
        self.tracking_uri = tracking_uri
        self._run: Optional[Run] = None

        if tracking_uri:
            os.environ["WANDB_BASE_URL"] = tracking_uri

    @contextmanager
    def start_run(
        self,
        config: Optional[Any] = None,
        run_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        notes: Optional[str] = None,
        resume: Optional[str] = None,
    ) -> Generator[Run, None, None]:
        """Start a new wandb run with Hydra config support.

        Args:
            config: Hydra DictConfig or plain dict of parameters
            run_name: Name for this run
            tags: List of tags for categorization
            group: Group name for related runs
            job_type: Type of job (train, eval, sweep)
            notes: Optional notes about this run
            resume: Resume mode ('allow', 'must', 'never', run_id)

        Yields:
            wandb.Run object for logging metrics and artifacts
        """
        # Convert OmegaConf to dict if needed
        config_dict = _omegaconf_to_dict(config) if config else None

        # Auto-detect run name from Hydra output dir if available
        if run_name is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_dir = Path(HydraConfig.get().runtime.output_dir)
                run_name = hydra_dir.name
                if group is None:
                    group = hydra_dir.parent.name
            except Exception:
                pass

        # Initialize wandb run
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config_dict,
            tags=tags or [],
            group=group,
            job_type=job_type,
            notes=notes,
            resume=resume,
            settings=wandb.Settings(start_method="thread"),
        )

        try:
            # Log Hydra config file as artifact if available
            self._log_hydra_config_artifact()
            yield self._run
        finally:
            if self._run:
                self._run.finish()
                self._run = None

    def _log_hydra_config_artifact(self) -> None:
        """Log the Hydra config.yaml as an artifact if available."""
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_dir = Path(HydraConfig.get().runtime.output_dir)
            config_file = hydra_dir / ".hydra" / "config.yaml"

            if config_file.exists():
                artifact = wandb.Artifact(
                    name=f"hydra-config-{self._run.id}",
                    type="config",
                    description="Hydra configuration file",
                )
                artifact.add_file(str(config_file))
                self._run.log_artifact(artifact)
        except Exception:
            pass  # Hydra config not available

    @property
    def run(self) -> Optional[Run]:
        """Get the current active run."""
        return self._run

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step/epoch number
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")
        self._run.log(metrics, step=step)

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_type: str = "file",
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact to the current run.

        Args:
            file_path: Path to the file to log
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Optional name for the artifact
            metadata: Optional metadata dict
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        file_path = Path(file_path)
        artifact_name = name or f"{artifact_type}-{self._run.id}"

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {},
        )

        if file_path.is_dir():
            artifact.add_dir(str(file_path))
        else:
            artifact.add_file(str(file_path))

        self._run.log_artifact(artifact)


def track_hydra_run(
    project: str = "align-experiments",
    entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
    job_type: str = "train",
):
    """Decorator to automatically track Hydra experiments with wandb.

    Example:
        >>> @hydra.main(config_path="conf", config_name="config")
        ... @track_hydra_run(project="my-project")
        ... def train(cfg):
        ...     # Your training code
        ...     pass

    Args:
        project: wandb project name
        entity: wandb entity (team or username)
        tags: List of tags for the run
        job_type: Type of job (train, eval, etc.)

    Returns:
        Decorated function with automatic wandb tracking
    """
    def decorator(func):
        def wrapper(cfg, *args, **kwargs):
            tracker = WandbTracker(project=project, entity=entity)

            with tracker.start_run(
                config=cfg,
                tags=tags,
                job_type=job_type,
            ) as run:
                # Inject run into kwargs for optional access
                kwargs["_wandb_run"] = run
                kwargs["_wandb_tracker"] = tracker

                try:
                    result = func(cfg, *args, **kwargs)
                    return result
                except Exception as e:
                    # Log error before re-raising
                    run.log({"error": str(e)})
                    raise

        return wrapper
    return decorator


# Convenience function for quick setup
def init_wandb_tracking(
    project: str = "align-experiments",
    entity: Optional[str] = None,
    config: Optional[Any] = None,
    **kwargs,
) -> Run:
    """Initialize wandb tracking with sensible defaults for align-system.

    This is a convenience function for quick setup without the context manager.
    Remember to call wandb.finish() when done.

    Args:
        project: wandb project name
        entity: wandb entity
        config: Hydra config or dict
        **kwargs: Additional arguments passed to wandb.init()

    Returns:
        wandb.Run object
    """
    _check_wandb_available()

    config_dict = _omegaconf_to_dict(config) if config else None

    return wandb.init(
        project=project,
        entity=entity,
        config=config_dict,
        settings=wandb.Settings(start_method="thread"),
        **kwargs,
    )
