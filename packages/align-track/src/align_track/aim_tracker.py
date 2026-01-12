"""Aim experiment tracking for align-system with native OmegaConf/Hydra support.

Aim is a modern, open-source, self-hosted experiment tracking tool with
native OmegaConf support for seamless Hydra integration.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

try:
    from aim import Run, Repo

    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = Any  # type: ignore
    Repo = Any  # type: ignore


def _check_aim_available() -> None:
    """Check if aim is installed."""
    if not AIM_AVAILABLE:
        raise ImportError(
            "aim is not installed. Install with: pip install aim"
        )


class AimTracker:
    """Experiment tracker using Aim.

    Aim is a self-hosted, open-source experiment tracking tool with:
    - Native OmegaConf support (perfect for Hydra!)
    - Modern, performant UI for 1000s of runs
    - SQL-like query API for experiment analysis
    - No external dependencies or cloud services

    Example:
        >>> from align_track.aim_tracker import AimTracker
        >>>
        >>> tracker = AimTracker()
        >>> with tracker.start_run(experiment="training") as run:
        ...     run["hparams"] = hydra_cfg  # Native OmegaConf support!
        ...     for epoch in range(100):
        ...         run.track(loss, name="loss", step=epoch)

    Setup:
        1. pip install aim
        2. aim init (in your repo)
        3. aim up (to start the UI)
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = None,
        capture_terminal_logs: bool = True,
    ):
        """Initialize the Aim tracker.

        Args:
            repo: Path to Aim repo or remote tracking server URL
                  (e.g., "aim://tracking-server:53800")
            experiment: Default experiment name
            system_tracking_interval: Interval for system metric tracking (seconds)
            capture_terminal_logs: Whether to capture terminal output
        """
        _check_aim_available()
        self.repo = repo
        self.experiment = experiment
        self.system_tracking_interval = system_tracking_interval
        self.capture_terminal_logs = capture_terminal_logs
        self._run: Optional[Run] = None

    @contextmanager
    def start_run(
        self,
        experiment: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> Generator[Run, None, None]:
        """Start a new Aim run with native Hydra config support.

        Aim natively supports OmegaConf objects, so Hydra configs can
        be assigned directly without conversion!

        Args:
            experiment: Experiment name (overrides default)
            run_name: Optional name for this run
            config: Hydra DictConfig or dict (assigned to run["hparams"])

        Yields:
            Aim Run object for tracking
        """
        exp_name = experiment or self.experiment or "default"

        # Auto-detect run name from Hydra if not provided
        if run_name is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_dir = Path(HydraConfig.get().runtime.output_dir)
                run_name = hydra_dir.name
            except Exception:
                pass

        # Create Aim run
        self._run = Run(
            repo=self.repo,
            experiment=exp_name,
            system_tracking_interval=self.system_tracking_interval,
            capture_terminal_logs=self.capture_terminal_logs,
        )

        # Set run name if provided
        if run_name:
            self._run.name = run_name

        # Log Hydra config - Aim natively supports OmegaConf!
        if config is not None:
            self._run["hparams"] = config

        # Add git info
        self._log_git_info()

        try:
            yield self._run
        finally:
            self._run.close()
            self._run = None

    def _log_git_info(self) -> None:
        """Log git information to the run."""
        if self._run is None:
            return

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
            self._run["git_commit"] = commit[:8]
            self._run["git_branch"] = branch
        except Exception:
            pass

    @property
    def run(self) -> Optional[Run]:
        """Get the current active run."""
        return self._run

    def track(
        self,
        value: float,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """Track a metric value.

        Args:
            value: Metric value
            name: Metric name
            step: Step/iteration number
            context: Additional context (e.g., {"subset": "train"})
            epoch: Epoch number (separate from step)
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        self._run.track(
            value,
            name=name,
            step=step,
            context=context,
            epoch=epoch,
        )

    def track_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value
            step: Step/iteration number
            context: Additional context
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        for name, value in metrics.items():
            self._run.track(value, name=name, step=step, context=context)

    def log_figure(
        self,
        figure: Any,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a matplotlib figure.

        Args:
            figure: Matplotlib figure object
            name: Figure name
            step: Step number
            context: Additional context
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        from aim import Figure
        self._run.track(Figure(figure), name=name, step=step, context=context)

    def log_image(
        self,
        image: Any,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image.

        Args:
            image: Image (numpy array, PIL Image, or path)
            name: Image name
            step: Step number
            context: Additional context
            caption: Image caption
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        from aim import Image
        self._run.track(
            Image(image, caption=caption),
            name=name,
            step=step,
            context=context,
        )

    def log_text(
        self,
        text: str,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log text data.

        Args:
            text: Text content
            name: Text name
            step: Step number
            context: Additional context
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        from aim import Text
        self._run.track(Text(text), name=name, step=step, context=context)

    def set_param(self, key: str, value: Any) -> None:
        """Set a run parameter.

        Args:
            key: Parameter key
            value: Parameter value
        """
        if self._run is None:
            raise RuntimeError("No active run. Use start_run() first.")

        self._run[key] = value


def query_runs(
    repo_path: Optional[str] = None,
    query: Optional[str] = None,
    experiment: Optional[str] = None,
) -> list:
    """Query runs from an Aim repository.

    Args:
        repo_path: Path to Aim repo
        query: Aim query string (e.g., "run.hparams.lr > 0.01")
        experiment: Filter by experiment name

    Returns:
        List of matching Run objects
    """
    _check_aim_available()

    repo = Repo(repo_path) if repo_path else Repo.from_path(".")

    runs = repo.iter_runs()

    if experiment:
        runs = (r for r in runs if r.experiment == experiment)

    if query:
        # Aim's query API
        runs = repo.query_runs(query).iter_runs()

    return list(runs)


def query_metrics(
    repo_path: Optional[str] = None,
    query: str = "",
) -> list:
    """Query metrics from an Aim repository.

    Args:
        repo_path: Path to Aim repo
        query: Aim metric query (e.g., "metric.name == 'loss'")

    Returns:
        List of metric collections
    """
    _check_aim_available()

    repo = Repo(repo_path) if repo_path else Repo.from_path(".")
    return list(repo.query_metrics(query).iter_runs())


def track_hydra_run(
    experiment: str = "align-experiments",
    repo: Optional[str] = None,
):
    """Decorator to automatically track Hydra experiments with Aim.

    Aim has native OmegaConf support, so the Hydra config is automatically
    captured without any conversion needed!

    Example:
        >>> @hydra.main(config_path="conf", config_name="config")
        ... @track_hydra_run(experiment="my-experiment")
        ... def train(cfg):
        ...     # cfg is automatically logged to Aim!
        ...     pass

    Args:
        experiment: Aim experiment name
        repo: Aim repository path or remote URL

    Returns:
        Decorated function with automatic Aim tracking
    """
    def decorator(func):
        def wrapper(cfg, *args, **kwargs):
            tracker = AimTracker(repo=repo, experiment=experiment)

            with tracker.start_run(config=cfg) as run:
                # Inject run for optional access
                kwargs["_aim_run"] = run
                kwargs["_aim_tracker"] = tracker

                try:
                    result = func(cfg, *args, **kwargs)
                    return result
                except Exception as e:
                    run["error"] = str(e)
                    raise

        return wrapper
    return decorator


def setup_aim() -> None:
    """Print instructions for setting up Aim."""
    print("""
Aim Setup Instructions
======================

1. Initialize Aim in your repository:
   aim init

2. Start the Aim UI:
   aim up

3. View experiments at http://localhost:43800

Remote Tracking Server (for distributed experiments):
------------------------------------------------------
1. Start tracking server:
   aim server --host 0.0.0.0 --port 53800

2. Connect from remote machines:
   tracker = AimTracker(repo="aim://server-ip:53800")

Docker Deployment:
------------------
docker run -d \\
    -p 43800:43800 \\
    -p 53800:53800 \\
    -v /path/to/aim:/aim \\
    aimstack/aim

Aim is 100% free and open source with no usage limits.
""")
