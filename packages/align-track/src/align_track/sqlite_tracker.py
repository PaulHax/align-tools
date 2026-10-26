"""Lightweight SQLite-based experiment tracking with zero external dependencies.

This module provides a minimal experiment tracking solution using only
Python's built-in sqlite3 module. Perfect for local experiments where
you want tracking without additional infrastructure or dependencies.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union


# Database schema
SCHEMA = """
-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    git_commit TEXT,
    git_branch TEXT
);

-- Runs within experiments
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    run_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled'))
        DEFAULT 'running',
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Parameters/config (flattened key-value)
CREATE TABLE IF NOT EXISTS parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    value_type TEXT CHECK(value_type IN ('int', 'float', 'str', 'bool', 'json'))
        DEFAULT 'str',
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
    UNIQUE(run_id, key)
);

-- Time-series metrics
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    timestamp REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Artifacts (files)
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    artifact_type TEXT,
    file_path TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Tags for categorization
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Indices for fast queries
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_parameters_run ON parameters(run_id);
CREATE INDEX IF NOT EXISTS idx_parameters_key ON parameters(key);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics(step);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_tags_run ON tags(run_id);
"""


def _flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, tuple[str, str]]:
    """Flatten nested dict, returning (value_type, str_value) tuples."""
    items: List[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, bool):
                value_type, value_str = "bool", str(v)
            elif isinstance(v, int):
                value_type, value_str = "int", str(v)
            elif isinstance(v, float):
                value_type, value_str = "float", str(v)
            elif isinstance(v, (list, tuple)):
                value_type, value_str = "json", json.dumps(v)
            else:
                value_type, value_str = "str", str(v) if v is not None else ""
            items.append((new_key, (value_type, value_str)))

    return dict(items)


def _omegaconf_to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert OmegaConf to dict if needed."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except ImportError:
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        return dict(cfg) if cfg else {}


class SQLiteTracker:
    """Lightweight experiment tracker using SQLite.

    Zero external dependencies - uses only Python's built-in sqlite3 module.
    Perfect for local experiments without infrastructure overhead.

    Example:
        >>> from align_track.sqlite_tracker import SQLiteTracker
        >>>
        >>> tracker = SQLiteTracker("experiments.db")
        >>> with tracker.start_run("my-experiment", config=hydra_cfg) as run:
        ...     for epoch in range(100):
        ...         run.log_metric("loss", loss_value, step=epoch)
        ...     run.log_artifact("model.pt")

    Features:
        - Zero dependencies (built-in sqlite3)
        - Portable single-file database
        - Git-friendly (can be committed for small DBs)
        - Full SQL query capability
        - Hydra config support via flattening
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "experiments.db",
        artifacts_dir: Union[str, Path] = "artifacts",
    ):
        """Initialize the SQLite tracker.

        Args:
            db_path: Path to SQLite database file
            artifacts_dir: Directory for storing artifact files
        """
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_id: Optional[int] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_git_info(self) -> Dict[str, str]:
        """Get git commit and branch info."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return {"commit": commit[:8], "branch": branch}
        except Exception:
            return {}

    def create_experiment(self, name: str, notes: Optional[str] = None) -> int:
        """Create or get an experiment.

        Args:
            name: Experiment name
            notes: Optional notes

        Returns:
            Experiment ID
        """
        git_info = self._get_git_info()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO experiments (name, notes, git_commit, git_branch)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                   RETURNING id""",
                (name, notes, git_info.get("commit"), git_info.get("branch")),
            )
            return cursor.fetchone()[0]

    @contextmanager
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator["RunContext", None, None]:
        """Start a new run.

        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this run
            config: Hydra config or dict to log as parameters
            tags: Optional tags dict

        Yields:
            RunContext for logging metrics and artifacts
        """
        # Get or create experiment
        exp_id = self.create_experiment(experiment_name)

        # Auto-detect run name from Hydra if not provided
        if run_name is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_dir = Path(HydraConfig.get().runtime.output_dir)
                run_name = hydra_dir.name
            except Exception:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO runs (experiment_id, run_name, started_at, status)
                   VALUES (?, ?, ?, 'running') RETURNING id""",
                (exp_id, run_name, datetime.now().isoformat()),
            )
            run_id = cursor.fetchone()[0]
            self._current_run_id = run_id

            # Log config as parameters
            if config:
                config_dict = _omegaconf_to_dict(config)
                flat_params = _flatten_dict(config_dict)
                for key, (value_type, value_str) in flat_params.items():
                    conn.execute(
                        """INSERT INTO parameters (run_id, key, value, value_type)
                           VALUES (?, ?, ?, ?)""",
                        (run_id, key, value_str, value_type),
                    )

            # Log tags
            if tags:
                for key, value in tags.items():
                    conn.execute(
                        "INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)",
                        (run_id, key, value),
                    )

        run_context = RunContext(self, run_id)

        try:
            yield run_context
            self._finish_run(run_id, "completed")
        except Exception as e:
            self._finish_run(run_id, "failed", str(e))
            raise
        finally:
            self._current_run_id = None

    def _finish_run(
        self,
        run_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark a run as finished."""
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE runs SET status = ?, ended_at = ?, error_message = ?
                   WHERE id = ?""",
                (status, datetime.now().isoformat(), error_message, run_id),
            )

    def log_metric(
        self,
        run_id: int,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric value.

        Args:
            run_id: Run ID
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO metrics (run_id, key, value, step, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, key, value, step, time.time()),
            )

    def log_metrics(
        self,
        run_id: int,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics.

        Args:
            run_id: Run ID
            metrics: Dict of metric name to value
            step: Optional step number
        """
        timestamp = time.time()
        with self._get_connection() as conn:
            conn.executemany(
                """INSERT INTO metrics (run_id, key, value, step, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                [(run_id, k, v, step, timestamp) for k, v in metrics.items()],
            )

    def log_artifact(
        self,
        run_id: int,
        file_path: Union[str, Path],
        artifact_type: str = "file",
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log an artifact file.

        Args:
            run_id: Run ID
            file_path: Path to file
            artifact_type: Type of artifact (model, data, etc.)
            name: Optional name (defaults to filename)
            metadata: Optional metadata dict
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        # Create run-specific artifact directory
        run_artifact_dir = self.artifacts_dir / f"run_{run_id}"
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        artifact_name = name or file_path.name
        dest_path = run_artifact_dir / artifact_name
        shutil.copy2(file_path, dest_path)

        # Store relative path
        relative_path = dest_path.relative_to(self.artifacts_dir)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO artifacts
                   (run_id, name, artifact_type, file_path, size_bytes, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    artifact_name,
                    artifact_type,
                    str(relative_path),
                    file_path.stat().st_size,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def query_runs(
        self,
        experiment_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query runs.

        Args:
            experiment_name: Filter by experiment
            status: Filter by status
            limit: Maximum results

        Returns:
            List of run dicts
        """
        query = """
            SELECT r.*, e.name as experiment_name
            FROM runs r
            JOIN experiments e ON r.experiment_id = e.id
            WHERE 1=1
        """
        params: List[Any] = []

        if experiment_name:
            query += " AND e.name = ?"
            params.append(experiment_name)

        if status:
            query += " AND r.status = ?"
            params.append(status)

        query += " ORDER BY r.created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_run_metrics(
        self,
        run_id: int,
        key: Optional[str] = None,
    ) -> List[Dict]:
        """Get metrics for a run.

        Args:
            run_id: Run ID
            key: Optional metric name filter

        Returns:
            List of metric dicts
        """
        query = "SELECT * FROM metrics WHERE run_id = ?"
        params: List[Any] = [run_id]

        if key:
            query += " AND key = ?"
            params.append(key)

        query += " ORDER BY timestamp"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_run_parameters(self, run_id: int) -> Dict[str, Any]:
        """Get parameters for a run.

        Args:
            run_id: Run ID

        Returns:
            Dict of parameter name to value
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key, value, value_type FROM parameters WHERE run_id = ?",
                (run_id,),
            )
            result = {}
            for row in cursor.fetchall():
                key, value, value_type = row["key"], row["value"], row["value_type"]
                # Convert back to original type
                if value_type == "int":
                    result[key] = int(value)
                elif value_type == "float":
                    result[key] = float(value)
                elif value_type == "bool":
                    result[key] = value.lower() == "true"
                elif value_type == "json":
                    result[key] = json.loads(value)
                else:
                    result[key] = value
            return result

    def compare_runs(self, run_ids: List[int]) -> Dict[str, Dict[int, Any]]:
        """Compare parameters across runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict of parameter name to {run_id: value}
        """
        result: Dict[str, Dict[int, Any]] = {}

        for run_id in run_ids:
            params = self.get_run_parameters(run_id)
            for key, value in params.items():
                if key not in result:
                    result[key] = {}
                result[key][run_id] = value

        return result

    def query_sql(self, sql: str, params: tuple = ()) -> List[Dict]:
        """Execute raw SQL query.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            List of result dicts
        """
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]


class RunContext:
    """Context for logging within a run."""

    def __init__(self, tracker: SQLiteTracker, run_id: int):
        """Initialize run context.

        Args:
            tracker: Parent SQLiteTracker
            run_id: Run ID
        """
        self.tracker = tracker
        self.run_id = run_id
        self._step = 0

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        self.tracker.log_metric(self.run_id, key, value, step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        self.tracker.log_metrics(self.run_id, metrics, step)

    def log(self, metrics: Dict[str, float]) -> None:
        """Log metrics at current step and increment step."""
        self.tracker.log_metrics(self.run_id, metrics, self._step)
        self._step += 1

    def log_artifact(
        self,
        file_path: Union[str, Path],
        artifact_type: str = "file",
        **kwargs,
    ) -> None:
        """Log an artifact."""
        self.tracker.log_artifact(
            self.run_id, file_path, artifact_type=artifact_type, **kwargs
        )


def track_hydra_run(
    experiment_name: str = "default",
    db_path: str = "experiments.db",
):
    """Decorator to automatically track Hydra experiments.

    Example:
        >>> @hydra.main(config_path="conf", config_name="config")
        ... @track_hydra_run(experiment_name="my-experiment")
        ... def train(cfg):
        ...     pass

    Args:
        experiment_name: Experiment name
        db_path: Path to SQLite database

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(cfg, *args, **kwargs):
            tracker = SQLiteTracker(db_path)

            with tracker.start_run(experiment_name, config=cfg) as run:
                kwargs["_sqlite_run"] = run
                kwargs["_sqlite_tracker"] = tracker
                return func(cfg, *args, **kwargs)

        return wrapper
    return decorator


# Convenience context manager
@contextmanager
def track_run(
    experiment_name: str,
    config: Optional[Any] = None,
    db_path: str = "experiments.db",
) -> Generator[RunContext, None, None]:
    """Quick context manager for tracking a run.

    Example:
        >>> with track_run("my-experiment", config=cfg) as run:
        ...     for epoch in range(100):
        ...         run.log({"loss": loss})

    Args:
        experiment_name: Experiment name
        config: Optional config dict
        db_path: Path to SQLite database

    Yields:
        RunContext for logging
    """
    tracker = SQLiteTracker(db_path)
    with tracker.start_run(experiment_name, config=config) as run:
        yield run
