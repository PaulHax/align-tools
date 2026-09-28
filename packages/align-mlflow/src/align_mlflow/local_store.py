"""Filesystem layout for the local SQLite setup scripts."""

import os
from pathlib import Path

from sqlalchemy.engine import make_url


def prepare_local_store(tracking_uri: str) -> str:
    """Create artifact storage beside an explicitly located SQLite database."""
    url = make_url(tracking_uri)
    if (
        url.drivername != "sqlite"
        or not url.database
        or not Path(url.database).is_absolute()
        or url.query
    ):
        raise ValueError(
            "Local setup requires an absolute SQLite file URI without query options, "
            "for example sqlite:////path/to/mlflow.db"
        )
    artifacts = Path(url.database).parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts.as_uri()


if __name__ == "__main__":
    try:
        print(prepare_local_store(os.environ["MLFLOW_TRACKING_URI"]))
    except ValueError as error:
        raise SystemExit(str(error)) from error
