"""Refresh query statistics after imports into a local SQLite store."""

import sqlite3
from contextlib import closing
from pathlib import Path

from sqlalchemy.engine import make_url


def refresh_sqlite_statistics(tracking_uri: str) -> bool:
    """Analyze an existing SQLite file; HTTP and other stores are left alone."""
    if not tracking_uri.startswith("sqlite:"):
        return False
    url = make_url(tracking_uri)
    if (
        url.drivername != "sqlite"
        or not url.database
        or url.database == ":memory:"
        or url.query
    ):
        return False
    database = Path(url.database).resolve()
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=5)
    ) as db:
        with db:
            db.execute("ANALYZE")
    return True
