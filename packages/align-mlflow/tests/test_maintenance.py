"""Local statistics refresh and import status reporting."""

import sqlite3
from contextlib import closing

import pytest
from click.testing import CliRunner
from open_world_fixtures import completion_line, episode_records, write_run

import align_mlflow.cli as cli_module
from align_mlflow.maintenance import refresh_sqlite_statistics
from mlflow import MlflowClient


def test_refresh_updates_statistics_without_changing_data(tmp_path):
    database = tmp_path / "store with spaces.db"
    uri = f"sqlite:///{database}"
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE traces (id TEXT PRIMARY KEY, review TEXT)")
        db.execute("INSERT INTO traces VALUES ('original-id', 'keep this note')")
    assert refresh_sqlite_statistics(uri)
    with closing(sqlite3.connect(database)) as db, db:
        assert db.execute("SELECT * FROM traces").fetchall() == [
            ("original-id", "keep this note")
        ]
        assert (
            db.execute("SELECT stat FROM sqlite_stat1 WHERE tbl='traces'")
            .fetchone()[0]
            .split()[0]
            == "1"
        )
        db.execute("INSERT INTO traces VALUES ('new-id', 'new note')")
    assert refresh_sqlite_statistics(uri)
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT * FROM traces ORDER BY id").fetchall() == [
            ("new-id", "new note"),
            ("original-id", "keep this note"),
        ]
        assert (
            db.execute("SELECT stat FROM sqlite_stat1 WHERE tbl='traces'")
            .fetchone()[0]
            .split()[0]
            == "2"
        )


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:5000",
        "https://example.com",
        "databricks",
        "sqlite:///:memory:",
    ],
)
def test_refresh_does_not_open_http_or_memory_stores(monkeypatch, uri):
    def unexpected_connection(*args, **kwargs):
        pytest.fail("Skipped stores must not open a local database")

    monkeypatch.setattr(sqlite3, "connect", unexpected_connection)
    assert refresh_sqlite_statistics(uri) is False


def test_refresh_never_creates_a_missing_database(tmp_path):
    database = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        refresh_sqlite_statistics(f"sqlite:///{database}")
    assert not database.exists()


def test_statistics_failure_does_not_report_a_successful_import_as_failed(
    tmp_path, monkeypatch
):
    run = write_run(
        tmp_path / "2026-08-31__09-17-11",
        episode_records(None)[:1],
        [completion_line("target", 0.8)],
    )
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment(
        "status test", artifact_location=(tmp_path / "artifacts").as_uri()
    )

    def unavailable_statistics(_uri):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cli_module, "refresh_sqlite_statistics", unavailable_statistics)
    result = CliRunner().invoke(
        cli_module.cli,
        ["sync", str(run), "--tracking-uri", uri, "--experiment", "status test"],
        env={"MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    assert result.exit_code == 0, result.output
    assert (
        "WARNING: SQLite statistics refresh failed: database is locked" in result.output
    )
    assert "Imported data is retained." in result.output
    assert (
        "Import successful: 1 run, 1 new traces, 1 new episode scores." in result.output
    )
    experiment = client.get_experiment_by_name("status test")
    traces = client.search_traces(locations=[experiment.experiment_id])
    assert len(traces) == 1
    assert traces[0].info.assessments[0].value == 0.8
