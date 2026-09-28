"""Repository scripts against an explicitly configured temporary store."""

import json
import os
import re
import signal
import subprocess
from pathlib import Path
from urllib.request import urlopen

import pytest
from mlflow import MlflowClient
from open_world_fixtures import completion_line, episode_records, write_run

REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPTS = REPOSITORY / "packages" / "align-mlflow" / "scripts"


def environment():
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("MLFLOW_")
        },
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }


@pytest.mark.parametrize(
    "script,args,message",
    [
        ("ingest.sh", [], "SOURCE_DIRECTORY"),
        ("ingest.sh", ["."], "MLFLOW_TRACKING_URI"),
        ("server.sh", [], "MLFLOW_TRACKING_URI"),
        ("views.sh", ["http://localhost:5000"], "MLFLOW_TRACKING_URI"),
    ],
)
def test_scripts_require_explicit_destinations(tmp_path, script, args, message):
    result = subprocess.run(
        [str(SCRIPTS / script), *args],
        cwd=tmp_path,
        env=environment(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("transport", ["sqlite", "http"])
def test_import_server_and_views_share_configured_storage(tmp_path, transport):
    target = "ADEPT-June2025-merit-0.4"
    records = episode_records(target)[:1]
    source = write_run(
        tmp_path / "source runs" / "2026-08-31__09-17-11",
        records,
        [completion_line(target, 0.8)],
    )
    storage = tmp_path / "chosen storage"
    artifacts = storage / "artifacts"
    env = {
        **environment(),
        "MLFLOW_TRACKING_URI": f"sqlite:///{storage}/tracking.db",
        "MLFLOW_EXPERIMENT_NAME": "script configuration test",
        "MLFLOW_PORT": "0",
    }

    def import_twice(import_env):
        for expected in [
            "1 new steps, 1 new episode scores",
            "0 new steps, 0 new episode scores",
        ]:
            result = subprocess.run(
                [str(SCRIPTS / "ingest.sh"), "source runs"],
                cwd=tmp_path,
                env=import_env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert expected in result.stdout

    if transport == "sqlite":
        import_twice(env)
        command = [str(SCRIPTS / "server.sh")]
    else:
        uv = ["uv", "run", "--project", str(REPOSITORY), "--no-sync"]
        prepared = subprocess.run(
            [*uv, "python", "-m", "align_mlflow.local_store"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        command = [
            *uv,
            "mlflow",
            "server",
            "--backend-store-uri",
            env["MLFLOW_TRACKING_URI"],
            "--artifacts-destination",
            prepared.stdout.strip(),
            "--default-artifact-root",
            "mlflow-artifacts:/",
            "--serve-artifacts",
            "--workers",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ]

    server = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        startup = []
        for line in server.stdout:
            startup.append(line)
            if match := re.search(
                r"Uvicorn running on (http://127\.0\.0\.1:\d+)", line
            ):
                url = match.group(1)
                break
        else:
            pytest.fail("Server did not start: " + "".join(startup))

        with urlopen(f"{url}/health") as response:
            assert response.read() == b"OK"
        if transport == "http":
            import_twice({**env, "MLFLOW_TRACKING_URI": url})
        client = MlflowClient(tracking_uri=url)
        experiment = client.get_experiment_by_name(env["MLFLOW_EXPERIMENT_NAME"])
        assert experiment.artifact_location == (
            artifacts.as_uri()
            if transport == "sqlite"
            else f"mlflow-artifacts:/{experiment.experiment_id}"
        )
        traces = client.search_traces(locations=[experiment.experiment_id])
        assert len(traces) == 1
        root = next(span for span in traces[0].data.spans if span.parent_id is None)
        assert {"input": root.inputs["source"], **root.outputs["source"]} == records[0]
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        artifact_directory = (
            artifacts if transport == "sqlite" else artifacts / experiment.experiment_id
        )
        for relative in [".hydra/config.yaml", "meta.json"]:
            copied = next(artifact_directory.glob(f"*/artifacts/source/{relative}"))
            assert copied.read_bytes() == (source / relative).read_bytes()
            if transport == "http":
                download = tmp_path / "downloads"
                download.mkdir(exist_ok=True)
                downloaded = client.download_artifacts(
                    runs[0].info.run_id, f"source/{relative}", str(download)
                )
                assert Path(downloaded).read_bytes() == (source / relative).read_bytes()

        views = subprocess.run(
            [str(SCRIPTS / "views.sh"), url],
            cwd=tmp_path,
            env={**env, "MLFLOW_TRACKING_URI": url},
            capture_output=True,
            text=True,
        )
        assert views.returncode == 0, views.stderr
        assert "Episodes by ADM and target" in views.stdout
        saved = client.get_experiment(experiment.experiment_id).tags
        assert {
            json.loads(value)["name"]
            for key, value in saved.items()
            if key.startswith("mlflow.tracesV4ViewState.")
        } == {"Open-world actions", "Episodes by ADM and target"}
        assert not (tmp_path / "mlflow.db").exists()
        assert not (tmp_path / "server.log").exists()
        assert not (tmp_path / "server.pid").exists()
    finally:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


@pytest.mark.parametrize("script", ["ingest.sh", "server.sh"])
def test_local_scripts_reject_ambiguous_storage(tmp_path, script):
    result = subprocess.run(
        [str(SCRIPTS / script), *(["."] if script == "ingest.sh" else [])],
        cwd=tmp_path,
        env={**environment(), "MLFLOW_TRACKING_URI": "sqlite:///relative.db"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "absolute SQLite file URI" in result.stderr
    assert list(tmp_path.iterdir()) == []
