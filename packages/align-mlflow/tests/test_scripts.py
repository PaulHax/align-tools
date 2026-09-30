"""Repository scripts against an explicitly configured temporary store."""

import json
import os
import re
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest
from dotenv import dotenv_values
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
        settings = dotenv_values(SCRIPTS.parent / "examples/server.env.example")
        settings.update(
            MLFLOW_BACKEND_STORE_URI=env["MLFLOW_TRACKING_URI"],
            MLFLOW_ARTIFACTS_DESTINATION=str(artifacts),
            MLFLOW_HOST="127.0.0.1",
            MLFLOW_PORT="0",
            MLFLOW_SERVER_ALLOWED_HOSTS="127.0.0.1:*",
            MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="http://127.0.0.1:*",
        )
        config = tmp_path / "server.env"
        config.write_text(
            "".join(f"{key}={json.dumps(value)}\n" for key, value in settings.items())
        )
        command = [
            *uv,
            "mlflow",
            "--env-file",
            str(config),
            "server",
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
        session_view = next(
            line
            for line in views.stdout.splitlines()
            if line.startswith("Session comparison:")
        )
        assert "groupBy=session" in session_view
        saved = client.get_experiment(experiment.experiment_id).tags
        assert {
            json.loads(value)["name"]
            for key, value in saved.items()
            if key.startswith("mlflow.tracesV4ViewState.")
        } == {"Open-world actions", "Episodes by ADM and target", "Session comparison"}
        # A renamed preset belongs to the user and cannot be silently overwritten.
        key = "mlflow.tracesV4ViewState.open-world-session-comparison"
        custom = json.loads(saved[key])
        custom["name"] = "Team decisions"
        client.set_experiment_tag(experiment.experiment_id, key, json.dumps(custom))
        first_key = "mlflow.tracesV4ViewState.open-world-actions"
        first_view = json.loads(saved[first_key])
        first_view["state"] = "{}"
        client.set_experiment_tag(
            experiment.experiment_id, first_key, json.dumps(first_view)
        )
        before = client.get_experiment(experiment.experiment_id).tags
        collision = subprocess.run(
            [str(SCRIPTS / "views.sh"), url],
            cwd=tmp_path,
            env={**env, "MLFLOW_TRACKING_URI": url},
            capture_output=True,
            text=True,
        )
        assert collision.returncode != 0
        assert "Team decisions" in collision.stderr
        assert "No views changed" in collision.stderr
        assert client.get_experiment(experiment.experiment_id).tags == before
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


def test_view_setup_uses_standard_tracking_credentials(tmp_path):
    class VersionEndpoint(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer test-credential":
                self.send_error(401)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"3.16.1")

        def log_message(self, *_args):
            pass

    uri = f"sqlite:///{tmp_path}/tracking.db"
    client = MlflowClient(tracking_uri=uri)
    eid = client.create_experiment(
        "align-system open world", artifact_location=str(tmp_path / "artifacts")
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), VersionEndpoint) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = subprocess.run(
                [str(SCRIPTS / "views.sh"), f"http://127.0.0.1:{server.server_port}"],
                cwd=tmp_path,
                env={
                    **environment(),
                    "MLFLOW_TRACKING_URI": uri,
                    "MLFLOW_TRACKING_TOKEN": "test-credential",
                },
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert "Session comparison:" in result.stdout
            assert len(client.get_experiment(eid).tags) == 3
        finally:
            server.shutdown()
            thread.join()
