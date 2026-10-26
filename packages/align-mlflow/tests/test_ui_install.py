"""Exercise cached frontend installation without downloading or compiling."""

import importlib.util
import io
import subprocess
import tarfile
from pathlib import Path

import pytest


def test_failed_build_retries_cleanly_and_reuses_ready_build(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "ui" / "install.py"
    spec = importlib.util.spec_from_file_location("ui_install", script)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)

    patch = tmp_path / "frontend.patch"
    patch.write_text(
        "diff --git a/mlflow/server/js/added.txt b/mlflow/server/js/added.txt\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/mlflow/server/js/added.txt\n"
        "@@ -0,0 +1 @@\n"
        "+patched frontend\n"
    )
    monkeypatch.setattr(installer, "PATCH", patch)
    archive = tmp_path / f"mlflow-v{installer.VERSION}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        content = b"upstream frontend\n"
        entry = tarfile.TarInfo(
            f"mlflow-{installer.VERSION}/mlflow/server/js/upstream.txt"
        )
        entry.size = len(content)
        bundle.addfile(entry, io.BytesIO(content))
    monkeypatch.setattr(installer, "ARCHIVE_SHA256", installer.digest(archive))
    monkeypatch.setattr(
        installer.subprocess, "check_output", lambda *args, **kwargs: "v24.14.0"
    )
    other_cache = tmp_path / "other-build" / "ready"
    other_cache.parent.mkdir()
    other_cache.write_text("preserved")
    calls = []
    fail_install = True

    def run(*command, cwd):
        nonlocal fail_install
        calls.append(command)
        if command[0] == "git":
            subprocess.run(command, cwd=cwd, check=True)
        elif "install" in command and fail_install:
            fail_install = False
            raise subprocess.CalledProcessError(1, command)
        elif "build" in command:
            (cwd / "build").mkdir()
            (cwd / "build" / "index.html").write_text((cwd / "added.txt").read_text())

    monkeypatch.setattr(installer, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        installer.build(tmp_path)
    unfinished = tmp_path / installer.digest(patch)
    assert (
        unfinished / f"mlflow-{installer.VERSION}/mlflow/server/js/added.txt"
    ).exists()

    output = installer.build(tmp_path)
    assert (output / "index.html").read_text() == "patched frontend\n"
    completed_calls = list(calls)
    assert installer.build(tmp_path) == output
    assert calls == completed_calls
    assert other_cache.read_text() == "preserved"
    assert installer.digest(archive) == installer.ARCHIVE_SHA256
