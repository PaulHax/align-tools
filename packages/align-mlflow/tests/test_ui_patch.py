"""Check that published patches reproduce edited sources without losing files."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def load_updater(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "ui"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "update_patch", directory / "update_patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_archive(tmp_path, files):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(f"mlflow-3.17.2/mlflow/server/js/src/{name}")
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return archive


def test_patch_roundtrip_includes_added_changed_and_removed_sources(
    tmp_path, monkeypatch
):
    updater = load_updater(monkeypatch)
    archive = source_archive(
        tmp_path, {"changed.ts": b"before", "removed.ts": b"remove\n"}
    )
    monkeypatch.setattr(
        updater, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    source = tmp_path / "source"
    frontend = source / "mlflow/server/js/src"
    frontend.mkdir(parents=True)
    (frontend / "changed.ts").write_bytes(b"after\n")
    (frontend / "added.ts").write_bytes(b"new source\n")
    output = tmp_path / "sessions.patch"
    assert updater.update_patch(source, archive, output) == 3
    patch = output.read_text()
    assert "No newline at end of file" in patch
    assert "new source" in patch
    assert "-remove" in patch


def test_wrong_archive_preserves_existing_patch(tmp_path, monkeypatch):
    updater = load_updater(monkeypatch)
    archive = source_archive(tmp_path, {"example.ts": b"original\n"})
    monkeypatch.setattr(updater, "ARCHIVE_SHA256", "0" * 64)
    output = tmp_path / "sessions.patch"
    output.write_text("existing patch\n")
    with pytest.raises(ValueError, match="pinned MLflow"):
        updater.update_patch(tmp_path / "source", archive, output)
    assert output.read_text() == "existing patch\n"


def test_prepare_applies_current_patch_in_a_separate_source_tree(tmp_path, monkeypatch):
    updater = load_updater(monkeypatch)
    archive = source_archive(tmp_path, {"example.ts": b"original\n"})
    monkeypatch.setattr(
        updater, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    patch = tmp_path / "sessions.patch"
    patch.write_text(
        "--- a/mlflow/server/js/src/example.ts\n"
        "+++ b/mlflow/server/js/src/example.ts\n"
        "@@ -1 +1 @@\n-original\n+current frontend\n"
    )
    source = tmp_path / "edited"
    updater.prepare_source(source, archive, patch)
    assert (
        source / "mlflow/server/js/src/example.ts"
    ).read_text() == "current frontend\n"
    with pytest.raises(ValueError, match="new source directory"):
        updater.prepare_source(source, archive, patch)
