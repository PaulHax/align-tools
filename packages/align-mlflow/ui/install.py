"""Build and install the pinned MLflow session UI, or restore the stock UI."""

import argparse
import hashlib
import importlib.metadata
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

VERSION = "3.16.1"
ARCHIVE_SHA256 = "362b825f2c9ffa6f85e90e3633ca12031cd8229745553ad73692b837ef3ed494"
SOURCE_URL = f"https://codeload.github.com/mlflow/mlflow/tar.gz/refs/tags/v{VERSION}"
PATCH = Path(__file__).with_name("sessions.patch")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def replace_frontend(source: Path, target: Path) -> None:
    with tempfile.TemporaryDirectory(
        dir=target.parent, prefix=".session-ui-"
    ) as temporary:
        staging = Path(temporary) / "build"
        previous = Path(temporary) / "previous"
        shutil.copytree(source, staging)
        target.rename(previous)
        try:
            staging.rename(target)
        except OSError:
            previous.rename(target)
            raise


def build(directory: Path) -> Path:
    node_version = subprocess.check_output(["node", "--version"], text=True).strip()
    major, minor, *_ = map(int, node_version.lstrip("v").split("."))
    if major != 24 or minor < 14:
        raise SystemExit("Use Node.js 24.14 or later in the 24.x series on PATH.")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"mlflow-v{VERSION}.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".download")
        print(f"Downloading MLflow {VERSION} source...", flush=True)
        with urlopen(SOURCE_URL, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.rename(archive)
    if digest(archive) != ARCHIVE_SHA256:
        raise SystemExit(f"Source checksum mismatch: remove {archive} and retry.")

    # The patch digest isolates builds when the maintained source changes.
    work = directory / digest(PATCH)
    source = work / f"mlflow-{VERSION}"
    frontend = source / "mlflow/server/js"
    ready = work / "ready"
    if not ready.exists():
        with tarfile.open(archive) as bundle:
            prefix = f"mlflow-{VERSION}/mlflow/server/js/"
            members = [
                entry for entry in bundle.getmembers() if entry.name.startswith(prefix)
            ]
            bundle.extractall(work, members=members, filter="data")
        run("git", "apply", "--check", str(PATCH), cwd=source)
        run("git", "apply", str(PATCH), cwd=source)
        yarn = ("node", "yarn/releases/yarn-4.12.0.cjs")
        run(*yarn, "install", "--immutable", cwd=frontend)
        run(*yarn, "build", cwd=frontend)
        ready.write_text(digest(PATCH) + "\n")
    return frontend / "build"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path.home() / ".cache/align-mlflow/ui" / VERSION,
        help="Download and build cache; keep outside the checkout",
    )
    parser.add_argument(
        "--restore", action="store_true", help="Restore the stock frontend"
    )
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("mlflow")
    if distribution.version != VERSION:
        parser.error(
            f"This frontend requires MLflow {VERSION}, found {distribution.version}."
        )
    target = Path(str(distribution.locate_file("mlflow/server/js/build"))).resolve()
    backup = target.with_name(f"build.stock-{VERSION}")
    if args.restore:
        if not backup.is_dir():
            parser.error("No stock frontend backup found in this Python environment.")
        replace_frontend(backup, target)
        print("Stock frontend restored. Restart MLflow and reload the browser.")
        return

    output = build(args.build_dir.expanduser().resolve())
    if not (output / "index.html").is_file():
        parser.error(
            "The build did not produce index.html; installed frontend is unchanged."
        )
    if not backup.exists():
        shutil.copytree(target, backup)
    replace_frontend(output, target)
    (target / "align-session-ui.txt").write_text(
        f"MLflow {VERSION}\npatch {digest(PATCH)}\n"
    )
    print(f"Session frontend installed in {target}.")
    print(
        "Restart MLflow and reload the browser. Database and artifacts are unchanged."
    )


if __name__ == "__main__":
    main()
