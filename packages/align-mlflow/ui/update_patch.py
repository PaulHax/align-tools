"""Regenerate and verify the maintained patch from an edited MLflow source tree."""

import argparse
import difflib
import subprocess
import tarfile
import tempfile
from pathlib import Path

from install import ARCHIVE_SHA256, PATCH, VERSION, digest

SOURCE_PREFIX = "mlflow/server/js/src/"


def prepare_source(source: Path, archive: Path, patch: Path = PATCH) -> None:
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("The archive does not match the pinned MLflow source.")
    if source.exists():
        raise ValueError("Use a new source directory outside the checkout.")
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as bundle:
            prefix = f"mlflow-{VERSION}/mlflow/server/js/"
            members = [item for item in bundle if item.name.startswith(prefix)]
            bundle.extractall(staging, members=members, filter="data")
        extracted = staging / f"mlflow-{VERSION}"
        subprocess.run(
            ["git", "apply", "--check", str(patch.resolve())],
            cwd=extracted,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(patch.resolve())], cwd=extracted, check=True
        )
        extracted.rename(source)


def update_patch(source: Path, archive: Path, output: Path) -> int:
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("The archive does not match the pinned MLflow source.")
    if not (source / SOURCE_PREFIX).is_dir():
        raise ValueError("Source must contain mlflow/server/js/src/.")
    original = {}
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name = member.name.removeprefix(f"mlflow-{VERSION}/")
            if not member.isfile() or not name.startswith(SOURCE_PREFIX):
                continue
            if ".." in Path(name).parts:
                raise ValueError("The source archive contains an unsafe path.")
            stream = bundle.extractfile(member)
            if stream is not None:
                original[name] = stream.read()
    current = {}
    for path in sorted((source / SOURCE_PREFIX).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Source files must not be symlinks: {path}")
        if path.is_file() and path.suffix != ".tsbuildinfo":
            current[path.relative_to(source).as_posix()] = path.read_bytes()
    changed = []
    lines = []
    for name in sorted(original.keys() | current.keys()):
        before, after = original.get(name, b""), current.get(name, b"")
        if before == after:
            if (name in original) != (name in current):
                raise ValueError(
                    f"Empty file additions or removals need an explicit patch: {name}"
                )
            continue
        changed.append(name)
        for line in difflib.unified_diff(
            before.decode().splitlines(keepends=True),
            after.decode().splitlines(keepends=True),
            fromfile=f"a/{name}" if name in original else "/dev/null",
            tofile=f"b/{name}" if name in current else "/dev/null",
        ):
            lines.append(line)
            if not line.endswith("\n"):
                lines.append("\n\\ No newline at end of file\n")
    with tempfile.TemporaryDirectory() as temporary:
        verification = Path(temporary)
        candidate = verification / "candidate.patch"
        candidate.write_text("".join(lines))
        for name in changed:
            if name in original:
                target = verification / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original[name])
        if changed:
            subprocess.run(
                ["git", "apply", "--check", str(candidate)],
                cwd=verification,
                check=True,
            )
            subprocess.run(
                ["git", "apply", str(candidate)], cwd=verification, check=True
            )
        for name in changed:
            target = verification / name
            if name in current:
                if target.read_bytes() != current[name]:
                    raise ValueError(f"The patch did not reproduce {name}.")
            elif target.exists():
                raise ValueError(f"The patch did not remove {name}.")
        output.write_bytes(candidate.read_bytes())
    return len(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "update"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.output is not None:
            parser.error("--output applies only to update.")
        prepare_source(args.source.resolve(), args.archive.resolve())
        print(f"Prepared editable frontend sources in {args.source}.")
    else:
        output = args.output or PATCH
        count = update_patch(args.source.resolve(), args.archive.resolve(), output)
        print(f"Updated and verified {count} source files in {output}.")


if __name__ == "__main__":
    main()
