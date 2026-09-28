#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s SOURCE_DIRECTORY\n' "$0"
    exit 0
fi
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s SOURCE_DIRECTORY\n' "$0" >&2
    exit 2
fi

: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to the destination database or server}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
source_dir="$(cd -- "$1" && pwd)"

export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONUNBUFFERED=1
experiment="${MLFLOW_EXPERIMENT_NAME:-align-system open world}"

# Choose artifact storage once; subsequent imports reuse the experiment.
uv run --project "$repo_dir" --no-sync python - "$experiment" <<'PY'
import os
import sqlite3
import sys

from mlflow import MlflowClient
from sqlalchemy.engine import make_url

from align_mlflow.local_store import prepare_local_store

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
if tracking_uri.startswith(("http://", "https://")):
    artifact_root = None
else:
    try:
        artifact_root = prepare_local_store(tracking_uri)
    except ValueError as error:
        raise SystemExit(str(error)) from error
client = MlflowClient()
name = sys.argv[1]
if client.get_experiment_by_name(name) is None:
    client.create_experiment(name, artifact_location=artifact_root)
if tracking_uri.startswith("sqlite:"):
    with sqlite3.connect(make_url(tracking_uri).database, timeout=30) as database:
        database.execute("PRAGMA journal_mode=WAL")
PY

printf 'Importing %s into experiment %s\n' "$source_dir" "$experiment"
exec uv run --project "$repo_dir" --no-sync align-mlflow sync "$source_dir" \
    --tracking-uri "$MLFLOW_TRACKING_URI" \
    --experiment "$experiment"
