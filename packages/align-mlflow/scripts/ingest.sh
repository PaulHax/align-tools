#!/usr/bin/env bash
# Import Open World runs recursively: pass one run folder or a parent, not a JSON file.
# Each run needs input_output.json and .hydra/config.yaml with an itm_open_world driver.
# Set MLFLOW_TRACKING_URI (server URL) and MLFLOW_EXPERIMENT_NAME; see ../INGESTION.md.

set -euo pipefail

usage() {
    printf 'Usage: %s SOURCE_DIRECTORY\n\n' "$0"
    printf '%s\n' \
        'Pass one run folder or a parent; nested runs are scanned recursively.' \
        'Pass a directory, not input_output.json.' \
        'MLFLOW_TRACKING_URI: server URL or absolute SQLite URI (required).' \
        'MLFLOW_EXPERIMENT_NAME: experiment (default: align-system open world).'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to the destination database or server}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
source_dir="$(cd -- "$1" && pwd)"

export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONUNBUFFERED=1
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-align-system open world}"

# Choose artifact storage once; subsequent imports reuse the experiment.
uv run --project "$repo_dir" --no-sync python - <<'PY'
import os
import sqlite3

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
name = os.environ["MLFLOW_EXPERIMENT_NAME"]
if client.get_experiment_by_name(name) is None:
    client.create_experiment(name, artifact_location=artifact_root)
if tracking_uri.startswith("sqlite:"):
    with sqlite3.connect(make_url(tracking_uri).database, timeout=30) as database:
        database.execute("PRAGMA journal_mode=WAL")
PY

printf 'Importing %s into experiment %s\n' "$source_dir" "$MLFLOW_EXPERIMENT_NAME"
exec uv run --project "$repo_dir" --no-sync align-mlflow sync "$source_dir"
