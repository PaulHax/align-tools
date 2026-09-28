#!/usr/bin/env bash
set -euo pipefail

: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to an absolute SQLite database URI}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"

export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
export MLFLOW_PORT="${MLFLOW_PORT:-5000}"
export MLFLOW_WORKERS="${MLFLOW_WORKERS:-1}"
artifact_root="$(uv run --project "$repo_dir" --no-sync python -m align_mlflow.local_store)"
exec uv run --project "$repo_dir" --no-sync mlflow server \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --default-artifact-root "$artifact_root" \
    --no-serve-artifacts
