#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: %s UI_URL\n' "$0"
    exit 0
fi
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s UI_URL\n' "$0" >&2
    exit 2
fi
: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to the destination database or server}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
export MLFLOW_DISABLE_AGENT_HINT=1
exec uv run --project "$repo_dir" --no-sync python \
    "$script_dir/../examples/configure_views.py" \
    --tracking-uri "$MLFLOW_TRACKING_URI" \
    --ui-url "$1" \
    --experiment "${MLFLOW_EXPERIMENT_NAME:-align-system open world}"
