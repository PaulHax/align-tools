#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"

uv sync --project "$repo_dir" --frozen --dev
exec "$script_dir/install-ui.sh"
