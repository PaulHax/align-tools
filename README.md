# align-tools

Tools for reading align-system runs and importing Open World results into MLflow.

| Package | Use |
| --- | --- |
| [align-mlflow](packages/align-mlflow/README.md) | [Import runs](packages/align-mlflow/INGESTION.md) or [manage a shared server](packages/align-mlflow/SERVER.md) |
| [align-utils](packages/align-utils/README.md) | Parse run data, load Open World episodes, and export CSV/TSV |
| [align-track](packages/align-track/README.md) | List run folders, ADMs, and alignment targets |

## Install

Requires Git and [uv](https://docs.astral.sh/uv/getting-started/installation/); Python 3.10+.

```bash
git clone https://github.com/PaulHax/align-tools.git
cd align-tools
uv sync --frozen --dev
```

Server setup also installs the session UI; follow the [server guide](packages/align-mlflow/SERVER.md#quick-start).

## Development

Run the CI checks from the repository root:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy packages/
uv run --no-sync pytest packages/
```

- **Format:** `uv run --no-sync ruff format .`
- **Package tests:** `uv run --no-sync pytest packages/align-mlflow/`
- **Dependencies:** `uv add --package align-utils requests`; `uv add --dev pytest-mock` for workspace development dependencies.
- **Contribute:** branch from `main`, run the checks, and open a pull request. Use `feat:` or `fix:` commit prefixes; mark breaking changes with `BREAKING CHANGE:`.

## Releases

The [release workflow](.github/workflows/release.yml) runs on `main`, tags semantic versions, builds the packages, and publishes [align-utils](https://pypi.org/project/align-utils/) and [align-track](https://pypi.org/project/align-track/) to PyPI. Install `align-mlflow` from this checkout.
