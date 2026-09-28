# Ingestion

Run `uv sync --frozen --dev` from the repository root, then [configure storage](SERVER.md#configure-storage) in the shell used for ingestion.

```bash
./packages/align-mlflow/scripts/ingest.sh /path/to/runs
./packages/align-mlflow/scripts/ingest.sh ../data/some-sweep
```

The source directory is required. Relative paths resolve from your current directory. The importer searches nested folders for recognized open-world runs; unsupported drivers are skipped.

New traces include [readable episode cards](REFERENCE.md#episode-cards) with situation text, the chosen action, and its justification. Original JSON remains available in each trace's `source` fields. Pipeline spans include [component evidence](REFERENCE.md#component-evidence) from `choice_info` and matched logs.

## Destination

`MLFLOW_TRACKING_URI` is the only required environment variable. With an absolute SQLite URI, the script creates the database's parent directory and places artifacts beside it in `artifacts/`. The server script uses the same setting. The server need not be running during direct SQLite ingestion.

To import through an existing server, set `MLFLOW_TRACKING_URI` to its HTTP URL. New experiments then use that server's artifact configuration. With direct file artifacts, the importer must also have access to their filesystem location.

`MLFLOW_EXPERIMENT_NAME` optionally selects the experiment; its default is `align-system open world`. Existing experiments retain their recorded artifact locations. Changing settings does not relocate their files.

## Repeating an import

Repeat the command to resume or add runs. Confirmed actions retain their IDs, and copied run folders are deduplicated. Use one importer at a time for a source/store. Editing previously imported actions requires a separate experiment.

Output goes to the terminal. Redirect it to a chosen log file if needed. Run summaries print after the scan finishes. On failure, address the reported error and rerun; see [recovery details](REFERENCE.md#notes).

## Underlying CLI

For an existing experiment, the CLI also reads `MLFLOW_TRACKING_URI`:

```bash
uv run align-mlflow sync /path/to/runs
uv run align-mlflow watch /path/to/growing-runs
```

Use `--experiment` or `--tracking-uri` to override CLI settings. The setup script additionally creates the experiment with the chosen artifact location and enables WAL for direct SQLite imports.
