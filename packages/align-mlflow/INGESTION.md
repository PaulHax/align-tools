# Ingestion

For the short directory-import workflow, start with the [researcher quickstart](RESEARCHER-QUICKSTART.md).

Run `uv sync --frozen --dev` from the repository root, then [configure storage](SERVER.md#configure-storage) in the shell used for ingestion.

```bash
./packages/align-mlflow/scripts/ingest.sh /path/to/runs
./packages/align-mlflow/scripts/ingest.sh ../data/some-sweep
```

The source directory is required. Relative paths resolve from your current directory. The importer searches nested folders for recognized open-world runs; unsupported drivers are skipped.

Recognized runs contain `input_output.json` and `.hydra/config.yaml` selecting an open-world driver. Keep the complete folder, including `meta.json` and `raw_align_system.log` when present. The importer preserves action records in traces and copies available Hydra files and `meta.json`; full-folder archival is a separate operation.

New traces include readable episode cards with situation text, the chosen action, and its justification. Original JSON remains available in each trace's `source` fields. Pipeline spans include component evidence from `choice_info` and matched logs.

## Destination

`MLFLOW_TRACKING_URI` is the only required environment variable. With an absolute SQLite URI, the script creates the database's parent directory and places artifacts beside it in `artifacts/`. The server script uses the same setting. The server need not be running during direct SQLite ingestion.

To import through an existing server, set `MLFLOW_TRACKING_URI` to its HTTP URL. New experiments then use that server's artifact configuration. With direct file artifacts, the importer must also have access to their filesystem location.

`MLFLOW_EXPERIMENT_NAME` optionally selects the experiment; its default is `align-system open world`. Existing experiments retain their recorded artifact locations. Changing settings does not relocate their files.

## Repeating an import

Repeat the command to resume or add runs. Confirmed actions retain their IDs, and copied run folders are deduplicated. Do not overlap imports of the same source run into the same experiment: resume checks are not atomic. For the initial shared SQLite setup, run bulk imports one at a time to limit write contention; browsing and annotations can continue. Editing previously imported actions requires a separate experiment.

Output goes to the terminal. Redirect it to a chosen log file if needed. Run summaries print after the scan finishes. On failure, address the reported error and rerun.

Successful runs report new steps and episode scores. `not synced (...)` reports an error and the command exits nonzero. `No open-world runs found` can exit successfully without importing anything; check the directory and Hydra driver configuration. Preserve run folder names and provenance files when copying runs so deduplication can recognize them.

## Underlying CLI

For an existing experiment, the CLI also reads `MLFLOW_TRACKING_URI`:

```bash
uv run align-mlflow sync /path/to/runs
uv run align-mlflow watch /path/to/growing-runs
```

Use `--experiment` or `--tracking-uri` to override CLI settings. The setup script additionally creates the experiment with the chosen artifact location and enables WAL for direct SQLite imports.
