# Import Align runs into MLflow

## Quick Start

1. **Install the importer once.** Requires Git and `uv`.

   ```bash
   git clone https://github.com/PaulHax/align-tools.git
   cd align-tools
   uv sync --frozen --dev
   ```

2. **Add runs to the existing Open World experiment.** From the repository root, change only the source path:

   ```bash
   MLFLOW_TRACKING_URI='http://10.50.57.47:5000' \
   MLFLOW_EXPERIMENT_NAME='Open World Phase 2' \
   ./packages/align-mlflow/scripts/ingest.sh /data/shared/my-study
   ```

   Supply **one run folder** or a **parent directory containing several runs**.
   Keep the experiment name to compare new runs with earlier results over time.

3. **Open the results:** <http://10.50.57.47:5000>. Select **Open World Phase 2** and open **Traces**. Choose **All time** for older runs.

**Rerun step 2** to resume or add records. Initially, run **one bulk import at a time**; browsing and annotations can continue.

## Supported inputs

- **Required:** `input_output.json` and `.hydra/config.yaml` selecting an `align_system.drivers.itm_open_world*` driver.
- **Keep when present:** `meta.json`, `raw_align_system.log`, and the other Hydra files.
- **Recursive scan:** nested run folders are discovered automatically. Relative source paths resolve from the current directory.
- **Source access:** run the importer on a computer that can read the directory; the destination server receives its data through HTTP.
- **Unsupported drivers are skipped.** `No open-world runs found` can finish without importing anything; check the source path and driver.

## What gets imported

- **Run:** one source folder, with available Hydra configuration and metadata copied as artifacts.
- **Session:** one episode, with TA3 completion scores when available.
- **Trace:** one action, including readable decision cards, original JSON in `source` fields, and recorded component evidence.
- **Source files stay unchanged.** Complete folder archival is a separate operation; ordinary ingestion does not archive every original file.

## Destination and uploads

- **Shared server:** `MLFLOW_TRACKING_URI=http://10.50.57.47:5000`. Database writes and artifact uploads go through HTTP.
- **Experiment:** use `MLFLOW_EXPERIMENT_NAME='Open World Phase 2'` to reuse the existing collection. Choose another name only when you want a separate collection. Set it explicitly here; the wrapper default is `align-system open world`.
- **Local SQLite:** an absolute `sqlite:////path/to/mlflow.db` URI creates adjacent artifact storage and enables WAL. The server need not be running for direct local ingestion.
- **Existing experiments keep their artifact locations.** An older experiment with `file://` locations still requires filesystem access until migrated.

See [server management and storage](SERVER.md) for configuration and backups.

## Resume and troubleshoot

- **Repeat the command:** confirmed actions retain their IDs; unchanged copied run folders are deduplicated.
- **One importer per source/experiment:** resume checks are not atomic. Overlapping imports of the same run can race, even with PostgreSQL.
- **Preserve folder names and provenance files** when copying runs so deduplication can recognize them.
- **Changed provenance:** existing archived files are retained. Use a separate experiment for changed previously imported actions.
- **Success:** output reports new steps and episode scores.
- **Failure:** `not synced (...)` reports the error and exits nonzero. Fix the reported problem and rerun.

## Direct CLI and watching

For the shared server:

```bash
export MLFLOW_TRACKING_URI=http://10.50.57.47:5000
uv run --no-sync align-mlflow sync /path/to/runs --experiment 'Open World Phase 2'
uv run --no-sync align-mlflow watch /path/to/growing-runs --experiment 'Open World Phase 2'
```

- **`--tracking-uri`** overrides the destination; **`--experiment`** selects the experiment.
- The ingestion wrapper additionally prepares local storage or creates the experiment with the server's artifact configuration.
