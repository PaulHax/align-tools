# Import Align runs into MLflow

## Quick Start

1. **Install the importer once.** Requires Git and `uv`.

   ```bash
   git clone https://github.com/PaulHax/align-tools.git
   cd align-tools
   uv sync --frozen --dev
   ```

2. **Add runs to the existing Open World experiment.** From the repository root, set the MLflow server URL and experiment name:

   ```bash
   export MLFLOW_TRACKING_URI='http://10.50.57.47:5000'
   export MLFLOW_EXPERIMENT_NAME='Open World Phase 2'

   # All three baseline runs:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/

   # One run:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/APRIL_OPENWORLD3/2026-09-24__16-05-25/
   ```

   The importer scans recursively. Each timestamp folder is one run; its files are listed below.

3. **Open the results:** [Open World sessions](http://10.50.57.47:5000/#/experiments/1/traces?workflowType=genai&startTimeLabel=ALL&groupBy=session). This opens **Open World Phase 2** with **All time** selected and traces grouped by **session**.

**Rerun step 2** to resume or add records. Initially, run **one bulk import at a time**; browsing and annotations can continue.

## Supported inputs

| File | Required | Contents / use |
| --- | --- | --- |
| `input_output.json` | Yes | Action inputs, outputs, and component data → traces. |
| `.hydra/config.yaml` | Yes | Driver and ADM settings; uploaded as an artifact. |
| `.hydra/overrides.yaml`, `.hydra/hydra.yaml` | No | Launch settings; uploaded when present. |
| `meta.json` | No | Version and TA3 username; uploaded as an artifact. |
| `raw_align_system.log` | No | Decision evidence and TA3 session scores; not uploaded. |

- **Driver:** `.hydra/config.yaml` must select an `align_system.drivers.itm_open_world*` driver.
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
