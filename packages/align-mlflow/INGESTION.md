# Import Align runs into MLflow

## Quick Start

1. **Install once.** Requires Git and `uv`.

   ```bash
   git clone https://github.com/PaulHax/align-tools.git
   cd align-tools
   uv sync --frozen --dev
   ```

2. **Add runs to the existing Open World experiment.** Set the server URL and experiment; choose either source directory:

   ```bash
   export MLFLOW_TRACKING_URI='http://10.50.57.47:5000'
   export MLFLOW_EXPERIMENT_NAME='Open World Phase 2'

   # All three baseline runs:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/

   # One run:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/APRIL_OPENWORLD3/2026-09-24__16-05-25/
   ```

   The parent scans `APRIL_OPENWORLD3`, `FEB_OPENWORLD3`, and `JUNE25_OPENWORLD3` recursively. Each timestamp folder contains the files below and becomes one MLflow run. Pass a **directory**, not `input_output.json`.

3. **View results:** [Open World sessions](http://10.50.57.47:5000/#/experiments/1/traces?workflowType=genai&startTimeLabel=ALL&groupBy=session).

Rerun the command to resume or add records. Run **one bulk import at a time**; browsing and annotations can continue.

## Supported files

| File in each run folder | Required | Used for |
| --- | --- | --- |
| `input_output.json` | Yes | Action inputs, outputs, and component data → traces |
| `.hydra/config.yaml` | Yes | Driver and ADM settings; uploaded as an artifact |
| `.hydra/overrides.yaml`, `.hydra/hydra.yaml` | No | Launch settings; uploaded when present |
| `meta.json` | No | Version and TA3 username; uploaded when present |
| `timing.json` | No | Recorded action durations when aligned with the saved actions |
| `align_system.log` | No | Recorded start date when the run folder has no timestamp |
| `raw_align_system.log` | No | Decision evidence and TA3 completion scores; upload is optional |

- **Driver:** `driver._target_` in `.hydra/config.yaml` must start with `align_system.drivers.itm_open_world`. Other drivers and folders missing required files are skipped.
- **Import location:** run the script on a computer that can read the source directory. Relative paths resolve from your current directory.
- **MLflow mapping:** source folder → run; episode → session; action → trace. Traces retain original action JSON in `source` fields. Source files stay unchanged; the entire directory is not archived.
- **Metadata:** ADM settings or recorded driver, architecture, toolkit, and chat-model settings supply display labels. Labels do not change run or session identity.
- **Evidence:** additional outputs expose arbitrary recorded `choice_info` through the existing JSON viewer. Plain-text excerpts include source references only when action association is verified. Unrecognized or ambiguous logs remain unassigned.
- **Durations:** timing is optional. Missing or invalid values produce a warning and an unknown duration with a minimum display placeholder. New imports without component timings use aligned `raw_times_s` values for whole actions. Reruns preserve captured durations, including unknowns; appended actions use raw timing only when the complete file agrees with previously measured durations.
- **Timestamps:** absolute times are synthetic. The origin is the run folder's timestamp, then the first dated `align_system.log` entry, then the config file's modification time. Dates without a timezone use local time. Reruns retain the captured origin.

To retain the full raw log and its association index as downloadable run artifacts, use `--include-log` with the CLI, or set `MLFLOW_INCLUDE_LOG=1` with `ingest.sh`. Snapshots use content digests and leave launch provenance unchanged. With `watch --include-log`, each changed log is retained as a full snapshot; use a final `sync --include-log` to retain one completed snapshot. Without this option, the complete log remains only in its source directory. Recorded configuration, logged assessments, and evaluation targets are separate evidence; the importer does not infer which target governed a decision.

If a saved timing snapshot cannot be recovered, imports warn and new actions resume after the last confirmed trace with unknown durations. Existing traces keep their timestamps.

## Server and experiment

- **`MLFLOW_TRACKING_URI`:** server URL. Database writes and artifact uploads go through HTTP. Existing `file://` artifact locations still require filesystem access until migrated.
- **`MLFLOW_EXPERIMENT_NAME`:** experiment to receive the runs; defaults to `align-system open world`. Keep `Open World Phase 2` to compare with existing results; another name creates a separate collection.
- **Local storage:** an absolute `sqlite:////path/to/mlflow.db` URI prepares adjacent artifacts, enables WAL, and refreshes query statistics after ingestion. **HTTP imports skip statistics refresh**; see the [server command](SERVER.md#slow-sqlite-trace-search).

## Results and reruns

- **`Import successful`:** counts new traces and scores; zero means nothing new was added.
- **`Session scores skipped`:** actions were imported, but the optional log could not be matched. Check `raw_align_system.log` and rerun to add scores.
- **`FAILED`:** identifies the run and error; the final summary exits nonzero. Fix the problem and rerun. `No open-world runs found` means nothing was imported; check the path and driver.
- **Resume:** confirmed actions retain their IDs. Preserve run folder names and provenance files when copying. Changed provenance is rejected; previously imported actions are not replaced. Use a separate experiment for changed results.
- **Avoid duplicates:** never overlap imports of the same source run into the same experiment, including with PostgreSQL.

## CLI options

`ingest.sh` takes one directory; use the environment variables above for its server and experiment. With the same exports, the CLI also accepts flags:

```bash
uv run --no-sync align-mlflow sync /path/to/runs --tracking-uri http://10.50.57.47:5000 --experiment 'Open World Phase 2'
uv run --no-sync align-mlflow watch /path/to/growing-runs --interval 2
```

Flags override environment variables. `watch` uses the same importer; stop with **Ctrl-C**. The wrapper additionally prepares artifact storage. Append `--help` to either CLI command for options.
