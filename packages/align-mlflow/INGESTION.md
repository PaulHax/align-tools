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
- **Metadata:** ADM settings or recorded driver, architecture, toolkit, and chat-model settings supply recorded labels. The resolved ADM, profile, folder name, and TA3 username identify each imported run.
- **Evidence:** additional outputs expose arbitrary recorded `choice_info` through the existing JSON viewer. Plain-text excerpts include source references only when action association is verified. Unrecognized or ambiguous logs remain unassigned.
- **Durations:** timing is optional. Missing or invalid values produce a warning and an unknown duration with a minimum display placeholder. New imports without component timings use aligned `raw_times_s` values for whole actions. Reruns preserve captured durations, including unknowns; appended actions use raw timing only when the complete file agrees with previously measured durations.
- **Timestamps:** absolute times are synthetic. The origin is the run folder's timestamp, then the first dated `align_system.log` entry, then the config file's modification time. Dates without a timezone use local time. Reruns retain the captured origin.

To retain the full raw log and its association index as downloadable run artifacts, use `--include-log` with the CLI, or set `MLFLOW_INCLUDE_LOG=1` with `ingest.sh`. Snapshots use content digests and leave launch provenance unchanged. With `watch --include-log`, each changed log is retained as a full snapshot; use a final `sync --include-log` to retain one completed snapshot. Without this option, the complete log remains only in its source directory. Recorded configuration, logged assessments, and evaluation targets are separate evidence; the importer does not infer which target governed a decision.

If a saved timing snapshot cannot be recovered, imports warn and new actions resume after the last confirmed trace with unknown durations. Existing traces keep their timestamps.

## Server and experiment

- **`MLFLOW_TRACKING_URI`:** server URL. Database writes and artifact uploads go through HTTP. Existing `file://` artifact locations still require filesystem access until migrated.
- **`MLFLOW_EXPERIMENT_NAME`:** experiment to receive the runs; defaults to `align-system open world`. Keep `Open World Phase 2` to compare with existing results; another name creates a separate collection.
- **Local storage:** an absolute `sqlite:////path/to/mlflow.db` URI prepares adjacent artifacts, enables WAL, and refreshes query statistics after ingestion. **HTTP imports skip statistics refresh**; see the [server command](SERVER.md#slow-sqlite-trace-search).

## Comparing ADMs

Keep invocations that answer the same evaluation question in one experiment. For example:

```text
Experiment: Open World Phase 2
  Run: ADM A, Qwen, invocation 3813
    Session: desert, merit 0.0, episode 1
    Session: desert, merit 0.2, episode 2
  Run: ADM B, Qwen, invocation 3817
    Session: desert, merit 0.0, episode 1
    Session: desert, merit 0.2, episode 2
```

A run is one align-system invocation, usually one Hydra call with one ADM and model. Its sessions contain the scenario and alignment-target variations. Each action is a trace within its session. A Hydra multirun that produces separate invocation folders produces separate MLflow runs.

This follows [MLflow's tracking concepts](https://mlflow.org/docs/latest/ml/tracking/): a run records an execution, and an experiment groups comparable runs. A new benchmark, evaluation protocol, or study can justify a separate experiment. Creating an experiment for every invocation makes routine comparisons harder to browse.

Use **ADM performance** in the [saved views](SESSION-UI.md#saved-views) for one row per episode, including the recorded TA3 score and its evaluated target. Match scenario and target across invocations before comparing ADMs. The target recorded on an action and the target reported by TA3 are separate evidence; a mismatch needs investigation before treating results as a comparison of the requested targets.

Use **Run:** views to focus on one invocation and **Session comparison** to return to the experiment-wide view. These are filters within the same experiment. They do not copy results or create another experiment.

### Run summary scores

In MLflow, switch to **Model training**, open **Runs**, select the invocations,
and choose **Compare**. The Metrics section shows their means and coverage side
by side. In the comparison chart, choose `ta3_alignment_mean` as the metric.

Successful imports refresh these metrics on the invocation's MLflow run. Select
runs in MLflow's run table to compare their metrics:

| Metric | Meaning |
| --- | --- |
| `ta3_alignment_mean` | Arithmetic mean of the recorded TA3 scores, with equal weight per scored episode |
| `episode_count` | Number of imported episodes |
| `scored_episode_count` | Episodes with a valid recorded TA3 score |
| `missing_score_count` | Imported episodes without a valid recorded TA3 score |
| `target_mismatch_count` | Scored episodes whose nonempty requested and evaluated targets differ |

An episode with no score is excluded from the mean, not treated as zero. A new
run without scores has no mean metric. If every score is later removed or
invalidated, refreshing sets the current mean to `NaN` (unavailable), preserving
the historical metrics. Unaligned episodes without a requested target are not
counted as mismatches. Mismatched outcomes remain in the recorded-score mean;
inspect the mismatch count before interpreting a run as evaluating its requested
targets.
The mismatch count compares recorded action and assessment metadata only; zero
does not establish which target the ADM actually used internally.

The summary reads persisted episode assessments, so a temporarily missing source
log does not erase recorded scores. Repeated imports leave unchanged metric
history untouched. Run means are comparable only for the same scenario/target
mix and comparable score coverage; the importer does not adjust for different
case mixes or infer missing outcomes.

## Results and reruns

- **`Import successful`:** counts new traces and scores; zero means nothing new was added.
- **`Session scores skipped`:** actions were imported, but the optional log could not be matched. Check `raw_align_system.log` and rerun to add scores.
- **`FAILED`:** identifies the run and error; the final summary exits nonzero. Fix the problem and rerun. `No open-world runs found` means nothing was imported; check the path and driver.
- **Resume:** confirmed actions retain their IDs. Preserve run folder names and provenance files when copying. Changed provenance is rejected; previously imported actions are not replaced. Use a separate experiment for changed results.
- **Clean imports:** changing identity fields creates a different run. Rebuild imports made with the older identity rule in an empty database or experiment.
- **Avoid duplicates:** never overlap imports of the same source run into the same experiment, including with PostgreSQL.

## CLI options

`ingest.sh` takes one directory; use the environment variables above for its server and experiment. With the same exports, the CLI also accepts flags:

```bash
uv run --no-sync align-mlflow sync /path/to/runs --tracking-uri http://10.50.57.47:5000 --experiment 'Open World Phase 2'
uv run --no-sync align-mlflow watch /path/to/growing-runs --interval 2
```

Flags override environment variables. `watch` uses the same importer; stop with **Ctrl-C**. The wrapper additionally prepares artifact storage. Append `--help` to either CLI command for options.
