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

   # All runs under a parent directory:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/

   # One run:
   ./packages/align-mlflow/scripts/ingest.sh /path/to/ow_part3_live_GOOD_sept25/baseline-worldstate/APRIL_OPENWORLD3/2026-09-24__16-05-25/
   ```

   Parent directories are scanned recursively. Each run folder contains the files below and becomes one MLflow run. Pass a **directory**, not `input_output.json`.

3. **View results:** [Open World sessions](http://10.50.57.47:5000/#/experiments/1/traces?workflowType=genai&startTimeLabel=ALL&groupBy=session).

Rerun the command to resume or add records. Run **one bulk import at a time**; browsing and annotations can continue.

## Supported files

| File in each run folder | Required | Used for |
| --- | --- | --- |
| `input_output.json` | Yes | Action inputs, outputs, and component data |
| `.hydra/config.yaml` | Yes | Driver and ADM settings; uploaded as an artifact |
| `.hydra/overrides.yaml`, `.hydra/hydra.yaml` | No | Launch settings; uploaded when present |
| `meta.json` | No | Version and TA3 username; uploaded when present |
| `timing.json` | No | Recorded action durations when aligned with the saved actions |
| `align_system.log` | No | Recorded start date when the run folder has no timestamp |
| `raw_align_system.log` | No | Decision evidence and TA3 completion scores; uploaded when present |

- **Driver:** `driver._target_` in `.hydra/config.yaml` must start with `align_system.drivers.itm_open_world`. Other drivers and folders missing required files are skipped.
- **Import location:** run the script on a computer that can read the source directory. Relative paths resolve from your current directory.
- **MLflow mapping:** source folder → run; episode → session; action → trace. Original action JSON is available in each trace's `source` fields. Source files stay unchanged; the entire directory is not archived.
- **Evidence:** additional outputs show recorded `choice_info` and verified decision-log excerpts. Unrecognized or ambiguous logs remain unassigned.
- **Durations:** optional timing data must align with the saved actions. Missing or invalid timing produces a warning and an unknown duration with a minimum display placeholder. Reruns retain captured durations; appended actions use raw timing only when it agrees with previously measured durations.
- **Timestamps:** absolute times are synthetic, based on the run folder's timestamp, then the first dated `align_system.log` entry, then the config file's modification time. Dates without a timezone use local time. Reruns retain the captured origin.

Available raw logs are saved automatically as downloadable run artifacts. Watch retains a full snapshot whenever the log changes.

If saved timing cannot be recovered, imports warn and new actions resume after the last confirmed trace with unknown durations. Existing traces keep their timestamps.

## Server and experiment

- **`MLFLOW_TRACKING_URI`:** server URL. Database writes and artifact uploads go through HTTP. Existing `file://` artifact locations still require filesystem access until migrated.
- **`MLFLOW_EXPERIMENT_NAME`:** experiment to receive the runs; defaults to `align-system open world`. Keep `Open World Phase 2` to compare with existing results; another name creates a separate collection.
- **Local storage:** use an absolute `sqlite:////path/to/mlflow.db` URI. Artifacts are stored beside the database. After large HTTP imports, see the [server command](SERVER.md#slow-sqlite-trace-search) if trace searches are slow.

## Comparing ADMs

Keep runs that answer the same evaluation question in one experiment. Each align-system invocation folder becomes a run, with sessions for its scenario and target variations. A Hydra multirun with separate invocation folders produces separate MLflow runs. Use a separate experiment for a different benchmark, evaluation protocol, or study.

Use [Open World results](SESSION-UI.md#open-world-results) to compare recorded scores and inspect the sessions behind them. Match scenario and target across runs before comparing ADMs. The target recorded on an action and the target reported by TA3 are separate evidence; a mismatch needs investigation before treating results as a comparison of the requested targets.

Use column-header filters or **Filters** to focus on a run, scenario, or target. See the [UI guide](SESSION-UI.md) for session comparisons and saved views.

### Run summary scores

In MLflow, switch to **Model training**, open **Runs**, select the runs,
and choose **Compare**. The Metrics section shows their means and coverage side
by side. In the comparison chart, choose `ta3_alignment_mean` as the metric.

Successful imports refresh these metrics on each MLflow run:

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
The mismatch count compares recorded action and assessment metadata; zero does
not establish which target governed the ADM's decisions.

A temporarily missing source log does not erase recorded scores. Run means are
comparable only for the same scenario/target mix and comparable score coverage;
the importer does not adjust for different case mixes or infer missing outcomes.

## Results and reruns

- **`Import successful`:** counts new traces and scores; zero means nothing new was added.
- **`Session scores skipped`:** actions were imported, but the optional log could not be matched. Check `raw_align_system.log` and rerun to add scores.
- **`FAILED`:** identifies the run and error; the final summary exits nonzero. Fix the problem and rerun. `No open-world runs found` means nothing was imported; check the path and driver.
- **Resume:** confirmed actions retain their IDs. Preserve run folder names and provenance files when copying. Changed provenance is rejected; previously imported actions are not replaced. Use a separate experiment for changed results.
- **Run identity:** the resolved ADM name, profile, folder name, and recorded TA3 username identify a run. Preserve them when resuming an import; changing them creates a different run.
- **Avoid duplicates:** never overlap imports of the same source run into the same experiment, including with PostgreSQL.

## CLI options

`ingest.sh` takes one directory; use the environment variables above for its server and experiment. With the same exports, the CLI also accepts flags:

```bash
uv run --no-sync align-mlflow sync /path/to/runs --tracking-uri http://10.50.57.47:5000 --experiment 'Open World Phase 2'
uv run --no-sync align-mlflow watch /path/to/growing-runs --interval 2
```

Flags override environment variables. Stop `watch` with **Ctrl-C**. Append `--help` to either CLI command for options.
