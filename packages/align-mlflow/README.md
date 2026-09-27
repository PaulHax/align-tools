# align-mlflow

Load align-system open-world runs into MLflow so each episode can be read step by step in the MLflow UI.

- Each action the ADM (or driver) took becomes one **trace**.
- Each episode (one TA3 scenario session) becomes one **session**. Group traces by session in the UI to read an episode turn by turn.
- The ADM pipeline components reported in `choice_info.per_step_timing_stats` become child spans with their recorded durations.
- TA3's session alignment score for a finished episode is logged as session feedback on the episode's first step.

Runs are recognized by a `.hydra/config.yaml` whose driver is an `align_system.drivers.itm_open_world*` driver.

## Import a completed run

From the repository root, install the workspace once:

```bash
uv sync --dev
```

Import a completed run, then start the UI on the same local store:

```bash
uv run align-mlflow sync path/to/completed-run
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
```

Open the local address printed by MLflow. In the experiment (default `align-system open world`), open **Traces**, set the time range to **All** (steps are timestamped from the run's start time), and click **Group by session**.

The path can also be a parent directory containing several runs, such as a completed sweep. `sync` imports every open-world run at or below it and exits. Running the same command again resumes interrupted imports and adds only missing steps and scores. Missing episode outcomes or optional timing files do not prevent importing the recorded actions.

Both commands above use `sqlite:///mlflow.db` in the current directory. To choose another store, pass `--tracking-uri` to `sync` and the same URI to `mlflow server --backend-store-uri`. `MLFLOW_TRACKING_URI` also overrides the importer's default; keep it consistent with the UI's store.

## Optional: watch a growing run or sweep

For longer sweeps where inspecting early results could help you intervene before completion, run the watcher in a separate terminal while the UI is open:

```bash
uv run align-mlflow watch path/to/outputs
```

Use the same store as the UI. Stop watching with Ctrl-C. Watching calls the same importer as `sync`; it requires no separate service or live-mode setup. You can run `sync` after stopping it to import any remaining results.

The driver rewrites `input_output.json` after every action, so new steps appear as they are taken. Completed episode scores are picked up from `raw_align_system.log`, even when no other file changes. The watcher checks every two seconds by default; `--interval` changes that interval.

## What is logged

| Where | Contents |
| --- | --- |
| Trace name | `<step> <action type> <character>`, for example `05 TAG_CHARACTER Patient 2` |
| Inputs | Scene, the unstructured state the ADM saw, the characters as TA3 reported them, and the generic choices |
| Outputs | The concrete action with its parameters, the justification, and the rest of `choice_info` (for example predicted KDMA values and alignment votes) |
| Tags | `run`, `episode`, `step`, `scenario_id`, `alignment_target_id`, `scene_id`, `action_type`, `character_id`, `action_detail`, `chosen_by`, `adm`, `llm` |
| Session feedback | `ta3_session_alignment_score`, with the completion line as rationale and the target TA3 scored against in its metadata |

Steps the driver chose itself (ending a scene once everyone is tagged and treated, or a random fallback after a component failure) are tagged `chosen_by=driver` and have no component spans, because align-system leaves the previous step's `choice_info` on those records.

Episode boundaries, completion and scores come from `align_utils.open_world`: a new episode starts when the scenario or target changes or when TA3's clock restarts, and completions are read in order from `raw_align_system.log`. Unaligned ADMs such as the baseline record no target, so the target TA3 scored against is taken from that log.

## Notes

- Records carry no wall-clock times, so traces are laid out from the run's start time plus the recorded component durations. The order is exact; the gaps between steps are not.
- A run is identified by its ADM, TA3 profile, run directory name and TA3 session name, so syncing the same run from another path or a copy does not log it twice.
- Session names include the stable run key so separate runs with the same readable name keep separate episodes and scores.
- An import is confirmed only after its spans, tags and session have been read back from the store. Failed imports are reported; rerun `sync` to resume them. The optional watcher retries them on its next check. Unconfirmed trace attempts are removed before replay. Use one importer at a time for the same source and store.
- Traces from earlier prototypes without the `align.import_complete` tag are re-imported once. Their trace links and manually added annotations may not persist.
- With a SQLite store MLflow writes each span in its own transaction. A large backfill can take time but can be stopped and resumed.
