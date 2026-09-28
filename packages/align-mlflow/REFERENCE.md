# MLflow adapter reference

## Saved views

For an episode comparison table, use ungrouped **Traces**, filter tag `step = 1`, and select `adm`, `llm`, `align.source_version`, `scenario_id`, and `alignment_target_id` under **Display > Columns**. Hide Input and Output, add tag filters as needed, and save the view. Each Session link opens the complete episode. The table's total counter is not an episode count.

For MLflow **3.16.1**, install the [example definitions](examples/views.json):

```bash
uv run python packages/align-mlflow/examples/configure_views.py \
  --tracking-uri "$MLFLOW_TRACKING_URI" --ui-url http://localhost:5000
```

Supply the destination tracking URI, UI URL, and optional `--experiment`. The script updates matching named views and prints their links. It uses MLflow's internal saved-view format, checks the server version, and runs separately from ingestion. Use the manual recipe for other versions.

Standard and assessment columns can be reordered; custom tag columns only support show/hide. Grouped session headers omit custom tags. Column widths are browser preferences. Saved table views do not customize the original JSON shown in episode turn cards.

`adm` names the driving implementation; `align.source_version` is the producer's align-system version, not a separate ADM release. Repeating `sync` adds the version to confirmed traces while retaining their IDs, payloads, and assessments. Target IDs are preserved as recorded; predicted KDMAs are not target settings. Unaligned ADMs can have no input target even when completion feedback names a scored target.

## What is logged

| Where | Contents |
| --- | --- |
| Trace name | `<step> <action type> <character>`, for example `05 TAG_CHARACTER Patient 2` |
| Inputs | The original record's complete `input` object, including full state and choice objects |
| Outputs | Every other field of the original record, including `output`, `choice_info`, `label`, and any unknown fields |
| Tags | `run`, `episode`, `step`, `scenario_id`, `alignment_target_id`, `scene_id`, `action_type`, `character_id`, `action_detail`, `chosen_by`, `adm`, `llm`, `align.source_version` |
| Session feedback | `ta3_session_alignment_score`, with the completion line as rationale and the target TA3 scored against in its metadata |

The full source JSON record can be reconstructed as `{"input": root.inputs, **root.outputs}`. Values, nulls, and unknown fields are preserved before model validation can normalize them. Trace names, tags, and component spans are derived navigation aids. JSON whitespace and formatting are not preserved in traces.

Steps the driver chose itself (ending a scene once everyone is tagged and treated, or a random fallback after a component failure) are tagged `chosen_by=driver` and have no component spans, because align-system leaves the previous step's `choice_info` on those records. That original metadata is retained, with the root attribute `align.choice_info_applies_to_action=false` to distinguish it from evidence for the current action.

Episode boundaries, completion and scores come from `align_utils.open_world`: a new episode starts when the scenario or target changes or when TA3's clock restarts, and completions are read in order from `raw_align_system.log`. Unaligned ADMs such as the baseline record no target, so the target TA3 scored against is taken from that log.

## Source configuration and provenance

Open the trace's source run to find artifacts with the original relative paths:

```text
source/
  .hydra/
    config.yaml
    overrides.yaml
    hydra.yaml
  meta.json
```

Available files are copied byte for byte into the run's configured MLflow artifact storage. They are independent copies, not links; source files are unchanged. The run records the first source path, producer version when present, ADM, model, and profile as searchable tags. Copies of the same source run share one MLflow run. A run marked `FINISHED` means its import completed, including when importing a simulation still in progress.

File hashes prevent repeated uploads. Missing optional files do not block ingestion, and later arrivals are picked up by `sync` or `watch`. If an already archived file changes, import reports a provenance conflict and keeps the archived bytes. Source configuration is expected to remain fixed for a run.

This preserves the recorded launch recipe. Hydra files can contain unresolved references, and the producer version alone does not capture the full execution environment or guarantee a reproducible rerun. The importer does not resolve configuration or add producer metadata. It preserves every action record in traces but does not archive the entire `input_output.json`, raw log, or all other files in the source folder.

## Notes

- Records carry no wall-clock times, so traces are laid out from the run's start time plus the recorded component durations. The order is exact; the gaps between steps are not.
- A run is identified by its ADM, TA3 profile, run directory name and TA3 session name, so syncing the same run from another path or a copy does not log it twice.
- Session names include the stable run key so separate runs with the same readable name keep separate episodes and scores.
- Records are expected to be append-only. Editing previously imported actions in place is not detected; use a new experiment for a revised dataset.
- An import is confirmed only after its spans, tags, session, and source-run association have been read back from the store. Failed imports are reported; rerun `sync` to resume them. The optional watcher retries them on its next check. Unconfirmed trace attempts are removed before replay. Use one importer at a time for the same source and store.
- Traces without the `align.import_complete=source-record-v1` marker are re-imported to preserve complete source records. Their trace links and manually added annotations may not persist.
- With a SQLite store MLflow writes each span in its own transaction. A large backfill can take time but can be stopped and resumed.
