# align-utils

Parse align-system run data, load Open World episodes, and export CSV/TSV. Install with the [repository setup](../../README.md#install).

## Read and write files

```python
from align_utils.discovery import load_json, load_yaml, save_json, save_yaml

config = load_yaml("config.yaml")
data = load_json("data.json")
save_yaml(config, "output.yaml")
save_json(data, "output.json")
```

## Export rows

```python
from align_utils.exporters import export_to_csv, export_to_tsv

rows = [{"run": "baseline", "score": 0.8}, {"run": "aligned", "score": 0.9}]
export_to_csv(rows, "results.csv")
export_to_tsv(rows, "results.tsv")
```

## Load Open World runs

Find runs recursively and load their episodes, including unfinished runs. See [supported files](../align-mlflow/INGESTION.md#supported-files).

```python
from pathlib import Path
from align_utils.open_world import find_open_world_runs, load_run

for run_dir in find_open_world_runs(Path("outputs")):
    run = load_run(run_dir)
    for episode in run.episodes:
        score = episode.outcome.session_alignment_score if episode.outcome else None
        print(episode.scenario_id, episode.alignment_target_id, len(episode.records), score)
```

- **Records:** actions retain their order, including repeats. `record.source` keeps original JSON values and unknown fields; model dumps exclude it.
- **Episodes:** scenario or target changes start a new episode. A clock reset in the same or initial scene starts another; entering a later scene can reset its clock within the current episode. Missing scene metadata uses the clock-only fallback.
- **Scores:** matching completion lines in `raw_align_system.log` supply `episode.outcome`. A mismatch sets `run.score_warning` and omits scores while retaining actions. The scored target is `episode.outcome.alignment_target_id`, including for unaligned ADMs.
- **Driver actions:** `record.chosen_by_driver` marks actions taken by the driver; their `choice_info` may describe the previous step.
