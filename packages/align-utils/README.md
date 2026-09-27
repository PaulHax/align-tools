# align-utils

Utilities for parsing and processing align-system experiment data.

## Installation

```bash
pip install align-utils
```

## Usage

### Working with Models

```python
from align_utils.models import AlignExperiment, AlignDataset

experiment = AlignExperiment(
    name="test_experiment",
    version="1.0.0",
    description="A test experiment",
    parameters={"learning_rate": 0.001},
    results={"accuracy": 0.95}
)

dataset = AlignDataset(
    name="training_data",
    path="/data/train.csv",
    format="csv",
    metadata={"size": 10000}
)
```

### Parsing Files

```python
from align_utils.discovery import load_yaml, load_json, save_yaml, save_json

# Load configuration
config = load_yaml("config.yaml")
data = load_json("data.json")

# Save data
save_yaml(config, "output.yaml")
save_json(data, "output.json")
```

### Exporting Data

```python
from align_utils.exporters import export_to_csv, export_to_tsv

data = [
    {"name": "exp1", "accuracy": 0.95},
    {"name": "exp2", "accuracy": 0.97}
]

export_to_csv(data, "results.csv")
export_to_tsv(data, "results.tsv")
```

### Open-World Runs

`align_utils.open_world` loads run directories written by align-system's open-world drivers, including runs still in progress.

```python
from pathlib import Path
from align_utils.open_world import find_open_world_runs, load_run

for run_dir in find_open_world_runs(Path("outputs")):
    run = load_run(run_dir)
    for episode in run.episodes:
        score = episode.outcome.session_alignment_score if episode.outcome else None
        print(episode.scenario_id, episode.alignment_target_id, len(episode.records), score)
        for record in episode.records:
            action = record.output.action
            print("  ", action.action_type, action.character_id, action.parameters)
```

- Every record is kept in order; repeated actions are not collapsed.
- `record.source` retains the complete original JSON values, including unknown fields and explicit nulls, separately from the validated fields. It is excluded from model dumps.
- A new episode starts when the scenario or target changes or when TA3's clock (`elapsed_time`) restarts, which separates the repeated sessions of unaligned ADMs that record no target.
- Episode outcomes are the completion lines in `raw_align_system.log`, paired with episodes in order. `EpisodeOutcome.alignment_target_id` is the target TA3 scored against, known even for unaligned runs.
- `record.chosen_by_driver` marks actions the driver took itself; their `choice_info` may belong to the previous step.

## Development

This package is part of the align-tools monorepo. See the main repository for development instructions.
