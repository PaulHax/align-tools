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

## Development

This package is part of the align-tools monorepo. See the main repository for development instructions.