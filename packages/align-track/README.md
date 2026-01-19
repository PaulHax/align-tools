# align-track

Experiment tracking and visualization for align-system using MLflow.

## Overview

The `align-track` package provides experiment tracking for align-system experiments using MLflow. Each experiment directory becomes one MLflow run, with MLflow traces capturing scene-level details (input text, choices, decisions, KDMA values).

This structure enables:
- **Run-level comparison**: Compare ADM configurations across alignment targets
- **Scene drill-down**: Explore individual scene decisions via traces
- **Searchable traces**: Filter scenes by scene_id, timing, or other attributes

## Installation

```bash
pip install align-track
```

## Development

This package is part of the align-tools monorepo and depends on `align-utils`.

For local development:
```bash
cd align-tools
uv sync --dev
```

## Quick Start

```bash
# Create a directory for MLflow data (gitignored)
mkdir -p data/mlflow

# Ingest experiments into MLflow
align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri sqlite:///data/mlflow/mlflow.db

# Launch MLflow UI
align-track ui --mlflow-uri sqlite:///data/mlflow/mlflow.db

# Open http://localhost:5000 in your browser
```

## CLI Commands

### Ingest Experiments

Ingest experiment directories into MLflow. Creates one run per experiment with traces for each scene.

```bash
# Basic usage (recommended: use data/mlflow directory)
align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri sqlite:///data/mlflow/mlflow.db

# Specify experiment name
align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri sqlite:///data/mlflow/mlflow.db \
  --experiment-name "My Experiment"

# Force re-ingest existing experiments
align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri sqlite:///data/mlflow/mlflow.db \
  --force
```

### Launch MLflow UI

```bash
# Start UI
align-track ui --mlflow-uri sqlite:///data/mlflow/mlflow.db

# Custom port and host
align-track ui --mlflow-uri sqlite:///data/mlflow/mlflow.db --port 8080 --host 0.0.0.0
```

### Search Runs

Search and filter MLflow runs from the command line.

```bash
# Search by ADM
align-track search --mlflow-uri sqlite:///data/mlflow/mlflow.db --adm pipeline_baseline

# Search by minimum score
align-track search --mlflow-uri sqlite:///data/mlflow/mlflow.db --min-score 0.8

# Search with MLflow filter string
align-track search --mlflow-uri sqlite:///data/mlflow/mlflow.db \
  --filter "params.adm = 'pipeline_baseline'"
```

### Export Runs

Export run data to CSV, TSV, or JSON.

```bash
# Export to CSV
align-track export --mlflow-uri sqlite:///data/mlflow/mlflow.db -o results.csv

# Export to JSON
align-track export --mlflow-uri sqlite:///data/mlflow/mlflow.db --format json -o results.json
```

### Generate Reports

Create pivot table reports from run data.

```bash
# Default pivot: alignment_target_id × adm, values = timing.avg_seconds
align-track report --mlflow-uri sqlite:///data/mlflow/mlflow.db -o report.xlsx

# Custom pivot configuration
align-track report --mlflow-uri sqlite:///data/mlflow/mlflow.db \
  --rows alignment_target_id \
  --cols adm \
  --values "metrics.score.alignment_score" \
  -o report.xlsx
```

### List Runs

List runs with optional align-app deep links.

```bash
align-track list-runs --mlflow-uri sqlite:///data/mlflow/mlflow.db

# With deep links to align-app
align-track list-runs --mlflow-uri sqlite:///data/mlflow/mlflow.db \
  --with-links --experiments-root /data/experiments
```

## Data Model

### MLflow Run (Experiment-Level)

Each experiment directory creates one MLflow run:

**Parameters:**
- `adm`: ADM name
- `llm_backbone`: LLM model name (if applicable)
- `alignment_target_id`: Alignment target identifier
- `kdma.*`: Individual KDMA values (e.g., `kdma.Moral judgement: 0.75`)
- `run_variant`: Experiment variant identifier
- `num_scenarios`: Total number of scenes
- `first_scenario_id`: First scenario identifier

**Metrics:**
- `timing.total_seconds`: Total runtime across all scenes
- `timing.avg_seconds`: Average time per scene
- `timing.min_seconds`: Fastest scene
- `timing.max_seconds`: Slowest scene
- `score.*`: Score metrics (if available)

**Tags:**
- `adm`: ADM name (for filtering)
- `llm`: LLM model (for filtering)
- `alignment_target_id`: Alignment target (for filtering)
- `experiment_path`: Relative path to experiment directory
- `experiment_timestamp`: Directory modification timestamp
- `has_hydra_config`: Whether .hydra/config.yaml exists

**Artifacts:**
- `input_output.json`: All scene inputs/outputs
- `timing.json`: Timing data
- `scores.json`: Scores (if available)
- `.hydra/config.yaml`: Hydra configuration (if available)

### MLflow Traces (Scene-Level)

Each scene in an experiment creates one MLflow trace linked to the parent run:

**Trace Name:** Scene ID (e.g., `June2025-SS-eval`)

**Inputs:**
- `scenario_id`: Scenario identifier
- `state`: Scene state text
- `scene_id`: Scene identifier
- `unstructured`: Scene description
- `characters`: Character information
- `choices`: Available action choices with KDMA associations

**Outputs:**
- `choice_index`: Index of chosen action
- `action`: Chosen action details (action_id, action_type, unstructured, justification)

**Attributes:**
- `scene_index`: Index in input_output.json
- `scenario_id`: Scenario identifier
- `adm`: ADM name
- `alignment_target_id`: Alignment target
- `timing_seconds`: Scene execution time
- `true_kdma_values`: Ground truth KDMA values
- `predicted_kdma_values`: ADM predicted KDMA values
- `choice_index`: Chosen action index
- `chosen_action`: Text of chosen action

**Tags:**
- `mlflow.runId`: Link to parent run
- `scene_id`: Scene identifier (for filtering)

## MLflow UI Tips

### Compare ADM Configurations
1. Go to "Runs" tab in your experiment
2. Filter by alignment target or scenario
3. Select runs to compare
4. Click "Compare" for side-by-side metrics

### Explore Scene Details
1. Go to "Traces" tab
2. Filter by `scene_id` to find specific scenes
3. Click a trace to see full input/output text, choices, and decision

### Filter Runs
Use the search bar with MLflow filter syntax:
- `params.adm = 'pipeline_baseline'`
- `params.alignment_target_id = 'MJ5'`
- `metrics.timing.avg_seconds < 30`
- `tags.llm = 'mistralai/Mistral-7B-Instruct-v0.3'`

### Group and Aggregate
Use "Group by" to organize runs by parameter values for pattern analysis.
