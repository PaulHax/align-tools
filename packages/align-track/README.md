# align-track

Experiment tracking and visualization for align-system using MLflow or Aim.

## Overview

The `align-track` package provides dual-backend experiment tracking for align-system experiments:

- **MLflow** (default): Hierarchical structure with parent experiment runs and nested child scene runs, ideal for aggregate analysis
- **Aim**: Flat scene-level structure optimized for browsing and comparing large numbers of runs (10k+ scenes), with inline text viewing

Both backends track the same data (parameters, metrics, text) and support filtering by ADM, LLM, alignment target, and scene ID.

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

## Features

- **Dual Backend Support**: Choose between MLflow or Aim for experiment tracking
- **MLflow**: Hierarchical structure, industry-standard format, experiment aggregates
- **Aim**: Flat scene structure, optimized UI for 10k+ runs, inline text viewing
- **Rich Metadata**: Parameters (ADM, LLM, KDMA values), metrics (scores, timing), and tags for filtering
- **Text Data**: Scene descriptions, available choices, decisions, and justifications logged inline
- **Flexible Querying**: Filter by ADM, LLM, alignment target, or scene ID
- **Scene Comparison**: Compare how different ADM/LLM combinations perform on the same scene

## Choosing a Backend

### Use MLflow when:
- You want hierarchical experiment organization
- You need experiment-level aggregate metrics
- You're familiar with MLflow's ecosystem
- You want industry-standard ML experiment tracking

### Use Aim when:
- You're working with many scenes (6k+ runs)
- You want faster UI for browsing and comparing runs
- You need inline viewing of scene text (probe descriptions, choices, decisions)
- You want direct scene-to-scene comparison without hierarchy
- UI responsiveness with large datasets is important

Both backends can coexist - you can ingest to both for different use cases.

## Usage

### Ingest Experiments into MLflow

#### Hierarchical Ingestion (Default)

Creates parent runs for experiments with nested child runs for each scene:

```bash
# Ingest all experiments
uv run align-track ingest \
  --experiments-dir /home/paulhax/src/itm/align-browser/experiment-data \
  --mlflow-uri file:///home/paulhax/src/itm/align-data/mlruns

# Ingest single experiment directory
uv run align-track ingest \
  --experiments-dir /home/paulhax/src/itm/align-browser/experiment-data/base-eval \
  --mlflow-uri file:///home/paulhax/src/itm/align-data/mlruns

# Force re-ingest existing experiments
uv run align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri file:///path/to/mlruns \
  --force
```

#### Flat Ingestion

Single-level runs (one run per experiment directory, no scene-level runs):

```bash
uv run align-track ingest \
  --experiments-dir /path/to/experiments \
  --mlflow-uri file:///path/to/mlruns \
  --flat
```

### View Experiments in MLflow UI

Start the MLflow UI to explore your experiments:

```bash
cd /home/paulhax/src/itm/align-tools
uv run mlflow ui --backend-store-uri file:///home/paulhax/src/itm/align-data/mlruns
```

Then open http://localhost:5000 in your browser.

### MLflow UI Tips

#### View Experiment Aggregates
Filter by `run_type = "parent"` to see only experiment-level runs with aggregate metrics.

#### Compare Scenes Across Experiments
Filter by `scene_id = "June2025-SS-eval"` to see all evaluations of a specific scene across different ADM/LLM/alignment combinations.

#### Drill Down into Experiments
Click on a parent run to see all its child scene runs.

#### Filter by ADM/LLM
Use tags: `adm = "pipeline_baseline"` or `llm = "mistralai/Mistral-7B-Instruct-v0.3"`

#### Compare Multiple Runs
Select multiple runs (use checkboxes) and click "Compare" to see side-by-side metrics and parameters.

### Ingest Experiments into Aim

Aim uses a **flat scene-level structure** where each scene becomes a top-level run. This enables direct scene comparison and better UI performance with large datasets.

```bash
# Ingest all experiments to Aim
uv run align-track ingest \
  --backend aim \
  --experiments-dir /home/paulhax/src/itm/align-browser/experiment-data \
  --aim-repo /home/paulhax/src/itm/align-data/aim-test

# Ingest single experiment directory
uv run align-track ingest \
  --backend aim \
  --experiments-dir /home/paulhax/src/itm/align-browser/experiment-data/baseline-eval \
  --aim-repo /home/paulhax/src/itm/align-data/aim-test

# Force re-ingest existing scenes
uv run align-track ingest \
  --backend aim \
  --experiments-dir /path/to/experiments \
  --aim-repo /path/to/aim/repo \
  --force
```

### Reindex Aim Repository (Important!)

After ingestion completes, you **must** reindex the Aim repository to make runs visible in the UI:

```bash
# Reindex to finalize all runs
uv run aim storage --repo /home/paulhax/src/itm/align-data/aim-test reindex -y
```

**Why?** Aim finalizes runs when the process terminates. Since we create many runs (6k+) in a single process, they need to be indexed manually after ingestion. This takes about 8 minutes for ~6,700 runs.

### View Experiments in Aim UI

Start the Aim UI to explore your experiments:

```bash
uv run aim up --repo /home/paulhax/src/itm/align-data/aim-test
```

Then open http://127.0.0.1:43800 in your browser.

### Aim UI Tips

#### Group Scenes by Experiment
Group by `experiment_path` parameter to see all scenes from one experiment together.

#### Compare ADM/LLM Combinations
Filter by `scenario_id` to see how different ADM/LLM combinations handled the same scene.

#### View Scene Text Inline
Parameters like `probe_text`, `choices_text`, `decision_text`, and `justification` are viewable directly in the UI without downloading artifacts.

#### Filter by Configuration
Use parameters to filter:
- `adm = "pipeline_baseline"`
- `llm_backbone = "mistralai/Mistral-7B-Instruct-v0.3"`
- `alignment_target_id = "MJ5"`
- `kdma.Moral judgement = 0.75`

#### Search and Compare
- Click "Runs" to see all scenes as flat list
- Use search/filter to narrow down
- Select multiple runs and click "Compare" for side-by-side view
- Group/pivot by parameters to analyze trends

## Data Structure

### Parent Run (Experiment-Level)

Each experiment directory creates one parent run:

**Parameters:**
- `adm`: ADM name
- `llm_backbone`: LLM model name
- `alignment_target_id`: Alignment target identifier
- `kdma.*`: Individual KDMA values (e.g., `kdma.merit: 0.5`)
- `num_scenarios`: Total number of scenes in experiment

**Metrics (Aggregates):**
- `timing.total_seconds`: Total runtime across all scenes
- `timing.avg_seconds`: Average time per scene
- `timing.min_seconds`: Fastest scene
- `timing.max_seconds`: Slowest scene

**Tags:**
- `run_type`: "parent"
- `adm`: ADM name (for filtering)
- `llm`: LLM model (for filtering)
- `experiment_path`: Relative path to experiment directory
- `has_hydra_config`: Whether .hydra/config.yaml exists

**Artifacts:**
- `input_output.json`: All scene inputs/outputs
- `timing.json`: All timing data
- `scores.json`: All scores (if available)
- `.hydra/config.yaml`: Full Hydra configuration

### Child Run (Scene-Level)

Each scene in an experiment creates one child run:

**Parameters:**
- All parent parameters, plus:
- `scene_id`: Scene identifier (e.g., "June2025-SS-eval")
- `scene_index`: Index in input_output.json

**Metrics (Scene-Specific):**
- `scene.timing_seconds`: Time for this scene
- `scene.score.*`: Per-scene scores (if available)

**Tags:**
- `scene_id`: Scene identifier (for filtering)
- `scene_index`: Index in experiment
- `mlflow.parentRunId`: Link to parent run
- All parent tags

**Artifacts:**
- `scene_{index}/scene_data.json`: Individual scene input/output

### Aim Run (Scene-Level, Flat Structure)

Each scene creates one top-level Aim run (no parent/child hierarchy):

**Run Name:**
- Format: `{scene_id}_idx{scene_index}` (e.g., "June2025-MF1-eval_idx0")

**Experiment Grouping:**
- `experiment`: Relative path to experiment directory (e.g., "baseline-eval")

**Parameters:**
- `adm`: ADM name
- `llm_backbone`: LLM model name (if applicable)
- `alignment_target_id`: Alignment target identifier
- `kdma.*`: Individual KDMA values (e.g., `kdma.merit: 0.5`)
- `run_variant`: Experiment variant identifier
- `scenario_id`: Scene scenario identifier
- `scene_id`: Scene identifier (same as scenario_id)
- `scene_index`: Index within experiment
- `experiment_path`: Relative path for grouping

**Text Parameters (Inline Viewing):**
- `probe_text`: Scene description/state
- `choices_text`: Available action choices (formatted list)
- `decision_text`: ADM's chosen action
- `justification`: Decision reasoning/explanation

**Metrics:**
- `timing_seconds`: Scene execution time
- `score.*`: Per-scene scores (if available)

**Key Differences from MLflow:**
- **Flat structure**: All scenes are top-level runs (no parent runs)
- **Text inline**: Scene text viewable as parameters, not artifacts
- **Grouping**: Use `experiment_path` parameter to group scenes by experiment
- **No aggregates**: No experiment-level summary metrics (can compute in UI)
- **Performance**: Optimized for 10k+ runs with fast filtering/grouping

## List Experiment Runs (Legacy)

To list all experiment runs in a directory without MLflow:

```bash
uv run list-runs <experiment_directory>

# Example
uv run list-runs /home/paulhax/src/itm/align-browser/experiment-data/test-experiments
```

This displays a table with:
- Run Path: The experiment run identifier
- ADM Name: The ADM configuration used
- Alignment: The alignment configuration
- Scenarios: Number of scenarios in the run
