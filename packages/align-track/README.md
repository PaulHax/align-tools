# align-track

Organizing utilities for align-system experiments.

## Overview

The `align-track` package provides tools for organzing experimental data from the align-system.

## Installation

```bash
pip install align-track
```

## Development

This package is part of the align-tools monorepo and depends on `align-utils`.

For local development:
```bash
uv pip install -e .
```

## Features

- Experiment tracking
- Data aggregation and analysis
- Integration with align-utils for data parsing

## Usage

### List Experiment Runs

To list all experiment runs in a directory:

```bash
uv run python -m align_track.list_runs <experiment_directory>

# Example
uv run python -m align_track.list_runs ../align-utils/experiment-data/test-experiments
```

This will display a table with:
- Run Path: The experiment run identifier
- ADM Name: The ADM configuration used
- Alignment: The alignment configuration
- Scenarios: Number of scenarios in the run
