"""Ingest align-system experiments into MLflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel

from align_utils.discovery import parse_experiments_directory
from align_utils.models import ExperimentData, InputOutputItem

from .mlflow_client import setup_mlflow_tracking, experiment_exists


def get_directory_timestamp(path: Path) -> Optional[str]:
    """Get modification timestamp of experiment directory."""
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).isoformat()
    except OSError:
        return None


class IngestSummary(BaseModel):
    """Summary of ingestion results."""

    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    errors: List[Tuple[str, str]] = []


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary with dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_parameters(experiment: ExperimentData) -> dict:
    """Extract parameters from experiment config.

    Args:
        experiment: ExperimentData object

    Returns:
        Dictionary of parameter key-value pairs for MLflow
    """
    params = {}

    params["adm"] = experiment.config.adm.name
    if experiment.config.adm.llm_backbone:
        params["llm_backbone"] = experiment.config.adm.llm_backbone

    params["alignment_target_id"] = experiment.config.alignment_target.id

    for kdma_value in experiment.config.alignment_target.kdma_values:
        params[f"kdma.{kdma_value.kdma}"] = kdma_value.value

    params["run_variant"] = experiment.config.run_variant

    params["num_scenarios"] = len(experiment.input_output.data)
    if experiment.input_output.data:
        first_scenario = experiment.input_output.data[0].input.scenario_id
        params["first_scenario_id"] = first_scenario

    return params


def extract_metrics(experiment: ExperimentData) -> dict:
    """Extract metrics from experiment scores and timing.

    Args:
        experiment: ExperimentData object

    Returns:
        Dictionary of metric key-value pairs for MLflow
    """
    metrics = {}

    if experiment.scores and experiment.scores.data:
        for score_entry in experiment.scores.data:
            for key, value in score_entry.items():
                if isinstance(value, (int, float)):
                    metrics[f"score.{key}"] = float(value)

    if experiment.timing:
        total_time = sum(t for t in experiment.timing.raw_times_s)
        avg_time = total_time / len(experiment.timing.raw_times_s) if experiment.timing.raw_times_s else 0

        metrics["timing.total_seconds"] = total_time
        metrics["timing.avg_seconds"] = avg_time

        if experiment.timing.raw_times_s:
            metrics["timing.max_seconds"] = max(experiment.timing.raw_times_s)
            metrics["timing.min_seconds"] = min(experiment.timing.raw_times_s)

    return metrics


def extract_tags(experiment: ExperimentData, experiments_root: Path) -> dict:
    """Extract tags for filtering experiments.

    Args:
        experiment: ExperimentData object
        experiments_root: Root path of all experiments (for relative paths)

    Returns:
        Dictionary of tag key-value pairs for MLflow
    """
    tags = {}

    try:
        relative_path = experiment.experiment_path.relative_to(experiments_root)
        tags["experiment_path"] = str(relative_path)
    except ValueError:
        tags["experiment_path"] = experiment.experiment_path.name

    tags["adm"] = experiment.config.adm.name

    if experiment.config.adm.llm_backbone:
        tags["llm"] = experiment.config.adm.llm_backbone

    if experiment.input_output.data:
        first_scenario = experiment.input_output.data[0].input.scenario_id
        tags["scenario"] = first_scenario

    config_path = experiment.experiment_path / ".hydra" / "config.yaml"
    tags["has_hydra_config"] = str(config_path.exists())

    tags["alignment_target_id"] = experiment.config.alignment_target.id

    timestamp = get_directory_timestamp(experiment.experiment_path)
    if timestamp:
        tags["experiment_timestamp"] = timestamp

    return tags


def log_artifacts(experiment: ExperimentData):
    """Log experiment artifacts to MLflow.

    Args:
        experiment: ExperimentData object
    """
    input_output_path = experiment.experiment_path / "input_output.json"
    if input_output_path.exists():
        mlflow.log_artifact(str(input_output_path))

    scores_path = experiment.experiment_path / "scores.json"
    if scores_path.exists():
        mlflow.log_artifact(str(scores_path))

    timing_path = experiment.experiment_path / "timing.json"
    if timing_path.exists():
        mlflow.log_artifact(str(timing_path))

    config_path = experiment.experiment_path / ".hydra" / "config.yaml"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path=".hydra")


def extract_scene_input(item: InputOutputItem) -> dict:
    """Extract input data from a scene item for trace logging."""
    input_data = item.input

    result = {
        "scenario_id": input_data.scenario_id,
        "state": input_data.state or "",
    }

    if input_data.full_state:
        meta = input_data.full_state.get("meta_info", {})
        result["scene_id"] = meta.get("scene_id", "")
        result["unstructured"] = input_data.full_state.get("unstructured", "")

        characters = input_data.full_state.get("characters", [])
        if characters:
            result["characters"] = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "unstructured": c.get("unstructured", ""),
                }
                for c in characters
            ]

    if input_data.choices:
        result["choices"] = [
            {
                "action_id": c.get("action_id", ""),
                "unstructured": c.get("unstructured", ""),
                "kdma_association": c.get("kdma_association", {}),
            }
            for c in input_data.choices
        ]

    return result


def extract_scene_output(item: InputOutputItem) -> dict:
    """Extract output data from a scene item for trace logging."""
    if not item.output:
        return {}

    result = {
        "choice_index": item.output.choice,
    }

    if item.output.action:
        result["action"] = {
            "action_id": item.output.action.action_id,
            "action_type": item.output.action.action_type,
            "unstructured": item.output.action.unstructured,
            "justification": item.output.action.justification or "",
        }

    return result


def create_scene_trace(
    client: MlflowClient,
    experiment: ExperimentData,
    scene_index: int,
    run_id: str,
) -> Optional[str]:
    """Create a trace for a single scene within an experiment run.

    Args:
        client: MlflowClient instance
        experiment: ExperimentData object
        scene_index: Index of scene in input_output.data
        run_id: The parent run ID to associate the trace with

    Returns:
        Trace ID if successful, None otherwise
    """
    scene_item = experiment.input_output.data[scene_index]

    scene_id = "unknown"
    if scene_item.input.full_state:
        meta = scene_item.input.full_state.get("meta_info", {})
        scene_id = meta.get("scene_id", f"scene_{scene_index}")

    trace_name = f"{scene_id}"

    input_data = extract_scene_input(scene_item)
    output_data = extract_scene_output(scene_item)

    attributes = {
        "scene_index": scene_index,
        "scenario_id": scene_item.input.scenario_id,
        "adm": experiment.config.adm.name,
        "alignment_target_id": experiment.config.alignment_target.id,
    }

    if experiment.timing and scene_index < len(experiment.timing.raw_times_s):
        attributes["timing_seconds"] = experiment.timing.raw_times_s[scene_index]

    if scene_item.choice_info:
        if scene_item.choice_info.true_kdma_values:
            attributes["true_kdma_values"] = json.dumps(
                scene_item.choice_info.true_kdma_values
            )
        if scene_item.choice_info.predicted_kdma_values:
            attributes["predicted_kdma_values"] = json.dumps(
                scene_item.choice_info.predicted_kdma_values.model_dump()
                if hasattr(scene_item.choice_info.predicted_kdma_values, 'model_dump')
                else scene_item.choice_info.predicted_kdma_values
            )

    if scene_item.output and scene_item.output.choice is not None:
        attributes["choice_index"] = scene_item.output.choice
        if scene_item.input.choices and scene_item.output.choice < len(scene_item.input.choices):
            chosen = scene_item.input.choices[scene_item.output.choice]
            attributes["chosen_action"] = chosen.get("unstructured", "")

    tags = {
        "mlflow.runId": run_id,
        "scene_id": scene_id,
    }

    try:
        root_span = client.start_trace(
            name=trace_name,
            inputs=input_data,
            attributes=attributes,
            tags=tags,
        )

        root_span.set_outputs(output_data)

        client.end_trace(
            trace_id=root_span.trace_id,
        )

        return root_span.trace_id
    except Exception as e:
        print(f"Warning: Failed to create trace for scene {scene_index}: {e}")
        return None


def ingest_experiment(
    experiment: ExperimentData,
    experiments_root: Path,
    experiment_id: str,
    force: bool = False,
) -> Tuple[bool, int]:
    """Ingest a single experiment into MLflow with traces for scenes.

    Args:
        experiment: ExperimentData object
        experiments_root: Root path of all experiments
        experiment_id: MLflow experiment ID
        force: Re-ingest even if already exists

    Returns:
        Tuple of (ingested: bool, num_traces: int)
        - ingested: True if run was created, False if skipped (duplicate)
        - num_traces: Number of traces created

    Raises:
        Exception if ingestion fails
    """
    try:
        relative_path = experiment.experiment_path.relative_to(experiments_root)
        experiment_path_str = str(relative_path)
    except ValueError:
        experiment_path_str = experiment.experiment_path.name

    if not force and experiment_exists(experiment_id, experiment_path_str):
        return False, 0

    run_name = experiment.experiment_path.name
    num_traces = 0

    with mlflow.start_run(run_name=run_name) as run:
        params = extract_parameters(experiment)
        for key, value in params.items():
            mlflow.log_param(key, value)

        metrics = extract_metrics(experiment)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        tags = extract_tags(experiment, experiments_root)
        for key, value in tags.items():
            mlflow.set_tag(key, value)

        log_artifacts(experiment)

        client = MlflowClient()
        for scene_index in range(len(experiment.input_output.data)):
            trace_id = create_scene_trace(
                client, experiment, scene_index, run.info.run_id
            )
            if trace_id:
                num_traces += 1

    return True, num_traces


def ingest_experiments_directory(
    experiments_dir: Path,
    mlflow_tracking_uri: str,
    experiment_name: str = "align-system-experiments",
    force: bool = False,
    verbose: bool = True,
) -> IngestSummary:
    """Ingest experiments from a directory into MLflow.

    Creates one run per experiment directory with traces for each scene.
    Traces capture input/output text, decisions, and KDMA values.

    Args:
        experiments_dir: Root directory containing experiments
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: Name of MLflow experiment
        force: Re-ingest existing experiments
        verbose: Print progress messages

    Returns:
        IngestSummary with results
    """
    if verbose:
        print(f"Discovering experiments in: {experiments_dir}")

    experiments = parse_experiments_directory(experiments_dir)
    summary = IngestSummary(total=len(experiments))

    if verbose:
        print(f"Found {len(experiments)} experiments\n")
        print("Ingesting experiments to MLflow (1 run + traces per scene)...")

    experiment_id = setup_mlflow_tracking(mlflow_tracking_uri, experiment_name)

    total_traces = 0
    for i, experiment in enumerate(experiments, 1):
        try:
            ingested, num_traces = ingest_experiment(
                experiment,
                experiments_dir,
                experiment_id,
                force=force,
            )

            if ingested:
                summary.success += 1
                total_traces += num_traces
                status = f"✓ ({num_traces} traces)"
            else:
                summary.skipped += 1
                status = "⊘ (duplicate)"

            if verbose:
                experiment_name_str = experiment.experiment_path.name
                print(f"[{i}/{len(experiments)}] {experiment_name_str} {status}")

        except Exception as e:
            summary.failed += 1
            error_msg = str(e)
            summary.errors.append((str(experiment.experiment_path), error_msg))

            if verbose:
                experiment_name_str = experiment.experiment_path.name
                print(f"[{i}/{len(experiments)}] {experiment_name_str} ✗ ({error_msg})")

    if verbose:
        print("\nSummary:")
        print("━" * 40)
        print(f"Successfully ingested:  {summary.success}")
        print(f"Total traces created:   {total_traces}")
        print(f"Skipped (duplicate):    {summary.skipped}")
        print(f"Failed:                 {summary.failed}")
        print("━" * 40)
        print(f"\nMLflow tracking URI: {mlflow_tracking_uri}")
        print(f"Start UI with: mlflow ui --backend-store-uri {mlflow_tracking_uri}")

    return summary
