"""Parser for experiment directory structures using Pydantic models."""

import re
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union
from collections import defaultdict

from align_utils.models import (
    ExperimentData,
    Manifest,
    InputOutputFile,
    calculate_file_checksums,
)


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    with file_path.open("r") as f:
        return yaml.safe_load(f)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    with file_path.open("r") as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    with file_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    with file_path.open("w") as f:
        json.dump(data, f, indent=2)


def discover_input_output_files(root: Path, recursive: bool = True) -> List[Path]:
    """Find all input_output.json files in a directory tree.

    Returns paths to files, not loaded data.

    Args:
        root: Root directory to search
        recursive: Whether to search recursively (default: True)

    Returns:
        List of paths to input_output.json files
    """
    if root.is_file() and root.name == "input_output.json":
        return [root]

    if not root.is_dir():
        return []

    if recursive:
        return list(root.rglob("input_output.json"))
    else:
        input_output_file = root / "input_output.json"
        if input_output_file.exists():
            return [input_output_file]
        return []


def load_input_output_files(
    root: Path, recursive: bool = True
) -> List[InputOutputFile]:
    """Find and load all JSON files that validate as InputOutputFile structures.

    Recursively searches for JSON files and validates them against the
    InputOutputFile Pydantic model. Filters out files like timing.json,
    scores.json, and other JSON files that don't match the expected structure.

    Args:
        root: Root directory or file to search
        recursive: Whether to search recursively (default: True)

    Returns:
        List of successfully loaded InputOutputFile models
    """
    if not root.exists():
        return []

    if root.is_file():
        try:
            return [InputOutputFile.from_file(root)]
        except Exception:
            return []

    pattern = "**/*.json" if recursive else "*.json"
    json_files = root.glob(pattern)

    loaded_files = []
    for json_file in json_files:
        try:
            loaded_files.append(InputOutputFile.from_file(json_file))
        except Exception:
            continue

    return loaded_files


def _extract_run_variant(
    experiment_dir: Path, experiments_root: Path, all_conflicting_dirs: List[Path]
) -> str:
    """
    Extract run variant from directory structure for distinguishing conflicting experiments.

    Args:
        experiment_dir: Path to the specific experiment directory
        experiments_root: Root path of all experiments
        all_conflicting_dirs: List of all directories that have conflicts (same ADM+LLM+KDMA)

    Returns:
        String representing the run variant, or "default" for default variant
    """
    # Get the relative path from experiments_root
    relative_path = experiment_dir.relative_to(experiments_root)
    path_parts = relative_path.parts

    # Skip KDMA configuration directories (contain dashes with numbers)
    # Examples: merit-0.4, affiliation-0.0, personal_safety-0.5
    def is_kdma_dir(dirname):
        return bool(re.match(r"^[a-z_]+-(0\.\d+|1\.0|0)$", dirname))

    # Find the ADM-level directory (first non-KDMA directory)
    adm_dir = None
    for part in path_parts:
        if not is_kdma_dir(part):
            adm_dir = part
            break

    if not adm_dir:
        return "default"

    # Extract ADM directories from all conflicting paths
    conflicting_adm_dirs = set()
    for conflict_dir in all_conflicting_dirs:
        try:
            conflict_relative = conflict_dir.relative_to(experiments_root)
            conflict_parts = conflict_relative.parts
            for part in conflict_parts:
                if not is_kdma_dir(part):
                    conflicting_adm_dirs.add(part)
                    break
        except (ValueError, AttributeError):
            continue

    # If there's only one unique ADM directory, no variant needed
    if len(conflicting_adm_dirs) <= 1:
        return "default"

    # Find the common prefix among all conflicting ADM directories
    adm_dir_list = sorted(conflicting_adm_dirs)
    common_prefix = ""

    if len(adm_dir_list) >= 2:
        # Find longest common prefix
        first_dir = adm_dir_list[0]
        for i, char in enumerate(first_dir):
            if all(i < len(d) and d[i] == char for d in adm_dir_list):
                common_prefix += char
            else:
                break

        # Remove trailing underscores
        common_prefix = common_prefix.rstrip("_")

    # Extract variant as the unique suffix after common prefix
    if common_prefix and adm_dir.startswith(common_prefix):
        variant = adm_dir[len(common_prefix) :].lstrip("_")
        # Use lexicographically first directory as "default"
        if adm_dir == min(adm_dir_list):
            return "default"
        return variant if variant else "default"

    # Fallback: use the full ADM directory name if no common prefix found
    # Choose the lexicographically first one as default
    if adm_dir == min(conflicting_adm_dirs):
        return "default"
    return adm_dir


def _create_experiments_from_directory(experiment_dir: Path) -> List[ExperimentData]:
    """Create experiments from a directory, handling both uniform and mixed KDMA alignment.

    This unified function handles both cases:
    - Uniform KDMA: All scenes use same alignment target (defined in config.yaml)
    - Mixed KDMA: Different scenes have different alignment targets (defined per input item)

    Returns a list of experiments (one per unique alignment target).
    """
    experiments = []

    input_output = InputOutputFile.from_file(experiment_dir / "input_output.json")

    # Load config to check for uniform alignment target
    config_path = experiment_dir / ".hydra" / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    has_config_alignment = "alignment_target" in config_data

    # Group by alignment_target_id
    grouped_data = defaultdict(list)

    for item in input_output.data:
        # Determine alignment target for this item
        if has_config_alignment:
            # Uniform KDMA: Use alignment target from config for all items
            alignment_target_id = config_data["alignment_target"]["id"]
        else:
            # Mixed KDMA: Use alignment target from input item
            alignment_target_id = item.input.alignment_target_id
            if alignment_target_id is None:
                alignment_target_id = "unaligned"  # Handle null alignment targets

        grouped_data[alignment_target_id].append(item)

    # Create experiments for each alignment target group
    for alignment_target_id, items in grouped_data.items():
        try:
            if has_config_alignment:
                # Uniform KDMA: Use standard from_directory method
                experiment = ExperimentData.from_directory(experiment_dir)
                experiments.append(experiment)
                break  # Only one experiment for uniform KDMA
            else:
                # Mixed KDMA: Create experiment for this specific alignment target
                experiment = ExperimentData.from_directory_mixed_kdma(
                    experiment_dir,
                    alignment_target_id,
                    items,
                )
                experiments.append(experiment)

        except Exception as e:
            print(
                f"Error processing alignment_target_id {alignment_target_id} in {experiment_dir}: {e}"
            )
            continue

    return experiments


def _create_experiments_from_directory_no_hydra(
    experiment_dir: Path,
) -> List[ExperimentData]:
    """Create experiments from a directory without .hydra/config.yaml.

    For directories that only have input_output.json and timing.json.
    """
    try:
        experiment = ExperimentData.from_directory_no_hydra(experiment_dir)
        return [experiment]
    except Exception as e:
        print(f"Error processing {experiment_dir}: {e}")
        return []


def parse_experiments_directory(experiments_root: Path) -> List[ExperimentData]:
    """
    Parse the experiments directory structure and return a list of ExperimentData.

    First checks if the given path itself is an experiment directory, then
    recursively searches through the directory structure to find all directories
    that contain the required experiment files (input_output.json, timing.json,
    and optionally .hydra/config.yaml). scores.json is optional.

    Args:
        experiments_root: Path to the root experiments directory or a direct experiment directory

    Returns:
        List of successfully parsed ExperimentData objects
    """
    experiments = []

    directories_found = 0
    directories_with_files = 0
    directories_processed = 0

    # First check if the root path itself is an experiment directory
    if ExperimentData.has_required_files(experiments_root):
        directories_found += 1
        directories_with_files += 1

        try:
            directory_experiments = _create_experiments_from_directory(experiments_root)
            experiments.extend(directory_experiments)
            directories_processed += 1
        except Exception as e:
            print(f"Error processing {experiments_root}: {e}")
    elif ExperimentData.has_required_files_no_hydra(experiments_root):
        # Check for experiment without hydra config
        directories_found += 1
        directories_with_files += 1

        directory_experiments = _create_experiments_from_directory_no_hydra(
            experiments_root
        )
        experiments.extend(directory_experiments)
        if directory_experiments:
            directories_processed += 1

    # Recursively find all directories that have required experiment files
    for experiment_dir in experiments_root.rglob("*"):
        if not experiment_dir.is_dir():
            continue

        directories_found += 1

        # Skip directories containing "OUTDATED" in their path
        if "OUTDATED" in str(experiment_dir).upper():
            continue

        # Check if directory has all required files (with hydra)
        if ExperimentData.has_required_files(experiment_dir):
            directories_with_files += 1

            try:
                directory_experiments = _create_experiments_from_directory(
                    experiment_dir
                )
                experiments.extend(directory_experiments)
                directories_processed += 1

            except Exception as e:
                print(f"Error processing {experiment_dir}: {e}")
                continue
        # Check if directory has required files without hydra
        elif ExperimentData.has_required_files_no_hydra(experiment_dir):
            directories_with_files += 1

            directory_experiments = _create_experiments_from_directory_no_hydra(
                experiment_dir
            )
            experiments.extend(directory_experiments)
            if directory_experiments:
                directories_processed += 1

    return experiments


def build_manifest_from_experiments(
    experiments: List[ExperimentData], experiments_root: Path
) -> Manifest:
    """
    Build the enhanced global manifest from a list of parsed experiments.

    Uses the new flexible parameter-based structure with integrity validation
    and fast lookup indices.

    Args:
        experiments: List of ExperimentData objects
        experiments_root: Path to experiments root (for calculating relative paths)

    Returns:
        Manifest object with new structure
    """
    from datetime import datetime, timezone

    # Initialize manifest
    manifest = Manifest(
        manifest_version="2.0", generated_at=datetime.now(timezone.utc).isoformat()
    )

    # Collect all input_output files for checksum calculation
    input_output_files = set()
    for experiment in experiments:
        # Add default input_output.json path
        input_output_files.add(experiment.experiment_path / "input_output.json")

    # Calculate checksums for all files
    source_file_checksums = calculate_file_checksums(list(input_output_files))

    # Process experiments with conflict detection similar to original
    # First pass: detect conflicts by grouping experiments by their base parameters
    base_key_groups: Dict[str, List[ExperimentData]] = {}

    for experiment in experiments:
        # Group experiments by their full key (including default run_variant)
        base_key = experiment.config.generate_key()

        if base_key not in base_key_groups:
            base_key_groups[base_key] = []
        base_key_groups[base_key].append(experiment)

    # Second pass: add run_variant for conflicts and process all experiments
    enhanced_experiments = []

    for base_key, group_experiments in base_key_groups.items():
        if len(group_experiments) == 1:
            # No conflict, use original experiment
            enhanced_experiments.append(group_experiments[0])
        else:
            # Conflict detected - add run_variant from directory structure
            conflicting_dirs = [exp.experiment_path for exp in group_experiments]
            for experiment in group_experiments:
                run_variant = _extract_run_variant(
                    experiment.experiment_path, experiments_root, conflicting_dirs
                )
                # Always create experiment with run_variant (will be "default" if not extracted)
                enhanced_config = experiment.config.model_copy(deep=True)
                enhanced_config.run_variant = run_variant

                enhanced_experiment = ExperimentData(
                    config=enhanced_config,
                    input_output=experiment.input_output,
                    scores=experiment.scores,
                    timing=experiment.timing,
                    experiment_path=experiment.experiment_path,
                )

                enhanced_experiments.append(enhanced_experiment)

    # Add experiments to enhanced manifest
    for experiment in enhanced_experiments:
        try:
            manifest.add_experiment(experiment, experiments_root, source_file_checksums)
        except Exception as e:
            print(f"Error adding experiment {experiment.experiment_path}: {e}")
            continue

    # Add metadata
    manifest.metadata = {
        "total_experiments": len(manifest.experiments),
        "total_scenarios": len(manifest.indices.by_scenario),
        "total_files": len(manifest.files),
        "adm_types": list(manifest.indices.by_adm.keys()),
        "llm_backbones": list(manifest.indices.by_llm.keys()),
        "kdma_combinations": list(manifest.indices.by_kdma.keys()),
    }

    return manifest


def copy_experiment_files(
    experiments: List[ExperimentData], experiments_root: Path, data_output_dir: Path
):
    """
    Copy experiment files to the output data directory.

    Args:
        experiments: List of ExperimentData objects
        experiments_root: Path to experiments root
        data_output_dir: Path to output data directory
    """
    import shutil

    for experiment in experiments:
        # Determine relative path for copying
        relative_experiment_path = experiment.experiment_path.relative_to(
            experiments_root
        )
        target_experiment_dir = data_output_dir / relative_experiment_path
        target_experiment_dir.mkdir(parents=True, exist_ok=True)

        # Copy relevant files
        shutil.copy(
            experiment.experiment_path / "input_output.json",
            target_experiment_dir / "input_output.json",
        )

        # Copy scores.json if it exists
        scores_path = experiment.experiment_path / "scores.json"
        if scores_path.exists():
            shutil.copy(scores_path, target_experiment_dir / "scores.json")

        shutil.copy(
            experiment.experiment_path / "timing.json",
            target_experiment_dir / "timing.json",
        )
