"""Pydantic models for experiment data structures and output formatting."""

import json
import yaml
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


KDMAScoreValue = Union[float, List[float]]
KDMAChoiceScores = Dict[str, KDMAScoreValue]
KDMANestedScores = Dict[str, KDMAChoiceScores]


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    if not file_path.exists():
        return ""

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return f"sha256:{sha256_hash.hexdigest()}"


def calculate_file_checksums(file_paths: List[Path]) -> Dict[str, str]:
    """Calculate checksums for multiple files."""
    checksums = {}
    for file_path in file_paths:
        checksums[str(file_path)] = calculate_file_checksum(file_path)
    return checksums


class KDMAValue(BaseModel):
    """Represents a KDMA (Key Decision Making Attributes) value."""

    kdma: str
    value: float
    kdes: Optional[Any] = None


def parse_alignment_target_id(alignment_target_id: str) -> List[KDMAValue]:
    """
    Parse alignment_target_id string to extract KDMA values.

    Supports both single and multi-KDMA formats:
    - Single: "ADEPT-June2025-merit-0.0" -> [KDMAValue(kdma="merit", value=0.0)]
    - Single with underscore: "ADEPT-June2025-personal_safety-0.0" -> [KDMAValue(kdma="personal_safety", value=0.0)]
    - Short format: "personal_safety-0.0" -> [KDMAValue(kdma="personal_safety", value=0.0)]
    - Multi: "ADEPT-June2025-affiliation_merit-0.0_0.0" ->
             [KDMAValue(kdma="affiliation", value=0.0), KDMAValue(kdma="merit", value=0.0)]
    - Unaligned: "unaligned" -> [] (no KDMAs)

    Args:
        alignment_target_id: String like "ADEPT-June2025-merit-0.0",
                           "ADEPT-June2025-personal_safety-0.0", "personal_safety-0.0",
                           "ADEPT-June2025-affiliation_merit-0.0_0.0", or "unaligned"

    Returns:
        List of KDMAValue objects
    """
    if not alignment_target_id or alignment_target_id == "unaligned":
        return []

    # Split by hyphens
    parts = alignment_target_id.split("-")
    if len(parts) < 2:
        return []

    # Extract KDMA names and values from the last two parts
    kdma_part = parts[-2]  # e.g., "affiliation_merit", "merit", or "personal_safety"
    value_part = parts[-1]  # e.g., "0.0_0.0" or "0.0"

    # Split values by underscore and convert to float
    try:
        value_strings = value_part.split("_")
        values = [float(v) for v in value_strings]
    except ValueError:
        return []

    # Determine how to split KDMA names based on number of values
    if len(values) == 1:
        # Single value: treat entire kdma_part as one KDMA name (handles personal_safety)
        kdma_names = [kdma_part]
    else:
        # Multiple values: split KDMA names by underscore
        kdma_names = kdma_part.split("_")

        # Ensure we have the same number of KDMAs and values
        if len(kdma_names) != len(values):
            return []

    # Create KDMAValue objects
    kdma_values = []
    for kdma_name, value in zip(kdma_names, values):
        kdma_values.append(KDMAValue(kdma=kdma_name, value=value))

    return kdma_values


class AlignmentTarget(BaseModel):
    """Represents an alignment target configuration."""

    id: str = "unknown_target"
    kdma_values: List[KDMAValue] = Field(default_factory=list)


class ADMConfig(BaseModel):
    """Represents ADM (Automated Decision Maker) configuration."""

    name: str = "unknown_adm"
    instance: Optional[Dict[str, Any]] = None
    structured_inference_engine: Optional[Dict[str, Any]] = None

    @property
    def llm_backbone(self) -> str | None:
        """Extract LLM backbone model name, or None if no LLM is configured."""
        if self.structured_inference_engine:
            return self.structured_inference_engine.get("model_name")
        return None


class ExperimentConfig(BaseModel):
    """Represents the complete experiment configuration from config.yaml."""

    name: str = "unknown"
    adm: ADMConfig = Field(default_factory=ADMConfig)
    alignment_target: AlignmentTarget = Field(default_factory=AlignmentTarget)
    run_variant: str = "default"

    def generate_key(self) -> str:
        """Generate a unique key for this experiment configuration."""
        kdma_parts = [
            f"{kv.kdma}-{kv.value}" for kv in self.alignment_target.kdma_values
        ]
        kdma_string = "_".join(sorted(kdma_parts))
        return (
            f"{self.adm.name}:{self.adm.llm_backbone}:{kdma_string}:{self.run_variant}"
        )

    def generate_experiment_key(self, experiment_path: Optional[Path] = None) -> str:
        """Generate hash-based experiment key for new manifest structure."""
        key_data = {
            "adm": self.adm.name,
            "llm": self.adm.llm_backbone,
            "kdma": self._get_kdma_key(),
            "run_variant": self.run_variant,
        }

        # Add experiment path to ensure uniqueness across different directories
        if experiment_path:
            key_data["path"] = str(experiment_path)

        # Create deterministic hash from sorted key data
        key_string = json.dumps(key_data, sort_keys=True)
        hash_obj = hashlib.sha256(key_string.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        return f"exp_{hash_hex[:8]}"

    def _get_kdma_key(self) -> str:
        """Generate KDMA key component for experiment identification."""
        if not self.alignment_target.kdma_values:
            return "unaligned"

        kdma_parts = [
            f"{kv.kdma}-{kv.value}" for kv in self.alignment_target.kdma_values
        ]
        return "_".join(sorted(kdma_parts))


class InputData(BaseModel):
    """Represents input data for an experiment."""

    scenario_id: str = "unknown_scenario"
    alignment_target_id: Optional[str] = None
    full_state: Optional[Dict[str, Any]] = None
    state: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None


class ChoiceInfo(BaseModel):
    """ADM execution metadata from align-system choice_info dict.

    All fields optional to match align-system's flexible output structure.
    Allows extra fields via ConfigDict to accommodate additional metadata.
    """

    model_config = ConfigDict(extra="allow")

    predicted_kdma_values: Optional[KDMANestedScores] = None
    true_kdma_values: Optional[Dict[str, Dict[str, float]]] = None
    true_relevance: Optional[Dict[str, float]] = None
    icl_example_responses: Optional[Dict[str, Any]] = None


class Action(BaseModel):
    """Action chosen by the ADM."""

    model_config = ConfigDict(extra="allow")

    action_id: str
    action_type: str
    unstructured: str
    justification: Optional[str] = None
    character_id: Optional[str] = None
    intent_action: Optional[bool] = None
    kdma_association: Optional[Dict[str, float]] = None


class Output(BaseModel):
    """Output from ADM execution."""

    choice: int
    action: Action


class InputOutputItem(BaseModel):
    """Represents a single input/output item from the experiment."""

    input: InputData
    output: Optional[Output] = None
    choice_info: Optional[ChoiceInfo] = None
    label: Optional[List[Dict[str, float]]] = None


class Decision(BaseModel):
    """Decision result from align-system ADM execution."""

    unstructured: str
    justification: str


class ADMResult(BaseModel):
    """Complete result from align-system ADM execution."""

    decision: Decision
    choice_info: ChoiceInfo


class ScenarioTiming(BaseModel):
    """Represents timing data for a scenario."""

    n_actions_taken: int
    total_time_s: float
    avg_time_s: float
    max_time_s: float
    raw_times_s: List[float]


class TimingData(BaseModel):
    """Represents timing data from timing.json."""

    scenarios: List[ScenarioTiming]
    raw_times_s: List[float]  # Indicies map to list in input_output.json


class InputOutputFile(BaseModel):
    """Wrapper for input_output.json which contains an array of items."""

    data: List[InputOutputItem]

    @classmethod
    def from_file(cls, path: Path) -> "InputOutputFile":
        """Load input_output.json file."""
        with open(path) as f:
            raw_data = json.load(f)

        items = [InputOutputItem(**item_data) for item_data in raw_data]

        return cls(data=items)

    @classmethod
    def load(cls, path: Path) -> "InputOutputFile":
        """Load input_output data from a file or directory.

        - If path is a JSON file: load directly
        - If path is a directory: look for input_output.json inside

        Args:
            path: Path to a JSON file or directory containing input_output.json

        Returns:
            InputOutputFile with data loaded
        """
        if path.is_file():
            return cls.from_file(path)
        elif path.is_dir():
            input_output_file = path / "input_output.json"
            if not input_output_file.exists():
                raise FileNotFoundError(
                    f"No input_output.json found in directory {path}"
                )
            return cls.from_file(input_output_file)
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")

    @property
    def first_scenario_id(self) -> str:
        """Get the scenario ID from the first item."""
        if self.data:
            return self.data[0].input.scenario_id
        return "unknown_scenario"


class ScoresFile(BaseModel):
    """Wrapper for scores.json which contains an array of scoring data."""

    data: List[Dict[str, Any]]

    @classmethod
    def from_file(cls, path: Path) -> "ScoresFile":
        """Load scores.json file."""
        with open(path) as f:
            raw_data = json.load(f)
        return cls(data=raw_data)


class ExperimentData(BaseModel):
    """Complete experiment data loaded from a directory."""

    config: ExperimentConfig
    input_output: InputOutputFile
    scores: Optional[ScoresFile] = None
    timing: Optional[TimingData] = None
    experiment_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow Path type

    @classmethod
    def from_directory(cls, experiment_dir: Path) -> "ExperimentData":
        """Load all experiment data from a directory."""
        # Load config
        config_path = experiment_dir / ".hydra" / "config.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        config = ExperimentConfig(**config_data)

        # Load other files
        input_output = InputOutputFile.from_file(experiment_dir / "input_output.json")

        # Load scores if available
        scores = None
        scores_path = experiment_dir / "scores.json"
        if scores_path.exists():
            scores = ScoresFile.from_file(scores_path)

        timing = None
        timing_path = experiment_dir / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_data = json.load(f)
            timing = TimingData(**timing_data)

        return cls(
            config=config,
            input_output=input_output,
            scores=scores,
            timing=timing,
            experiment_path=experiment_dir,
        )

    @classmethod
    def from_directory_mixed_kdma(
        cls,
        experiment_dir: Path,
        alignment_target_id: str,
        filtered_data: List[InputOutputItem],
    ) -> "ExperimentData":
        """Load experiment data from mixed KDMA directory for a specific alignment target.

        Mixed KDMA format: Handles experiments where different scenes have different KDMA
        configurations, with KDMAs defined per scene in alignment_target_id rather than config.yaml.

        This method works with logical filtering - the original files remain intact.
        """
        # Load config
        config_path = experiment_dir / ".hydra" / "config.yaml"
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Create alignment_target from alignment_target_id
        kdma_values = parse_alignment_target_id(alignment_target_id)
        alignment_target = AlignmentTarget(
            id=alignment_target_id, kdma_values=kdma_values
        )

        # Add alignment_target to config
        config_data["alignment_target"] = alignment_target.model_dump()
        config = ExperimentConfig(**config_data)

        # Create input_output from the logically filtered data (already InputOutputItems)
        input_output = InputOutputFile(data=filtered_data)

        # Load scores if available
        scores = None
        scores_path = experiment_dir / "scores.json"
        if scores_path.exists():
            scores = ScoresFile.from_file(scores_path)

        # Load timing data if available
        timing = None
        timing_path = experiment_dir / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_data = json.load(f)
            timing = TimingData(**timing_data)

        # Create experiment instance
        experiment = cls(
            config=config,
            input_output=input_output,
            scores=scores,
            timing=timing,
            experiment_path=experiment_dir,
        )

        return experiment

    @classmethod
    def from_directory_no_hydra(cls, experiment_dir: Path) -> "ExperimentData":
        """Load experiment data from a directory without .hydra/config.yaml.

        For directories that only have input_output.json and timing.json,
        extract metadata from the directory structure and file contents.
        """
        # Extract ADM name and alignment info from directory structure
        # e.g., combined_rerun/pipeline_baseline/affiliation-0.5
        parts = experiment_dir.parts

        # Try to extract ADM name from parent directory
        adm_name = "unknown"
        alignment_id = "unaligned"

        if len(parts) >= 2:
            # Look for pipeline_* pattern
            for i, part in enumerate(parts):
                if part.startswith("pipeline_"):
                    adm_name = part
                if i < len(parts) - 1:
                    # Check next part for KDMA pattern (e.g., affiliation-0.5)
                    next_part = parts[i + 1]
                    import re

                    if re.match(r"^[a-z_]+-(0\.\d+|1\.0|0)$", next_part):
                        alignment_id = next_part
                        break

        # Create minimal config
        kdma_values = parse_alignment_target_id(alignment_id)
        alignment_target = AlignmentTarget(id=alignment_id, kdma_values=kdma_values)

        config = ExperimentConfig(
            name=adm_name,
            adm=ADMConfig(name=adm_name),
            alignment_target=alignment_target,
        )

        input_output = InputOutputFile.from_file(experiment_dir / "input_output.json")

        scores = None
        scores_path = experiment_dir / "scores.json"
        if scores_path.exists():
            scores = ScoresFile.from_file(scores_path)

        timing = None
        timing_path = experiment_dir / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_data = json.load(f)
            timing = TimingData(**timing_data)

        return cls(
            config=config,
            input_output=input_output,
            scores=scores,
            timing=timing,
            experiment_path=experiment_dir,
        )

    @property
    def key(self) -> str:
        """Get the unique key for this experiment."""
        return self.config.generate_key()

    @property
    def scenario_id(self) -> str:
        """Get the scenario ID for this experiment."""
        return self.input_output.first_scenario_id

    @classmethod
    def has_required_files(cls, experiment_dir: Path) -> bool:
        """Check if directory has all required experiment files."""
        required_files = [
            "input_output.json",
            ".hydra/config.yaml",
        ]
        return all((experiment_dir / f).exists() for f in required_files)

    @classmethod
    def has_required_files_no_hydra(cls, experiment_dir: Path) -> bool:
        """Check if directory has required files without hydra config."""
        required_files = [
            "input_output.json",
        ]
        return all((experiment_dir / f).exists() for f in required_files)


class ExperimentItem(BaseModel):
    """Single item from an experiment with all associated context.

    Composes the raw InputOutputItem with timing and config data,
    providing a self-contained view of each decision in an experiment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: InputOutputItem
    timing_s: float = 0.0
    config: ExperimentConfig
    experiment_path: Path


def get_experiment_items(experiment: ExperimentData) -> List[ExperimentItem]:
    """Extract individual items from an ExperimentData with associated context.

    Each item gets the timing value from the corresponding index in raw_times_s
    and shares the experiment's config.
    """
    raw_times = experiment.timing.raw_times_s if experiment.timing else None
    return [
        ExperimentItem(
            item=item,
            timing_s=raw_times[i] if raw_times else 0.0,
            config=experiment.config,
            experiment_path=experiment.experiment_path,
        )
        for i, item in enumerate(experiment.input_output.data)
    ]


# Enhanced Manifest Models for New Structure
class SceneInfo(BaseModel):
    """Information about a scene within a scenario."""

    source_index: int  # Index in the source input_output.json file
    scene_id: str  # Scene ID from meta_info.scene_id
    timing_s: float | None  # Timing from timing.json, None if not available


class InputOutputFileInfo(BaseModel):
    """File information for input_output data."""

    file: str  # Path to the file
    checksum: str  # SHA256 checksum for integrity
    alignment_target_filter: Optional[str] = None  # Filter for multi-experiment files


class Scenario(BaseModel):
    """Enhanced scenario structure with scene mapping."""

    input_output: InputOutputFileInfo
    scores: Optional[str] = None  # Path to scores.json
    timing: str  # Path to timing.json
    scenes: Dict[str, SceneInfo] = Field(default_factory=dict)  # scene_id -> SceneInfo


class Experiment(BaseModel):
    """Enhanced experiment structure with flexible parameters."""

    parameters: Dict[str, Any]  # Flexible parameter structure
    scenarios: Dict[str, Scenario] = Field(
        default_factory=dict
    )  # scenario_id -> scenario


class FileInfo(BaseModel):
    """Metadata about a source file."""

    checksum: str  # SHA256 checksum
    size: int  # File size in bytes
    experiments: List[str] = Field(
        default_factory=list
    )  # Experiment keys using this file


class ManifestIndices(BaseModel):
    """Indices for fast experiment lookups."""

    by_adm: Dict[str, List[str]] = Field(default_factory=dict)
    by_llm: Dict[str, List[str]] = Field(default_factory=dict)
    by_kdma: Dict[str, List[str]] = Field(default_factory=dict)
    by_scenario: Dict[str, List[str]] = Field(default_factory=dict)


class Manifest(BaseModel):
    """Global manifest with hierarchical structure and integrity validation."""

    manifest_version: str = "1.0"
    generated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    experiments: Dict[str, Experiment] = Field(default_factory=dict)
    indices: ManifestIndices = Field(default_factory=ManifestIndices)
    files: Dict[str, FileInfo] = Field(default_factory=dict)

    def add_experiment(
        self,
        experiment: "ExperimentData",
        experiments_root: Path,
        source_file_checksums: Dict[str, str],
    ):
        """Add an experiment to the enhanced manifest."""
        # Generate experiment key with path for uniqueness
        exp_key = experiment.config.generate_experiment_key(experiment.experiment_path)

        # Create parameter structure
        parameters = {
            "adm": {
                "name": experiment.config.adm.name,
                "instance": experiment.config.adm.instance,
            },
            "llm": None
            if experiment.config.adm.llm_backbone is None
            else {
                "model_name": experiment.config.adm.llm_backbone,
                # Add other LLM config from structured_inference_engine if available
                **(experiment.config.adm.structured_inference_engine or {}),
            },
            "kdma_values": [
                kv.model_dump() for kv in experiment.config.alignment_target.kdma_values
            ],
            "alignment_target_id": experiment.config.alignment_target.id,
            "run_variant": experiment.config.run_variant,
        }

        # Calculate relative paths
        relative_experiment_path = experiment.experiment_path.relative_to(
            experiments_root
        )

        # Use standard file paths
        input_output_path = str(
            Path("data") / relative_experiment_path / "input_output.json"
        )
        timing_path = str(Path("data") / relative_experiment_path / "timing.json")

        # Get checksum for input_output file
        full_input_output_path = str(experiment.experiment_path / "input_output.json")
        input_output_checksum = source_file_checksums.get(full_input_output_path, "")

        # Create scenario mapping - group by actual scenario_id
        scenarios_dict = {}
        for i, item in enumerate(experiment.input_output.data):
            # Use the scenario_id as-is since we no longer add numeric suffixes
            scenario_id = item.input.scenario_id
            scene_id = "unknown"

            # Extract scene_id from full_state.meta_info.scene_id if available
            if item.input.full_state and isinstance(item.input.full_state, dict):
                meta_info = item.input.full_state.get("meta_info", {})
                if isinstance(meta_info, dict):
                    scene_id = meta_info.get("scene_id", f"scene_{i}")

            if scenario_id not in scenarios_dict:
                scores_path = None
                if experiment.scores is not None:
                    scores_path = str(
                        Path("data") / relative_experiment_path / "scores.json"
                    )

                scenarios_dict[scenario_id] = Scenario(
                    input_output=InputOutputFileInfo(
                        file=input_output_path,
                        checksum=input_output_checksum,
                        alignment_target_filter=experiment.config.alignment_target.id,
                    ),
                    scores=scores_path,
                    timing=timing_path,
                    scenes={},
                )

            scenarios_dict[scenario_id].scenes[scene_id] = SceneInfo(
                source_index=i,
                scene_id=scene_id,
                timing_s=experiment.timing.raw_times_s[i]
                if experiment.timing
                else None,
            )

        # Create enhanced experiment
        enhanced_exp = Experiment(parameters=parameters, scenarios=scenarios_dict)

        self.experiments[exp_key] = enhanced_exp

        # Update indices
        self._update_indices(exp_key, parameters, list(scenarios_dict.keys()))

        # Update file tracking
        self._update_file_info(input_output_path, input_output_checksum, exp_key)

    def _update_indices(
        self, exp_key: str, parameters: Dict[str, Any], scenario_ids: List[str]
    ):
        """Update lookup indices for the experiment."""
        adm_name = parameters["adm"]["name"]
        llm_name = parameters["llm"]["model_name"] if parameters["llm"] else "no-llm"
        kdma_key = parameters.get("kdma_key", "unaligned")  # Will be computed properly

        # Compute KDMA key from kdma_values
        if not parameters["kdma_values"]:
            kdma_key = "unaligned"
        else:
            kdma_parts = [
                f"{kv['kdma']}-{kv['value']}" for kv in parameters["kdma_values"]
            ]
            kdma_key = "_".join(sorted(kdma_parts))

        # Update indices
        if adm_name not in self.indices.by_adm:
            self.indices.by_adm[adm_name] = []
        self.indices.by_adm[adm_name].append(exp_key)

        if llm_name not in self.indices.by_llm:
            self.indices.by_llm[llm_name] = []
        self.indices.by_llm[llm_name].append(exp_key)

        if kdma_key not in self.indices.by_kdma:
            self.indices.by_kdma[kdma_key] = []
        self.indices.by_kdma[kdma_key].append(exp_key)

        for scenario_id in scenario_ids:
            if scenario_id not in self.indices.by_scenario:
                self.indices.by_scenario[scenario_id] = []
            self.indices.by_scenario[scenario_id].append(exp_key)

    def _update_file_info(self, file_path: str, checksum: str, exp_key: str):
        """Update file tracking information."""
        if file_path not in self.files:
            # Calculate file size if checksum is available (file exists)
            file_size = 0
            if checksum:
                try:
                    # Convert relative path to absolute for size calculation
                    # Remove "data/" prefix if present to get actual path
                    actual_path = file_path.replace("data/", "", 1)
                    file_size = os.path.getsize(actual_path)
                except (OSError, FileNotFoundError):
                    file_size = 0

            self.files[file_path] = FileInfo(
                checksum=checksum, size=file_size, experiments=[]
            )

        if exp_key not in self.files[file_path].experiments:
            self.files[file_path].experiments.append(exp_key)


class ChunkedExperimentData(BaseModel):
    """Chunked experiment data optimized for frontend loading."""

    chunk_id: str
    chunk_type: str  # "by_adm", "by_scenario", "by_kdma"
    experiments: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_adm_chunk(
        cls, adm_type: str, experiments: List[ExperimentData]
    ) -> "ChunkedExperimentData":
        """Create a chunk organized by ADM type."""
        return cls(
            chunk_id=f"adm_{adm_type}",
            chunk_type="by_adm",
            experiments=[exp.model_dump() for exp in experiments],
            metadata={"adm_type": adm_type, "count": len(experiments)},
        )

    @classmethod
    def create_scenario_chunk(
        cls, scenario_id: str, experiments: List[ExperimentData]
    ) -> "ChunkedExperimentData":
        """Create a chunk organized by scenario ID."""
        return cls(
            chunk_id=f"scenario_{scenario_id}",
            chunk_type="by_scenario",
            experiments=[exp.model_dump() for exp in experiments],
            metadata={"scenario_id": scenario_id, "count": len(experiments)},
        )
