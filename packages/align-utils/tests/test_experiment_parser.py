"""Tests for experiment parsing functionality."""

import json
import yaml
import tempfile
from pathlib import Path
from align_utils.models import (
    KDMAValue,
    ADMConfig,
    ExperimentConfig,
    InputOutputFile,
    ScoresFile,
    ExperimentData,
    ChunkedExperimentData,
)
from align_utils.discovery import (
    parse_experiments_directory,
    build_manifest_from_experiments,
)


def create_sample_config_data():
    """Create sample config.yaml data for testing."""
    return {
        "name": "action_based",
        "adm": {
            "name": "pipeline_random",
            "instance": {
                "_target_": "align_system.algorithms.pipeline_adm.PipelineADM",
                "steps": ["step1", "step2"],
            },
            "structured_inference_engine": {"model_name": "llama3.3-70b"},
        },
        "alignment_target": {
            "id": "ADEPT-June2025-affiliation-0.5",
            "kdma_values": [{"kdes": None, "kdma": "affiliation", "value": 0.5}],
        },
    }


def create_sample_input_output_data():
    """Create sample input_output.json data for testing."""
    return [
        {
            "input": {
                "scenario_id": "June2025-AF-train",
                "alignment_target_id": "ADEPT-June2025-affiliation-0.5",
                "full_state": {
                    "unstructured": "Test scenario description",
                    "characters": [],
                },
                "state": "Test scenario",
                "choices": [
                    {
                        "action_id": "treat_patient_a",
                        "action_type": "TREAT_PATIENT",
                        "unstructured": "Treat Patient A",
                    }
                ],
            },
            "output": {
                "choice": 0,
                "action": {
                    "action_id": "treat_patient_a",
                    "action_type": "TREAT_PATIENT",
                    "unstructured": "Treat Patient A",
                    "justification": "Test justification",
                },
            },
            "choice_info": {
                "true_kdma_values": {
                    "Treat Patient A": {"affiliation": 0.0, "medical": 0.99}
                }
            },
        }
    ]


def create_sample_scores_data():
    """Create sample scores.json data for testing."""
    return [
        {
            "alignment_source": [
                {"scenario_id": "June2025-AF-train", "probes": ["Probe 1", "Probe 2"]}
            ]
        }
    ]


def create_sample_timing_data():
    """Create sample timing.json data for testing."""
    return {
        "scenarios": [
            {
                "n_actions_taken": 92,
                "total_time_s": 0.026,
                "avg_time_s": 0.0003,
                "max_time_s": 0.0005,
                "raw_times_s": [0.0003, 0.0004, 0.0002],
            }
        ],
        "raw_times_s": [0.0003, 0.0004, 0.0002],
    }


def test_kdma_value_model():
    """Test KDMAValue model."""
    kdma = KDMAValue(kdma="affiliation", value=0.5)
    assert kdma.kdma == "affiliation"
    assert kdma.value == 0.5
    assert kdma.kdes is None


def test_adm_config_model():
    """Test ADMConfig model."""
    adm = ADMConfig(
        name="test_adm", structured_inference_engine={"model_name": "llama3.3-70b"}
    )
    assert adm.name == "test_adm"
    assert adm.llm_backbone == "llama3.3-70b"


def test_adm_config_no_llm():
    """Test ADMConfig model with no LLM backbone."""
    adm = ADMConfig(name="test_adm")
    assert adm.llm_backbone is None


def test_experiment_config_key_generation():
    """Test ExperimentConfig key generation."""
    config_data = create_sample_config_data()
    config = ExperimentConfig(**config_data)

    key = config.generate_key()
    expected_key = "pipeline_random:llama3.3-70b:affiliation-0.5:default"
    assert key == expected_key


def test_input_output_file_model():
    """Test InputOutputFile model."""
    data = create_sample_input_output_data()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        input_output = InputOutputFile.from_file(temp_path)
        assert len(input_output.data) == 1
        assert input_output.first_scenario_id == "June2025-AF-train"
        assert input_output.data[0].input.scenario_id == "June2025-AF-train"
    finally:
        temp_path.unlink()


def test_scores_file_model():
    """Test ScoresFile model."""
    data = create_sample_scores_data()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        scores = ScoresFile.from_file(temp_path)
        assert len(scores.data) == 1
    finally:
        temp_path.unlink()


def test_experiment_data_from_directory():
    """Test loading ExperimentData from a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create experiment directory structure
        experiment_dir = temp_path / "test_experiment"
        experiment_dir.mkdir()
        hydra_dir = experiment_dir / ".hydra"
        hydra_dir.mkdir()

        # Create config.yaml
        config_data = create_sample_config_data()
        with open(hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        # Create input_output.json
        input_output_data = create_sample_input_output_data()
        with open(experiment_dir / "input_output.json", "w") as f:
            json.dump(input_output_data, f)

        # Create scores.json
        scores_data = create_sample_scores_data()
        with open(experiment_dir / "scores.json", "w") as f:
            json.dump(scores_data, f)

        # Create timing.json
        timing_data = create_sample_timing_data()
        with open(experiment_dir / "timing.json", "w") as f:
            json.dump(timing_data, f)

        # Test loading (no experiments_root, so no directory context)
        experiment = ExperimentData.from_directory(experiment_dir)

        assert experiment.key == "pipeline_random:llama3.3-70b:affiliation-0.5:default"
        assert experiment.scenario_id == "June2025-AF-train"
        assert experiment.config.adm.name == "pipeline_random"
        assert len(experiment.input_output.data) == 1
        assert len(experiment.scores.data) == 1
        assert len(experiment.timing.scenarios) == 1


def test_has_required_files():
    """Test checking for required files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        experiment_dir = temp_path / "test_experiment"
        experiment_dir.mkdir()

        # Should fail with no files
        assert not ExperimentData.has_required_files(experiment_dir)

        # Create required files one by one
        hydra_dir = experiment_dir / ".hydra"
        hydra_dir.mkdir()
        (hydra_dir / "config.yaml").touch()
        assert not ExperimentData.has_required_files(experiment_dir)

        (experiment_dir / "input_output.json").touch()
        assert ExperimentData.has_required_files(experiment_dir)

        # Optional files don't change the result
        (experiment_dir / "scores.json").touch()
        assert ExperimentData.has_required_files(experiment_dir)

        (experiment_dir / "timing.json").touch()
        assert ExperimentData.has_required_files(experiment_dir)


def test_parse_experiments_directory():
    """Test parsing an experiments directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create experiments structure
        experiments_root = temp_path / "experiments"
        experiments_root.mkdir()

        pipeline_dir = experiments_root / "pipeline_test"
        pipeline_dir.mkdir()

        experiment_dir = pipeline_dir / "affiliation-0.5"
        experiment_dir.mkdir()
        hydra_dir = experiment_dir / ".hydra"
        hydra_dir.mkdir()

        # Create required files
        config_data = create_sample_config_data()
        with open(hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        with open(experiment_dir / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)

        with open(experiment_dir / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)

        with open(experiment_dir / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Test parsing
        experiments = parse_experiments_directory(experiments_root)
        assert len(experiments) == 1
        assert (
            experiments[0].key == "pipeline_random:llama3.3-70b:affiliation-0.5:default"
        )


def test_parse_experiments_directory_excludes_outdated():
    """Test that parse_experiments_directory correctly excludes OUTDATED directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create experiments structure
        experiments_root = temp_path / "experiments"
        experiments_root.mkdir()

        # Create a valid experiment directory
        pipeline_dir = experiments_root / "pipeline_test"
        pipeline_dir.mkdir()

        valid_experiment = pipeline_dir / "affiliation-0.5"
        valid_experiment.mkdir()
        hydra_dir = valid_experiment / ".hydra"
        hydra_dir.mkdir()

        # Create required files for valid experiment
        config_data = create_sample_config_data()
        with open(hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        with open(valid_experiment / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)
        with open(valid_experiment / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)
        with open(valid_experiment / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Create OUTDATED experiment directory with all required files
        outdated_experiment = pipeline_dir / "OUTDATED-affiliation-0.5"
        outdated_experiment.mkdir()
        outdated_hydra_dir = outdated_experiment / ".hydra"
        outdated_hydra_dir.mkdir()

        # Create required files for OUTDATED experiment (same structure)
        with open(outdated_hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        with open(outdated_experiment / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)
        with open(outdated_experiment / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)
        with open(outdated_experiment / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Test parsing - should only find the valid experiment, not the OUTDATED one
        experiments = parse_experiments_directory(experiments_root)
        assert len(experiments) == 1, f"Expected 1 experiment, found {len(experiments)}"
        assert (
            experiments[0].key == "pipeline_random:llama3.3-70b:affiliation-0.5:default"
        )

        # Verify the OUTDATED experiment was actually excluded
        experiment_paths = [str(exp.experiment_path) for exp in experiments]
        assert not any("OUTDATED" in path for path in experiment_paths), (
            f"OUTDATED experiment was not filtered out: {experiment_paths}"
        )


def test_run_variant_conflict_resolution():
    """Test that run_variant is added to experiment keys when conflicts occur."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create experiments structure with conflicts
        experiments_root = temp_path / "experiments"
        experiments_root.mkdir()

        # Create two experiments with identical parameters but in different directories
        # First experiment: experiments/pipeline_test/affiliation-0.5
        pipeline1_dir = experiments_root / "pipeline_test"
        pipeline1_dir.mkdir()
        experiment1_dir = pipeline1_dir / "affiliation-0.5"
        experiment1_dir.mkdir()
        hydra1_dir = experiment1_dir / ".hydra"
        hydra1_dir.mkdir()

        # Second experiment: experiments/pipeline_test_rerun/affiliation-0.5
        pipeline2_dir = experiments_root / "pipeline_test_rerun"
        pipeline2_dir.mkdir()
        experiment2_dir = pipeline2_dir / "affiliation-0.5"
        experiment2_dir.mkdir()
        hydra2_dir = experiment2_dir / ".hydra"
        hydra2_dir.mkdir()

        # Create identical config data for both experiments
        config_data = create_sample_config_data()

        # Setup first experiment
        with open(hydra1_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        with open(experiment1_dir / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)
        with open(experiment1_dir / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)
        with open(experiment1_dir / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Setup second experiment (identical config)
        with open(hydra2_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        with open(experiment2_dir / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)
        with open(experiment2_dir / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)
        with open(experiment2_dir / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Parse experiments and build manifest
        experiments = parse_experiments_directory(experiments_root)
        assert len(experiments) == 2, (
            f"Expected 2 experiments, found {len(experiments)}"
        )

        # Build manifest with conflict resolution
        manifest = build_manifest_from_experiments(experiments, experiments_root)

        # Verify that conflicts were resolved with run_variant in experiment keys
        experiment_keys = list(manifest.experiments.keys())
        assert len(experiment_keys) == 2, (
            f"Expected 2 unique experiment keys, got {len(experiment_keys)}"
        )

        # Check that experiment keys are hash-based
        for key in experiment_keys:
            assert key.startswith("exp_"), f"Expected hash-based key: {key}"

        # Check that run_variant is in the parameters
        has_run_variant = False
        for exp_key in experiment_keys:
            parameters = manifest.experiments[exp_key].parameters
            if parameters["run_variant"] and parameters["run_variant"] != "default":
                has_run_variant = True
                break
        assert has_run_variant, "Expected at least one experiment with run_variant"

        # Verify scenario IDs remain unchanged (no directory context in scenario IDs)
        all_scenarios = []
        for experiment in manifest.experiments.values():
            all_scenarios.extend(experiment.scenarios.keys())

        for scenario_id in all_scenarios:
            assert scenario_id.startswith("June2025"), (
                f"Scenario ID should not have directory context: {scenario_id}"
            )


def test_build_manifest_from_experiments():
    """Test building manifest from experiments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        experiments_root = temp_path / "experiments"
        experiments_root.mkdir()

        # Create a complete experiment structure for testing
        pipeline_dir = experiments_root / "test_pipeline"
        pipeline_dir.mkdir()
        experiment_dir = pipeline_dir / "test_experiment"
        experiment_dir.mkdir()
        hydra_dir = experiment_dir / ".hydra"
        hydra_dir.mkdir()

        # Create required files
        config_data = create_sample_config_data()
        with open(hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        with open(experiment_dir / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)

        with open(experiment_dir / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)

        with open(experiment_dir / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        # Load real experiment instead of using mocks
        experiment = ExperimentData.from_directory(experiment_dir)
        experiments = [experiment]

        manifest = build_manifest_from_experiments(experiments, experiments_root)

        # Check that at least one experiment was added (hash-based keys)
        assert len(manifest.experiments) >= 1, "Should have at least one experiment"

        # Get first experiment key
        first_key = list(manifest.experiments.keys())[0]
        assert first_key.startswith("exp_"), f"Expected hash-based key: {first_key}"

        # Check experiment structure
        experiment_obj = manifest.experiments[first_key]
        assert "scenarios" in experiment_obj.model_dump()
        assert len(experiment_obj.scenarios) >= 1, "Should have at least one scenario"


def test_chunked_experiment_data_model():
    """Test ChunkedExperimentData model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        experiments_root = temp_path / "experiments"
        experiments_root.mkdir()

        # Create sample experiment
        pipeline_dir = experiments_root / "pipeline_test"
        pipeline_dir.mkdir()
        experiment_dir = pipeline_dir / "test_experiment"
        experiment_dir.mkdir()
        hydra_dir = experiment_dir / ".hydra"
        hydra_dir.mkdir()

        # Create required files
        config_data = create_sample_config_data()
        with open(hydra_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        with open(experiment_dir / "input_output.json", "w") as f:
            json.dump(create_sample_input_output_data(), f)

        with open(experiment_dir / "scores.json", "w") as f:
            json.dump(create_sample_scores_data(), f)

        with open(experiment_dir / "timing.json", "w") as f:
            json.dump(create_sample_timing_data(), f)

        experiment = ExperimentData.from_directory(experiment_dir)

        # Test ADM chunk creation
        adm_chunk = ChunkedExperimentData.create_adm_chunk(
            "pipeline_random", [experiment]
        )
        assert adm_chunk.chunk_id == "adm_pipeline_random"
        assert adm_chunk.chunk_type == "by_adm"
        assert len(adm_chunk.experiments) == 1
        assert adm_chunk.metadata["adm_type"] == "pipeline_random"
        assert adm_chunk.metadata["count"] == 1

        # Test scenario chunk creation
        scenario_chunk = ChunkedExperimentData.create_scenario_chunk(
            "June2025-AF-train", [experiment]
        )
        assert scenario_chunk.chunk_id == "scenario_June2025-AF-train"
        assert scenario_chunk.chunk_type == "by_scenario"
        assert len(scenario_chunk.experiments) == 1
        assert scenario_chunk.metadata["scenario_id"] == "June2025-AF-train"
        assert scenario_chunk.metadata["count"] == 1


def run_all_tests():
    """Run all tests."""
    tests = [
        test_kdma_value_model,
        test_adm_config_model,
        test_adm_config_no_llm,
        test_experiment_config_key_generation,
        test_input_output_file_model,
        test_scores_file_model,
        test_experiment_data_from_directory,
        test_has_required_files,
        test_parse_experiments_directory,
        test_parse_experiments_directory_excludes_outdated,
        test_run_variant_conflict_resolution,
        test_build_manifest_from_experiments,
        test_chunked_experiment_data_model,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
