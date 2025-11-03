"""Simple tests for experiment parsing using real experiment data."""

import os
from pathlib import Path
import pytest
from align_utils.models import ExperimentData, Manifest
from align_utils.discovery import (
    parse_experiments_directory,
    build_manifest_from_experiments,
)


def get_experiments_path():
    """Get the path to the experiments directory.

    Priority order:
    1. Downloaded test data in experiment-data/test-experiments
    2. Path configured via environment variable TEST_EXPERIMENTS_PATH
    3. Default relative path "../../experiments"

    Returns:
        Path: The path to the experiments directory
    """
    # First check for downloaded test data
    package_root = Path(__file__).parent.parent
    downloaded_path = package_root / "experiment-data" / "test-experiments"
    if downloaded_path.exists() and any(downloaded_path.iterdir()):
        return downloaded_path

    # Then check environment variable
    env_path = os.environ.get("TEST_EXPERIMENTS_PATH")
    if env_path:
        return Path(env_path)
    else:
        # Default fallback to relative path (one directory above the root)
        return Path("../../experiments")


def check_experiments_path_exists():
    """Check if the experiments directory exists and return status.

    Returns:
        tuple: (exists: bool, path: Path, message: str)
    """
    experiments_path = get_experiments_path()

    if experiments_path.exists():
        message = f"✅ Experiments directory found at {experiments_path}"
        return (
            True,
            experiments_path,
            message,
        )
    else:
        env_var_set = "TEST_EXPERIMENTS_PATH" in os.environ
        if env_var_set:
            message = f"❌ Experiments directory not found at {experiments_path} (from TEST_EXPERIMENTS_PATH)"
        else:
            message = f"❌ Experiments directory not found at {experiments_path} (default path)"
        return False, experiments_path, message


def get_experiments_path_or_skip():
    """Get experiments path, or return None if it doesn't exist (for skipping tests).

    This function prints a message about the status and returns None if the path
    doesn't exist, which can be used to skip tests gracefully.

    Returns:
        Path or None: The experiments path if it exists, None otherwise
    """
    exists, path, message = check_experiments_path_exists()
    print(message)

    if not exists:
        env_var_set = "TEST_EXPERIMENTS_PATH" in os.environ
        if not env_var_set:
            print(
                "💡 Tip: Set TEST_EXPERIMENTS_PATH environment variable to specify a custom experiments directory"
            )
        return None

    return path


def test_parse_real_experiments():
    """Test parsing the real experiments directory."""
    experiments_root = get_experiments_path_or_skip()

    if not experiments_root:
        print("⏭️ Skipping test - experiments directory not available")
        return

    print(f"🔍 Parsing experiments from {experiments_root.resolve()}")

    experiments = parse_experiments_directory(experiments_root)
    print(f"✅ Successfully parsed {len(experiments)} experiments")

    if experiments:
        # Test the first experiment
        first_exp = experiments[0]
        print(f"📋 First experiment key: {first_exp.key}")
        print(f"📋 First experiment scenario: {first_exp.scenario_id}")
        print(f"📋 First experiment config ADM name: {first_exp.config.adm.name}")
        print(f"📋 First experiment path: {first_exp.experiment_path}")

        # Test key generation
        assert first_exp.key and first_exp.key != "unknown_adm_no_llm_", (
            "Key generation may have issues"
        )
        print("✅ Key generation working correctly")


def test_build_manifest():
    """Test building manifest from real experiments."""
    experiments_root = get_experiments_path_or_skip()

    if not experiments_root:
        print("⏭️ Skipping test - experiments directory not available")
        return

    experiments = parse_experiments_directory(experiments_root)
    manifest = build_manifest_from_experiments(experiments, experiments_root)

    print(
        f"✅ Built manifest with {len(manifest.experiments)} unique experiment configurations"
    )

    # Check manifest structure
    for key, value in list(manifest.experiments.items())[:3]:  # Show first 3
        scenarios = value.scenarios
        print(f"📋 Config '{key}' has {len(scenarios)} scenarios")

    # Verify manifest structure
    assert manifest, "Empty manifest generated"
    assert isinstance(manifest, Manifest), "Should return Manifest instance"

    if manifest.experiments:
        first_key = list(manifest.experiments.keys())[0]
        first_experiment = manifest.experiments[first_key]

        assert hasattr(first_experiment, "scenarios"), "Experiment missing scenarios"
        assert hasattr(first_experiment, "parameters"), "Experiment missing parameters"

        if first_experiment.scenarios:
            first_scenario = list(first_experiment.scenarios.values())[0]
            required_fields = ["input_output", "timing"]  # scores is optional

            for field in required_fields:
                assert hasattr(first_scenario, field), f"Scenario missing {field} field"

    print("✅ Manifest structure is correct")


def test_experiment_data_loading():
    """Test loading individual experiment data."""
    experiments_root = get_experiments_path_or_skip()

    if not experiments_root:
        print("⏭️ Skipping test - experiments directory not available")
        return

    # Find first valid experiment directory
    experiment_dir = None
    for pipeline_dir in experiments_root.iterdir():
        if not pipeline_dir.is_dir():
            continue
        for exp_dir in pipeline_dir.glob("*"):
            if exp_dir.is_dir() and ExperimentData.has_required_files(exp_dir):
                experiment_dir = exp_dir
                break
        if experiment_dir:
            break

    if not experiment_dir:
        print("⏭️ Skipping test - no valid experiment directories found")
        return

    print(f"🔍 Loading experiment from {experiment_dir}")

    # Test loading
    experiment = ExperimentData.from_directory(experiment_dir)

    # Validate basic properties
    assert experiment is not None, "Failed to load experiment"
    assert experiment.config is not None, "Config not loaded"
    assert experiment.input_output is not None, "Input/output not loaded"
    assert experiment.timing is not None, "Timing not loaded"

    # Test key generation
    assert experiment.key is not None, "Key not generated"
    assert experiment.key != "unknown_adm_no_llm_", "Invalid key generation"

    # Test scenario ID
    assert experiment.scenario_id is not None, "Scenario ID not detected"

    print(f"✅ Successfully loaded experiment: {experiment.key}")
    print(f"📋 Scenario: {experiment.scenario_id}")
    print(f"📋 ADM: {experiment.config.adm.name}")
    print(f"📋 LLM: {experiment.config.adm.llm_backbone}")

    # Test alignment target
    if experiment.config.alignment_target:
        print(f"📋 Alignment Target: {experiment.config.alignment_target.id}")
        print(
            f"📋 KDMAs: {[kv.kdma for kv in experiment.config.alignment_target.kdma_values]}"
        )


def test_parse_real_experiments_if_available(experiments_path):
    """Test parsing real experiments directory if available.

    Uses the experiments_path fixture which automatically downloads test data.
    """
    if not experiments_path:
        pytest.skip("Real experiment data not available")

    print(f"🔍 Using experiment data from: {experiments_path}")

    experiments = parse_experiments_directory(experiments_path)
    print(f"✅ Successfully parsed {len(experiments)} real experiments")

    if experiments:
        # Test that at least one experiment was parsed correctly
        first_exp = experiments[0]
        assert first_exp.key is not None
        assert first_exp.scenario_id is not None
        assert first_exp.config.adm.name is not None
        print(f"✅ Real experiment validation passed: {first_exp.key}")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_parse_real_experiments,
        test_build_manifest,
        test_experiment_data_loading,
        test_parse_real_experiments_if_available,
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
