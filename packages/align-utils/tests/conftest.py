#!/usr/bin/env python3
"""
Shared pytest fixtures for align-utils testing with real experiment data.
"""

import time
import urllib.request
import zipfile
from pathlib import Path
import pytest
import shutil


@pytest.fixture(scope="session")
def real_experiment_data():
    """Download and prepare real experiment data for testing.

    This fixture downloads the same test experiment data used by align-browser
    and makes it available for align-utils tests.
    """
    # Get the package root directory
    tests_dir = Path(__file__).parent
    package_root = tests_dir.parent

    # Use a dedicated directory for downloaded experiment data
    experiment_data_dir = package_root / "experiment-data"
    test_experiments_dir = experiment_data_dir / "test-experiments"
    lock_file = experiment_data_dir / ".download_lock"

    # Download experiments if test-experiments directory doesn't exist
    if not test_experiments_dir.exists():
        # Create the experiment data directory
        experiment_data_dir.mkdir(exist_ok=True)

        # Simple file-based lock for cross-platform compatibility
        max_wait = 60  # seconds
        wait_time = 0

        # Wait if another process is downloading
        while lock_file.exists() and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5

        # Check again after waiting (another process might have completed the download)
        if not test_experiments_dir.exists():
            try:
                # Create lock file to signal we're downloading
                lock_file.touch()

                print("Downloading experiment data for align-utils tests...")

                zip_path = experiment_data_dir / "experiments.zip"

                # Download the zip file (same URL as align-browser uses)
                url = "https://github.com/PaulHax/align-browser/releases/download/v0.2.1/experiments.zip"
                print(f"Downloading from {url}...")

                urllib.request.urlretrieve(url, zip_path)
                print(f"Downloaded to {zip_path}")

                # Extract the zip file to a temporary directory first
                print(f"Extracting {zip_path}...")
                temp_extract_dir = experiment_data_dir / "temp_extract"
                temp_extract_dir.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_extract_dir)

                # Find extracted directories in temp location
                extracted_items = list(temp_extract_dir.iterdir())
                extracted_dirs = [item for item in extracted_items if item.is_dir()]

                if len(extracted_dirs) == 1:
                    # Single directory - rename it to test-experiments
                    original_dir = extracted_dirs[0]
                    original_dir.rename(test_experiments_dir)
                    print(f"Renamed {original_dir.name} to test-experiments")
                elif len(extracted_dirs) > 1:
                    # Multiple directories - create test-experiments and move all under it
                    test_experiments_dir.mkdir(exist_ok=True)
                    for extracted_dir in extracted_dirs:
                        target_path = test_experiments_dir / extracted_dir.name
                        extracted_dir.rename(target_path)
                    print(
                        f"Moved {len(extracted_dirs)} directories under test-experiments"
                    )
                else:
                    # No directories found - this shouldn't happen but handle gracefully
                    print("Warning: No directories found in extracted zip")

                # Clean up temporary extraction directory
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)

                # Delete the zip file after extraction
                zip_path.unlink()
                print(f"Extracted to {experiment_data_dir}")
                print("Experiment data ready for testing!")

            except Exception as e:
                print(f"Error downloading experiment data: {e}")
                # Clean up on error
                if zip_path.exists():
                    zip_path.unlink()
                raise

            finally:
                # Clean up temporary extraction directory if it exists
                temp_extract_dir = experiment_data_dir / "temp_extract"
                if temp_extract_dir.exists():
                    shutil.rmtree(temp_extract_dir)

                # Always remove lock file
                if lock_file.exists():
                    lock_file.unlink()

    # Return the path to the test experiments
    if test_experiments_dir.exists() and any(test_experiments_dir.iterdir()):
        return test_experiments_dir
    else:
        return None


@pytest.fixture(scope="session")
def experiments_path(real_experiment_data):
    """Provide the path to real experiment data for tests.

    This fixture ensures experiment data is downloaded and returns the path.
    Tests can check if the path is None to skip if data is not available.
    """
    return real_experiment_data
