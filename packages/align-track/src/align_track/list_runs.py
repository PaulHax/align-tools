#!/usr/bin/env python3
"""CLI tool to display a table of runs from an align-utils manifest."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from align_utils.discovery import parse_experiments_directory


def format_table_row(columns: List[Tuple[str, int]]) -> str:
    """Format a single table row with proper spacing."""
    return " | ".join(
        value.ljust(width) if value else " " * width for value, width in columns
    )


def create_separator(widths: List[int]) -> str:
    """Create a table separator line."""
    return "-+-".join("-" * width for width in widths)


def calculate_column_widths(headers: List[str], rows: List[Dict]) -> List[int]:
    """Calculate optimal column widths for the table."""

    def max_width(header: str, key: str) -> int:
        values = [str(row.get(key, "")) for row in rows]
        return max(len(header), max(map(len, values)) if values else 0)

    return [
        max_width("Run Path", "path"),
        max_width("ADM Name", "adm_name"),
        max_width("Alignment", "alignment"),
        max_width("Scenarios", "num_scenarios"),
    ]


def format_table(runs: List[Dict]) -> List[str]:
    """Format runs data as a table."""
    if not runs:
        return ["No runs found."]

    headers = ["Run Path", "ADM Name", "Alignment", "Scenarios"]
    widths = calculate_column_widths(headers, runs)

    header_row = format_table_row(list(zip(headers, widths)))
    separator = create_separator(widths)

    data_rows = [
        format_table_row(
            [
                (run.get("path", ""), widths[0]),
                (run.get("adm_name", ""), widths[1]),
                (run.get("alignment", ""), widths[2]),
                (str(run.get("num_scenarios", 0)), widths[3]),
            ]
        )
        for run in runs
    ]

    return [header_row, separator] + data_rows


def process_manifest_path(manifest_path: Path) -> List[Dict]:
    """Process a manifest file or directory containing runs using align-utils."""
    # Parse experiments using align-utils
    experiments = parse_experiments_directory(manifest_path)

    # Convert experiments to run dictionaries
    runs = []
    for experiment in experiments:
        # Extract data source from config or use default
        data_source = "Unknown"
        if hasattr(experiment.config, "datasource"):
            data_source = experiment.config.datasource
        elif hasattr(experiment.config, "interface"):
            data_source = getattr(experiment.config.interface, "input", "Unknown")

        runs.append(
            {
                "path": str(experiment.experiment_path.name),
                "adm_name": experiment.config.adm.name,
                "data_source": data_source,
                "num_scenarios": len(experiment.input_output.data),
                "alignment": experiment.config.alignment_target.id,
            }
        )

    return sorted(runs, key=lambda x: x["path"])


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI tool."""
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help"]:
        print("Usage: list_runs <manifest_file_or_directory>")
        print(
            "\nDisplay a table of all runs from an align-utils manifest or directory."
        )
        print("\nA run is defined as a directory containing:")
        print("  - input_output.json")
        print("  - .hydra/config.yaml")
        return 0

    manifest_path = Path(args[0])

    if not manifest_path.exists():
        print(f"Error: Path does not exist: {manifest_path}", file=sys.stderr)
        return 1

    try:
        runs = process_manifest_path(manifest_path)
        table_lines = format_table(runs)

        for line in table_lines:
            print(line)

        print(f"\nTotal runs: {len(runs)}")
        return 0

    except Exception as e:
        print(f"Error processing manifest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
