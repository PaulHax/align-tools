#!/usr/bin/env python3
"""CLI for querying SQLite experiment database."""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .sqlite_tracker import SQLiteTracker


def format_duration(started: str, ended: Optional[str]) -> str:
    """Format run duration."""
    if not started or not ended:
        return "-"
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(ended)
        delta = end - start
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "-"


def cmd_list(args: List[str], tracker: SQLiteTracker) -> int:
    """List experiments and runs."""
    experiment = None
    status = None
    limit = 20

    i = 0
    while i < len(args):
        if args[i] in ["-e", "--experiment"]:
            experiment = args[i + 1]
            i += 2
        elif args[i] in ["-s", "--status"]:
            status = args[i + 1]
            i += 2
        elif args[i] in ["-n", "--limit"]:
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1

    runs = tracker.query_runs(
        experiment_name=experiment,
        status=status,
        limit=limit,
    )

    if not runs:
        print("No runs found.")
        return 0

    # Print header
    print(f"{'ID':<6} {'Experiment':<20} {'Run':<25} {'Status':<10} {'Duration':<15}")
    print("-" * 80)

    for run in runs:
        duration = format_duration(run["started_at"], run["ended_at"])
        print(
            f"{run['id']:<6} "
            f"{run['experiment_name'][:19]:<20} "
            f"{(run['run_name'] or '-')[:24]:<25} "
            f"{run['status']:<10} "
            f"{duration:<15}"
        )

    print(f"\nTotal: {len(runs)} runs")
    return 0


def cmd_show(args: List[str], tracker: SQLiteTracker) -> int:
    """Show details of a run."""
    if not args:
        print("Usage: experiments show <run_id> [--metrics]", file=sys.stderr)
        return 1

    run_id = int(args[0])
    show_metrics = "--metrics" in args or "-m" in args

    runs = tracker.query_runs()
    run = next((r for r in runs if r["id"] == run_id), None)

    if not run:
        print(f"Run {run_id} not found.", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"Run ID: {run['id']}")
    print(f"Experiment: {run['experiment_name']}")
    print(f"Name: {run['run_name'] or '-'}")
    print(f"Status: {run['status']}")
    print(f"Started: {run['started_at']}")
    print(f"Ended: {run['ended_at'] or '-'}")
    duration = format_duration(run["started_at"], run["ended_at"])
    print(f"Duration: {duration}")

    if run["error_message"]:
        print(f"Error: {run['error_message']}")

    # Show parameters
    params = tracker.get_run_parameters(run_id)
    if params:
        print(f"\n{'Parameters':-^60}")
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}")

    # Show metrics if requested
    if show_metrics:
        metrics = tracker.get_run_metrics(run_id)
        if metrics:
            print(f"\n{'Metrics':-^60}")
            # Group by key, show last 5 values
            from collections import defaultdict

            grouped = defaultdict(list)
            for m in metrics:
                grouped[m["key"]].append((m["step"], m["value"]))

            for key, values in sorted(grouped.items()):
                print(f"  {key}:")
                for step, value in values[-5:]:
                    step_str = f"step {step}" if step is not None else ""
                    print(f"    {step_str}: {value:.4f}")
                if len(values) > 5:
                    print(f"    ... ({len(values) - 5} more)")

    print(f"{'='*60}\n")
    return 0


def cmd_compare(args: List[str], tracker: SQLiteTracker) -> int:
    """Compare runs."""
    if len(args) < 2:
        print("Usage: experiments compare <run_id1> <run_id2> ...", file=sys.stderr)
        return 1

    run_ids = [int(x) for x in args]
    comparison = tracker.compare_runs(run_ids)

    if not comparison:
        print("No parameters found for comparison.")
        return 0

    # Print header
    header = "Parameter" + "".join(f" | Run {rid}" for rid in run_ids)
    print(header)
    print("-" * len(header))

    for key in sorted(comparison.keys()):
        values = comparison[key]
        row = key
        for rid in run_ids:
            val = values.get(rid, "-")
            row += f" | {val}"
        print(row)

    return 0


def cmd_query(args: List[str], tracker: SQLiteTracker) -> int:
    """Run custom SQL query."""
    if not args:
        print("Usage: experiments query '<SQL query>'", file=sys.stderr)
        return 1

    sql = " ".join(args)
    try:
        results = tracker.query_sql(sql)
        if not results:
            print("No results.")
            return 0

        # Print as table
        keys = list(results[0].keys())
        print(" | ".join(keys))
        print("-" * (sum(len(k) for k in keys) + 3 * (len(keys) - 1)))
        for row in results:
            print(" | ".join(str(row[k]) for k in keys))

        print(f"\n{len(results)} rows")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_help() -> int:
    """Show help."""
    print("""
Experiment Tracking CLI

Usage: experiments <command> [options]

Commands:
  list     List experiments and runs
           -e, --experiment <name>  Filter by experiment
           -s, --status <status>    Filter by status (running, completed, failed)
           -n, --limit <n>          Limit results (default: 20)

  show     Show details of a run
           experiments show <run_id> [--metrics]

  compare  Compare parameters across runs
           experiments compare <run_id1> <run_id2> ...

  query    Run custom SQL query
           experiments query "SELECT * FROM runs LIMIT 5"

Options:
  -d, --db <path>  Database path (default: experiments.db)
  -h, --help       Show this help message

Examples:
  experiments list
  experiments list -e my-experiment -s completed
  experiments show 42 --metrics
  experiments compare 42 43 44
  experiments query "SELECT key, AVG(value) FROM metrics GROUP BY key"
""")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        return cmd_help()

    # Parse global options
    db_path = "experiments.db"
    i = 0
    while i < len(args):
        if args[i] in ["-d", "--db"]:
            db_path = args[i + 1]
            args = args[:i] + args[i + 2:]
        else:
            i += 1

    if not args:
        return cmd_help()

    # Check if database exists for read commands
    command = args[0]
    remaining = args[1:]

    if command not in ["help"] and not Path(db_path).exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        print("Create experiments first, or specify --db path.", file=sys.stderr)
        return 1

    tracker = SQLiteTracker(db_path)

    if command == "list":
        return cmd_list(remaining, tracker)
    elif command == "show":
        return cmd_show(remaining, tracker)
    elif command == "compare":
        return cmd_compare(remaining, tracker)
    elif command == "query":
        return cmd_query(remaining, tracker)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        return cmd_help()


if __name__ == "__main__":
    sys.exit(main())
