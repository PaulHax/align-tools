"""Import completed open-world runs, or optionally watch growing runs."""

from pathlib import Path

import click

from .store import connect
from .refresh_cards import refresh_cards as refresh_card_payloads
from .refresh_components import refresh_components as refresh_component_payloads
from .sync import SyncResult, sync_run, sync_tree
from .watch import watch as watch_runs

_path_argument = click.argument(
    "path", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
_tracking_uri_option = click.option(
    "--tracking-uri",
    envvar="MLFLOW_TRACKING_URI",
    default="sqlite:///mlflow.db",
    show_default=True,
    help="MLflow tracking URI. Point `mlflow server --backend-store-uri` at the same store.",
)
_experiment_option = click.option(
    "--experiment",
    default="align-system open world",
    show_default=True,
    help="MLflow experiment that receives the traces.",
)


def describe(result: SyncResult) -> str:
    if result.error is not None:
        return f"{result.run_dir}: not synced ({result.error})"
    return (
        f"{result.run_label}: {result.new_steps} new steps, "
        f"{result.new_scores} new episode scores"
    )


@click.group()
def cli() -> None:
    """Load align-system open-world runs into MLflow.

    Start with sync PATH to import a completed run or sweep, then open the
    MLflow UI on the same store. Use watch for optional inspection during a run.

    Each step becomes a trace and each episode a session; group traces by
    session in the MLflow UI to read an episode turn by turn.
    """


@cli.command()
@_path_argument
@_tracking_uri_option
@_experiment_option
def sync(path: Path, tracking_uri: str, experiment: str) -> None:
    """Import a completed run or sweep at PATH, then exit.

    PATH can be one run directory or a parent containing several runs.
    Rerun to resume interrupted imports or add missing steps and scores.
    Available steps can be imported even if episode outcomes are absent.
    """
    experiment_id = connect(tracking_uri, experiment)
    results = sync_tree(experiment_id, path)
    if not results:
        click.echo(f"No open-world runs found under {path}")
    for result in results:
        click.echo(describe(result))
    if any(result.error is not None for result in results):
        raise SystemExit(1)


@cli.command()
@_tracking_uri_option
@_experiment_option
@click.option(
    "--dry-run", is_flag=True, help="Report older cards without changing them"
)
def refresh_cards(tracking_uri: str, experiment: str, dry_run: bool) -> None:
    """Give previously imported traces readable cards without replacing their IDs.

    Requires a local MLflow 3.16.1 SQLite store. Back up the store first.
    New imports already use these cards. Repeating this command skips current cards.
    """
    try:
        for updated, current in refresh_card_payloads(
            tracking_uri, experiment, dry_run
        ):
            action = "would refresh" if dry_run else "refreshed"
            click.echo(f"{updated} cards {action}, {current} already current")
    except ValueError as error:
        raise click.ClickException(str(error)) from error


@cli.command()
@_path_argument
@_tracking_uri_option
@_experiment_option
@click.option("--dry-run", is_flag=True, help="Report changes without writing them")
def refresh_components(
    path: Path, tracking_uri: str, experiment: str, dry_run: bool
) -> None:
    """Enrich existing component spans using source runs at PATH.

    Requires a local MLflow 3.16.1 SQLite store. Back up the store first and stop
    other importers. Trace IDs, decision cards, timing, and reviews are retained.
    New imports include component evidence automatically.
    """
    try:
        for source, count, statuses in refresh_component_payloads(
            tracking_uri, experiment, path, dry_run
        ):
            action = "would refresh" if dry_run else "refreshed"
            click.echo(f"{source}: {count} components {action}; log matches {statuses}")
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error


@cli.command()
@_path_argument
@_tracking_uri_option
@_experiment_option
@click.option(
    "--interval",
    default=2.0,
    show_default=True,
    help="Seconds between checks for files align-system has written.",
)
def watch(path: Path, tracking_uri: str, experiment: str, interval: float) -> None:
    """Optionally follow a growing run or sweep at PATH.

    Uses the same importer and store as sync. Stop with Ctrl-C.
    """
    experiment_id = connect(tracking_uri, experiment)

    def sync_one(run_dir: Path) -> SyncResult:
        return sync_run(experiment_id, run_dir)

    click.echo(f"Watching {path} for open-world runs")
    watch_runs(path, sync_one, lambda result: click.echo(describe(result)), interval)
