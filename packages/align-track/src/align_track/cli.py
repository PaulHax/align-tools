"""Command-line interface for align-track."""

import subprocess
import sys
from pathlib import Path

import click
import mlflow
import pandas as pd

from .ingest import ingest_experiments_directory


@click.group()
@click.version_option()
def cli():
    """Track and manage align-system experiments with MLflow."""
    pass


@cli.command()
@click.option(
    "--mlflow-uri",
    type=str,
    default="sqlite:///mlflow.db",
    help="MLflow tracking URI (default: sqlite:///mlflow.db)"
)
@click.option(
    "--port",
    type=int,
    default=5000,
    help="Port to run MLflow UI on (default: 5000)"
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)"
)
def ui(mlflow_uri: str, port: int, host: str):
    """Launch MLflow UI for browsing experiments.

    \b
    Examples:
        align-track ui
        align-track ui --mlflow-uri sqlite:///mlflow.db --port 8080
    """
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", mlflow_uri,
        "--port", str(port),
        "--host", host
    ]
    click.echo(f"Starting MLflow UI at http://{host}:{port}")
    click.echo(f"Tracking URI: {mlflow_uri}")
    subprocess.run(cmd)


@cli.command()
@click.option(
    "--mlflow-uri",
    type=str,
    required=True,
    help="MLflow tracking URI"
)
@click.option(
    "--experiment-name",
    type=str,
    help="Filter by experiment name"
)
@click.option(
    "--adm",
    type=str,
    help="Filter by ADM name"
)
@click.option(
    "--llm",
    type=str,
    help="Filter by LLM backbone"
)
@click.option(
    "--min-score",
    type=float,
    help="Filter by minimum score"
)
@click.option(
    "--filter",
    "filter_string",
    type=str,
    help="MLflow filter string (e.g., \"params.adm = 'pipeline_baseline'\")"
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of runs to return (default: 100)"
)
def search(
    mlflow_uri: str,
    experiment_name: str,
    adm: str,
    llm: str,
    min_score: float,
    filter_string: str,
    limit: int
):
    """Search MLflow runs with filters.

    \b
    Examples:
        align-track search --mlflow-uri ./mlruns --adm pipeline_baseline
        align-track search --mlflow-uri ./mlruns --min-score 0.8
        align-track search --mlflow-uri ./mlruns --filter "params.adm = 'pipeline_baseline'"
    """
    mlflow.set_tracking_uri(mlflow_uri)

    filters = []
    if adm:
        filters.append(f"params.adm = '{adm}'")
    if llm:
        filters.append(f"params.llm_backbone = '{llm}'")
    if min_score is not None:
        filters.append(f"metrics.`score.alignment_score` >= {min_score}")
    if filter_string:
        filters.append(filter_string)

    combined_filter = " AND ".join(filters) if filters else None

    experiment_names = [experiment_name] if experiment_name else None

    runs = mlflow.search_runs(
        experiment_names=experiment_names,
        filter_string=combined_filter,
        max_results=limit
    )

    if runs.empty:
        click.echo("No runs found matching criteria.")
        return

    display_cols = ["run_id", "experiment_id", "status"]
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]

    cols_to_show = display_cols + param_cols[:5] + metric_cols[:3]
    cols_to_show = [c for c in cols_to_show if c in runs.columns]

    click.echo(f"\nFound {len(runs)} runs:\n")
    click.echo(runs[cols_to_show].to_string())


@cli.command()
@click.option(
    "--mlflow-uri",
    type=str,
    required=True,
    help="MLflow tracking URI"
)
@click.option(
    "--experiment-name",
    type=str,
    help="Filter by experiment name"
)
@click.option(
    "--filter",
    "filter_string",
    type=str,
    help="MLflow filter string"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "tsv", "json"]),
    default="csv",
    help="Output format (default: csv)"
)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path"
)
def export(
    mlflow_uri: str,
    experiment_name: str,
    filter_string: str,
    output_format: str,
    output: Path
):
    """Export MLflow runs to CSV/TSV/JSON.

    \b
    Examples:
        align-track export --mlflow-uri ./mlruns -o results.csv
        align-track export --mlflow-uri ./mlruns --experiment-name "LOO Rerun" -o results.csv
        align-track export --mlflow-uri ./mlruns --format json -o results.json
    """
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_names = [experiment_name] if experiment_name else None

    runs = mlflow.search_runs(
        experiment_names=experiment_names,
        filter_string=filter_string
    )

    if runs.empty:
        click.echo("No runs found matching criteria.")
        return

    if output_format == "csv":
        runs.to_csv(output, index=False)
    elif output_format == "tsv":
        runs.to_csv(output, sep="\t", index=False)
    elif output_format == "json":
        runs.to_json(output, orient="records", indent=2)

    click.echo(f"Exported {len(runs)} runs to {output}")


@cli.command()
@click.option(
    "--mlflow-uri",
    type=str,
    required=True,
    help="MLflow tracking URI"
)
@click.option(
    "--experiment-name",
    type=str,
    help="Filter by experiment name"
)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output Excel file path"
)
@click.option(
    "--rows",
    type=str,
    default="alignment_target_id",
    help="Parameter for pivot table rows (default: alignment_target_id)"
)
@click.option(
    "--cols",
    type=str,
    default="adm",
    help="Parameter for pivot table columns (default: adm)"
)
@click.option(
    "--values",
    type=str,
    default="timing.avg_seconds",
    help="Metric for pivot table values (default: timing.avg_seconds)"
)
def report(
    mlflow_uri: str,
    experiment_name: str,
    output: Path,
    rows: str,
    cols: str,
    values: str
):
    """Generate pivot table report from MLflow runs.

    \b
    Examples:
        align-track report --mlflow-uri ./mlruns -o report.xlsx
        align-track report --mlflow-uri ./mlruns --rows alignment_target_id --cols adm -o report.xlsx
        align-track report --mlflow-uri ./mlruns --values "metrics.score.alignment_score" -o report.xlsx
    """
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_names = [experiment_name] if experiment_name else None

    runs = mlflow.search_runs(experiment_names=experiment_names)

    if runs.empty:
        click.echo("No runs found.")
        return

    row_col = f"params.{rows}" if not rows.startswith(("params.", "metrics.", "tags.")) else rows
    col_col = f"params.{cols}" if not cols.startswith(("params.", "metrics.", "tags.")) else cols
    val_col = f"metrics.{values}" if not values.startswith(("params.", "metrics.", "tags.")) else values

    missing_cols = []
    if row_col not in runs.columns:
        missing_cols.append(row_col)
    if col_col not in runs.columns:
        missing_cols.append(col_col)
    if val_col not in runs.columns:
        missing_cols.append(val_col)

    if missing_cols:
        click.echo(f"Missing columns in data: {missing_cols}")
        click.echo(f"Available param columns: {[c for c in runs.columns if c.startswith('params.')]}")
        click.echo(f"Available metric columns: {[c for c in runs.columns if c.startswith('metrics.')]}")
        return

    pivot = pd.pivot_table(
        runs,
        values=val_col,
        index=row_col,
        columns=col_col,
        aggfunc="mean"
    )

    pivot.to_excel(output)
    click.echo(f"Generated pivot table report: {output}")
    click.echo(f"  Rows: {row_col}")
    click.echo(f"  Columns: {col_col}")
    click.echo(f"  Values: {val_col} (mean)")


@cli.command("list-runs")
@click.option(
    "--mlflow-uri",
    type=str,
    required=True,
    help="MLflow tracking URI"
)
@click.option(
    "--experiment-name",
    type=str,
    help="Filter by experiment name"
)
@click.option(
    "--with-links",
    is_flag=True,
    help="Include align-app deep links"
)
@click.option(
    "--experiments-root",
    type=click.Path(exists=True, path_type=Path),
    help="Root path for experiments (required for deep links)"
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of runs to list (default: 50)"
)
def list_runs(
    mlflow_uri: str,
    experiment_name: str,
    with_links: bool,
    experiments_root: Path,
    limit: int
):
    """List MLflow runs with optional align-app deep links.

    \b
    Examples:
        align-track list-runs --mlflow-uri ./mlruns
        align-track list-runs --mlflow-uri ./mlruns --with-links --experiments-root /data/experiments
    """
    mlflow.set_tracking_uri(mlflow_uri)

    experiment_names = [experiment_name] if experiment_name else None

    runs = mlflow.search_runs(
        experiment_names=experiment_names,
        max_results=limit
    )

    if runs.empty:
        click.echo("No runs found.")
        return

    click.echo(f"\nFound {len(runs)} runs:\n")

    for _, run in runs.iterrows():
        run_id = run.get("run_id", "unknown")
        adm = run.get("params.adm", "unknown")
        target = run.get("params.alignment_target_id", "unknown")
        exp_path = run.get("tags.experiment_path", "")

        click.echo(f"Run: {run_id[:8]}")
        click.echo(f"  ADM: {adm}")
        click.echo(f"  Target: {target}")

        if with_links and exp_path and experiments_root:
            full_path = experiments_root / exp_path
            link = f"align-app --experiment {full_path}"
            click.echo(f"  Link: {link}")

        click.echo()


@cli.command()
@click.option(
    "--experiments-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing experiments to ingest"
)
@click.option(
    "--mlflow-uri",
    type=str,
    default="sqlite:///mlflow.db",
    help="MLflow tracking URI (default: sqlite:///mlflow.db)"
)
@click.option(
    "--experiment-name",
    type=str,
    default="align-system-experiments",
    help="Name of MLflow experiment (default: align-system-experiments)"
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-ingest experiments that already exist"
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress progress output"
)
def ingest(
    experiments_dir: Path,
    mlflow_uri: str,
    experiment_name: str,
    force: bool,
    quiet: bool,
):
    """Ingest experiments from a directory into MLflow.

    Creates one run per experiment with traces for each scene.
    Traces capture input/output text, decisions, and KDMA values for drill-down.

    \b
    Examples:
        # Ingest experiments
        align-track ingest --experiments-dir /path/to/experiments

        # Specify MLflow URI and experiment name
        align-track ingest \\
            --experiments-dir /path/to/experiments \\
            --mlflow-uri sqlite:///mlflow.db \\
            --experiment-name "My Experiment"

        # Force re-ingest existing experiments
        align-track ingest --experiments-dir /path/to/experiments --force
    """
    summary = ingest_experiments_directory(
        experiments_dir=experiments_dir,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        force=force,
        verbose=not quiet,
    )

    if summary.failed > 0:
        raise click.ClickException(f"Failed to ingest {summary.failed} experiments")


if __name__ == "__main__":
    cli()
