# Session UI

This optional frontend patch for **MLflow 3.16.1** lets the grouped session table respect column visibility and order, including custom tag columns. Expanded table rows, episode sidebars, and card headings show the trace's action name. Unnamed traces retain their turn number.

This changes MLflow's frontend. The ingestion package and database format are unchanged. The patch and build script live in this repository; installing it on the server makes the same UI available to all clients.

## Install on the server

Use the repository's locked Python environment, Node.js **24.14+ within 24.x**, and Git. First [configure storage](SERVER.md#configure-storage), then run from the repository root:

```bash
uv sync --frozen --dev
./packages/align-mlflow/scripts/install-ui.sh
./packages/align-mlflow/scripts/server.sh
```

Stop an existing server before installing, then restart it with its usual storage configuration. The installer verifies the upstream source checksum, applies [the patch](ui/sessions.patch), installs locked frontend dependencies, builds the UI, and replaces the frontend in this checkout's Python environment. It preserves a copy of the stock frontend.

The first build downloads the upstream source and dependencies and needs several GB of disk space. Build files default to `~/.cache/align-mlflow/ui/3.16.1`; use `--build-dir /path/to/cache` to choose another location. No new environment variables are required.

## Use and share

Install the [comparison views](REFERENCE.md#saved-views) after importing:

```bash
./packages/align-mlflow/scripts/views.sh http://localhost:5000
```

Open **Session comparison** from the Views menu. It groups actions into episode rows with ADM, model, source version, alignment target, and scenario columns. Use the filter control to select an ADM or alignment target. Source version identifies the recorded producer revision; it is not an independently recorded ADM release.

In **Display > Columns**, toggle Input and Output, drag columns, or use Ctrl+Up/Down on a column item. Headers can also be dragged. Save the view to share the column order and filters with teammates; ordinary display preferences remain specific to a browser.

Session summaries cover the traces in the current filtered page. Filtering by action-level fields can show only part of an episode. ADM and alignment target are appropriate session comparison filters because they are constant within an episode.

## Restore or upgrade

```bash
./packages/align-mlflow/scripts/install-ui.sh --restore
```

Restart the server and reload the browser after either installation or restoration. Reinstalling MLflow can replace the frontend; rerun the installer if needed. Upgrading MLflow requires reviewing and rebuilding the patch against that release. The installer rejects other versions rather than applying an unverified patch.
