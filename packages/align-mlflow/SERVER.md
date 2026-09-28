# MLflow server and storage

MLflow has three parts:

| Part | Purpose | Native MLflow setting |
| --- | --- | --- |
| Tracking database | Runs, traces, tags, scores, and saved views | Server: `MLFLOW_BACKEND_STORE_URI` |
| Artifact storage | Copied files, including Hydra configuration and metadata | `MLFLOW_DEFAULT_ARTIFACT_ROOT` for new experiments |
| Server | HTTP API and browser UI over that stored data | Clients connect with `MLFLOW_TRACKING_URI` |

The server process is separate from its data. Stopping it preserves the database and artifacts. A client can connect directly to a database or through the server's HTTP URL. Our local scripts below configure these parts from `MLFLOW_TRACKING_URI` alone.

On a new computer, clone this repository, check out the revision you want to deploy, and run `uv sync --frozen --dev` from its root. These Bash scripts run on Linux, macOS, or WSL. Use the local setup below for one computer, or [the team setup](#share-a-new-store-with-a-team) for browser access and ingestion from other computers.

## Configure storage

Keep scripts and guides in this repository and runtime data outside the checkout. For local use, set **one standard MLflow variable** in each terminal:

```bash
export MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow-store/mlflow.db
```

The scripts create the directory and use this layout:

- The specified file is the tracking database.
- An `artifacts/` directory beside it holds copied files for new experiments.

This layout is an **align-mlflow script convention**, not a new MLflow environment variable. The server script translates it into MLflow's native backend-store and artifact-root arguments. You do not need to set `MLFLOW_BACKEND_STORE_URI` or `MLFLOW_DEFAULT_ARTIFACT_ROOT` for this workflow.

Use an absolute SQLite file URI without query options. The four slashes in `sqlite:////...` include the leading slash of the filesystem path. Set the variable in each terminal, or source your own environment file; the scripts do not load `.env` automatically.

```text
align-tools/packages/align-mlflow/     Versioned code, scripts, and guides
/chosen/mlflow-store/                 Runtime data, outside Git
  mlflow.db
  mlflow.db-wal, mlflow.db-shm         SQLite working files while in use
  artifacts/<run ID>/artifacts/source/
    .hydra/{config,overrides,hydra}.yaml
    meta.json
```

Existing experiments retain their recorded artifact locations. Changing `MLFLOW_TRACKING_URI` selects another store; it does not move data. To reuse a store, point at its existing database and keep its artifact paths accessible.

## Start and stop

From the repository root after `uv sync --frozen --dev`, with storage configured:

```bash
./packages/align-mlflow/scripts/server.sh
```

The server runs in the foreground at [localhost:5000](http://localhost:5000). It writes logs to the terminal and uses one web worker. `MLFLOW_HOST`, `MLFLOW_PORT`, and `MLFLOW_WORKERS` can override those defaults.

Check it from another terminal with `curl -fsS http://localhost:5000/health`. Stop with **Ctrl-C** and wait for the process to exit. Restart with the same configuration and command. If shutdown stalls, close browser connections and inspect remaining server workers before starting another instance.

To install the optional comparison views:

```bash
./packages/align-mlflow/scripts/views.sh http://localhost:5000
```

The view recipe requires MLflow 3.16.1; see [saved views](REFERENCE.md#saved-views). For movable tag columns and hideable Input/Output in grouped sessions, install the optional [session UI](SESSION-UI.md) on the server.

## Share a new store with a team

On the server, check out the same repository revision and run `uv sync --frozen --dev` to use the committed dependency versions. To include the customized session table and action labels, install the [session UI](SESSION-UI.md) in this environment before starting the server. Choose a fresh database outside the checkout, then start MLflow with HTTP artifact serving:

```bash
export MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow-store/mlflow.db
uv run --no-sync mlflow server \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --artifacts-destination "$(uv run --no-sync python -m align_mlflow.local_store)" \
  --default-artifact-root mlflow-artifacts:/ \
  --serve-artifacts --workers 1 \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts 'mlflow.example.internal:5000,localhost:5000'
```

Replace the example hostname with the address teammates will use. Provide access through the team's private network or authenticated gateway. The host allowlist validates hostnames; it does not authenticate users.

In another terminal, on this server or an importing computer with the same checkout and dependencies:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.example.internal:5000
./packages/align-mlflow/scripts/ingest.sh /path/to/open-world-runs
./packages/align-mlflow/scripts/views.sh "$MLFLOW_TRACKING_URI"
```

Teammates open that HTTP address in a browser. The server owns the database and artifact files; clients do not need a shared filesystem. New imports include readable episode cards, and the last command installs the optional comparison tables.

Create experiments through the HTTP connection for this setup. Experiments previously created by direct SQLite ingestion keep their file artifact locations; changing server flags does not migrate them. Keep one importer active at a time, and use your server's process manager to keep MLflow running between logins.

## Deployment and maintenance

The local server script uses SQLite and direct artifact access (`--no-serve-artifacts`). For a server that is already running, clients only need `MLFLOW_TRACKING_URI=http://server:5000`; its administrator controls storage.

For PostgreSQL, separate artifact storage, or artifact proxying, use `mlflow server` directly with its native settings: `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_DEFAULT_ARTIFACT_ROOT`, or `MLFLOW_ARTIFACTS_DESTINATION`. MLflow has no single native directory variable that combines all of those deployment choices.

Back up the database and artifact storage together. For SQLite, stop writers and the server before making a filesystem copy, retaining any remaining working files. Do not delete the WAL manually. Changing artifact roots for future experiments does not migrate existing artifacts.

See MLflow's [server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) and [CLI environment variables](https://mlflow.org/docs/latest/api_reference/cli.html#server).
