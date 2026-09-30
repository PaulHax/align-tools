# MLflow server and storage

For a short startup recipe, see the [server quickstart](QUICKSTART.md).

MLflow has three parts:

| Part | Purpose | Native MLflow setting |
| --- | --- | --- |
| Tracking database | Runs, traces, tags, scores, and saved views | Server: `MLFLOW_BACKEND_STORE_URI` |
| Artifact storage | Copied files, including Hydra configuration and metadata | Server: `MLFLOW_ARTIFACTS_DESTINATION` for HTTP uploads; `MLFLOW_DEFAULT_ARTIFACT_ROOT` for new experiments' artifact URIs |
| Server | HTTP API and browser UI over that stored data | Clients connect with `MLFLOW_TRACKING_URI` |

The server process is separate from its data. Stopping it preserves the database and artifacts. A client can connect directly to a database or through the server's HTTP URL. Our local scripts below configure these parts from `MLFLOW_TRACKING_URI` alone.

Clone this repository and check out the revision you want to deploy. For a server, run `./packages/align-mlflow/scripts/setup.sh` from its root to install locked dependencies and the [session UI](SESSION-UI.md). For an importer only, use `uv sync --frozen --dev`. These Bash scripts run on Linux, macOS, or WSL. Use the local setup below for one computer, or [the team setup](#share-a-new-store-with-a-team) for browser access and ingestion from other computers.

## Configure storage

Keep scripts and guides in this repository and runtime data outside the checkout. For local use, set **one standard MLflow variable** in each terminal:

```bash
export MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow-store/mlflow.db
```

The scripts create the directory and use this layout:

- The specified file is the tracking database.
- An `artifacts/` directory beside it holds copied files for new experiments.

This layout is an **align-mlflow script convention**, not a new MLflow environment variable. The local server script sets MLflow's native backend-store and artifact-root environment variables. You do not need to set `MLFLOW_BACKEND_STORE_URI` or `MLFLOW_DEFAULT_ARTIFACT_ROOT` for this workflow.

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

From the repository root after server setup, with storage configured:

```bash
./packages/align-mlflow/scripts/server.sh
```

The server runs in the foreground at [localhost:5000](http://localhost:5000). It writes logs to the terminal and uses one web worker. `MLFLOW_HOST`, `MLFLOW_PORT`, and `MLFLOW_WORKERS` can override those defaults.

Check it from another terminal with `curl -fsS http://localhost:5000/health`. Stop with **Ctrl-C** and wait for the process to exit. Restart with the same configuration and command. If shutdown stalls, close browser connections and inspect remaining server workers before starting another instance.

To install the optional comparison views:

```bash
./packages/align-mlflow/scripts/views.sh http://localhost:5000
```

The view recipe requires MLflow 3.16.1 and updates matching named presets.

## Share a new store with a team

On the server, check out the same repository revision and run `./packages/align-mlflow/scripts/setup.sh` to install the locked environment and session UI. Copy the [server environment example](examples/server.env.example), edit it, then start MLflow:

```bash
cp packages/align-mlflow/examples/server.env.example .env
uv run --no-sync mlflow --env-file .env server
```

The example uses a fresh SQLite database at `/data/shared/mlflow/mlflow.db`, artifacts at `/data/shared/mlflow/artifacts`, and port 5000 on ITM. The server account must be able to write there. MLflow creates the database's parent directory and artifact directories as needed.

These are native MLflow settings. With no configuration, the server normally uses `sqlite:///mlflow.db` and `./mlartifacts` in its working directory; artifact serving is enabled by default. The example makes storage paths and HTTP artifact serving explicit. `MLFLOW_TRACKING_URI` is the client destination, not the server's backend configuration.

Choose another file with `--env-file /path/to/server.env` (before `server`). MLflow 3.16.1 loads it explicitly; it does not automatically load `.env`. Existing shell variables take precedence over the file, and explicit CLI options take precedence over both. The checkout's `.env` is ignored by Git.

If you change the address or port, update both the host allowlist (`host:port`) and browser-origin allowlist (`scheme://host:port`); browser POSTs can fail even when the HTML page loads if the actual Origin is missing. Provide access through the team's private network or authenticated gateway. These allowlists do not authenticate users.

In another terminal, on this server or an importing computer with the same checkout and dependencies:

```bash
export MLFLOW_TRACKING_URI=http://10.50.57.47:5000
./packages/align-mlflow/scripts/ingest.sh /path/to/open-world-runs
./packages/align-mlflow/scripts/views.sh "$MLFLOW_TRACKING_URI"
```

Teammates open that HTTP address in a browser. The server owns the database and artifact files; clients do not need a shared filesystem. New imports include readable episode cards, and the last command installs the optional comparison tables.

Create experiments through the HTTP connection for this setup. Experiments previously created by direct SQLite ingestion keep their file artifact locations; changing server settings does not migrate them. Use your server's process manager to keep MLflow running between logins.

Multiple HTTP clients can use the server concurrently. SQLite permits one write transaction at a time and queues competing writes; sustained contention can cause lock errors. For the initial SQLite setup, run bulk imports one at a time while other users browse or annotate. Do not overlap imports of the same source run into the same experiment: the importer's resume checks are not atomic, regardless of the database backend. See SQLite's [concurrency guidance](https://sqlite.org/whentouse.html).

## Deployment and maintenance

The local server script uses SQLite and direct artifact access (`MLFLOW_SERVE_ARTIFACTS=false`). For a server that is already running, clients select it with `MLFLOW_TRACKING_URI=http://server:5000`; its administrator controls storage. Complete imports also require clients to access the experiment's recorded artifact location. With local `file://` artifacts, that means filesystem permissions; with proxied artifacts, uploads go through HTTP.

For PostgreSQL, separate artifact storage, or artifact proxying, use `mlflow server` directly with its native settings: `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_DEFAULT_ARTIFACT_ROOT`, or `MLFLOW_ARTIFACTS_DESTINATION`. MLflow has no single native directory variable that combines all of those deployment choices.

Back up the database and artifact storage together. For SQLite, stop writers and the server before making a filesystem copy, retaining any remaining working files. Do not delete the WAL manually. Changing artifact roots for future experiments does not migrate existing artifacts.

See MLflow's [server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) and [CLI environment variables](https://mlflow.org/docs/latest/api_reference/cli.html#server).
