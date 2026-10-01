# Manage a shared MLflow server

## Quick Start

These steps create a new instance. For the installed ITM server, use the [service commands](#manage-the-instance-on-itm).

1. **Install the server and session UI.** Requires Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`.

   ```bash
   git clone https://github.com/PaulHax/align-tools.git
   cd align-tools
   ./packages/align-mlflow/scripts/setup.sh
   ```

2. **Configure storage.** The example uses `/data/shared/mlflow` on ITM. Edit the copied file if needed.

   ```bash
   mkdir -p /data/shared/mlflow
   cp packages/align-mlflow/examples/server.env.example /data/shared/mlflow/.env
   ```

3. **Start the server.** From the repository root:

   ```bash
   uv run --no-sync mlflow --env-file /data/shared/mlflow/.env server
   ```

   Open <http://10.50.57.47:5000>. Stop a foreground server with **Ctrl-C**. Follow [Import Align runs](INGESTION.md) to add data.

## Manage the instance on ITM

The installed instance uses the **user service** `open-world-mlflow.service`.

```bash
systemctl --user status open-world-mlflow.service --no-pager
systemctl --user start open-world-mlflow.service
systemctl --user stop open-world-mlflow.service
systemctl --user restart open-world-mlflow.service
journalctl --user -u open-world-mlflow.service -n 50 -f
```

- Run these commands over SSH in the account that owns the service; **no sudo is needed**.
- The service runs independently of a terminal and restarts after failure.
- **Lingering is currently disabled.** Boot/logout independence requires it to be enabled with administrator authorization.
- After editing `.env`, **restart the service** to apply changes.

## Files and configuration

| Location | Contents |
| --- | --- |
| `/data/shared/mlflow/.env` | Server configuration |
| `/data/shared/mlflow/mlflow.db` | Runs, traces, sessions, scores, annotations, and saved views |
| `/data/shared/mlflow/artifacts/` | Run artifacts and original source archives |

- **Runtime data stays outside Git.** Keep code, the UI patch, and configuration examples in the repository.
- **The server account owns writes.** Importers use HTTP and need no direct access to the database or artifact directory.
- **SQLite working files** (`mlflow.db-wal` and `mlflow.db-shm`) may appear while the server runs. Do not delete a live WAL.
- **ITM's copied store retains IDs**, annotations, and saved views. Its original store remains available for rollback.

### Native MLflow settings

The [environment example](examples/server.env.example) uses MLflow's built-in variables:

| Setting | Purpose |
| --- | --- |
| `MLFLOW_BACKEND_STORE_URI` | Server database URI |
| `MLFLOW_ARTIFACTS_DESTINATION` | Physical destination for HTTP artifact uploads |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | New experiments' artifact URI; `mlflow-artifacts:/` uses the server proxy |
| `MLFLOW_SERVE_ARTIFACTS` | Enable HTTP artifact upload/download |
| `MLFLOW_HOST`, `MLFLOW_PORT`, `MLFLOW_WORKERS` | Listener and worker settings |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | Allowed `host:port` values |
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | Allowed browser `scheme://host:port` origins |

- **`--env-file /path/to/server.env`** selects the configuration file, before `server`. Shell variables override file values; CLI options override both.
- **Configure both allowlists.** Missing browser origins can cause POST requests to return **403** even when the page loads.
- **No authentication is configured.** Access is through the intended internal network/VPN.

### Local SQLite launcher

- `scripts/server.sh` is a **local SQLite launcher**, using an absolute `MLFLOW_TRACKING_URI=sqlite:////path/to/mlflow.db`.
- It creates adjacent `artifacts/` and disables the HTTP artifact proxy. Use the native startup command above for the shared server.
- Server flags **do not migrate `file://` artifact locations**. Copy artifacts and update their recorded locations as a separate, verified migration.

## Upgrade

From a clean deployed checkout, after backing up. These commands deploy `origin/main`, including from an older integration checkout:

```bash
systemctl --user stop open-world-mlflow.service
git fetch origin
git switch --detach origin/main
./packages/align-mlflow/scripts/setup.sh
systemctl --user start open-world-mlflow.service
```

- Setup installs the **locked MLflow 3.16.1 environment and session UI together**. Reload browsers afterward; a global MLflow installation does not include the patch.
- Review dependency changes first. MLflow version changes may require database schema and UI patch updates.

## Back up, restore, or move

- **Back up:** stop imports and the service, then copy the entire data directory, including remaining SQLite working files. Restart afterward.
- **Restore:** stop the service, retain the current directory as a rollback copy, restore the complete backup at the configured path, and restart. Check `/health`, existing trace links, annotations, and artifact downloads.
- **Move:** preserve IDs and copy the database and artifacts together. Update configuration and any recorded absolute artifact locations; verify the copied store before cutover.
- **Concurrent use:** browsing and annotations can continue during an import. Initially, run **one bulk import at a time** to limit SQLite contention. Never overlap imports of the same source run into the same experiment.

## Slow SQLite trace search

Local SQLite `sync` refreshes query statistics automatically; **HTTP imports skip it**. After large HTTP imports, run from the server checkout on the host. This updates planner statistics and retains all data and IDs:

```bash
uv run --no-sync python - <<'PY'
import sqlite3
with sqlite3.connect("file:/data/shared/mlflow/mlflow.db?mode=rw", uri=True) as db:
    db.execute("ANALYZE")
PY
```

See MLflow's [server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) and [CLI settings](https://mlflow.org/docs/latest/api_reference/cli.html#server).
