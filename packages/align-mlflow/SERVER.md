# Manage a shared MLflow server

## Quick Start

1. **Install the server and session UI.** Requires Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`.

   ```bash
   git clone --branch open-world-traces https://github.com/PaulHax/align-tools.git
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
| `/data/shared/mlflow/migration-verification.json` | Verification of the copied store and artifact hashes |

- **Runtime data stays outside Git.** Code, the UI patch, and the configuration example belong in this repository.
- **The server account owns writes.** Importers use HTTP and need no direct access to the database or artifact directory.
- **SQLite working files** (`mlflow.db-wal` and `mlflow.db-shm`) may appear while the server runs. Do not delete a live WAL.
- **Existing IDs are retained** in the ITM copy, including traces, assessments, and saved views. The original store is retained for rollback.

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

- Choose another file with **`--env-file /path/to/server.env`**, before `server`.
- MLflow **3.16.1** loads that file explicitly. Existing shell variables override file values; CLI options override both.
- **Configure both allowlists.** Missing browser origins can cause POST requests to return **403** even when the page loads.
- **No authentication is configured.** Access is through the intended internal network/VPN.
- **Client destination:** set `MLFLOW_TRACKING_URI=http://10.50.57.47:5000` when logging or importing.

### Local SQLite launcher

- `scripts/server.sh` is a **local SQLite launcher**, using an absolute `MLFLOW_TRACKING_URI=sqlite:////path/to/mlflow.db`.
- It creates adjacent `artifacts/` and disables the HTTP artifact proxy. Use the native startup command above for the shared server.
- Changing server settings **does not migrate existing data** or convert recorded `file://` locations. Copy files and update recorded artifact locations as a separate, verified migration.

## Upgrade

From the deployed checkout, with the service stopped:

```bash
systemctl --user stop open-world-mlflow.service
git pull --ff-only
./packages/align-mlflow/scripts/setup.sh
systemctl --user start open-world-mlflow.service
```

- Setup installs the **locked environment and session UI together**. Reload browsers afterward.
- The UI and maintenance commands require **MLflow 3.16.1**. A different global installation does not include this patch.
- Review dependency changes before upgrading MLflow; database schema upgrades and UI patch updates need a separate plan.

## Back up, restore, or move

- **Back up:** stop imports and the service, then copy the entire data directory, including remaining SQLite working files. Restart afterward.
- **Restore:** stop the service, retain the current directory as a rollback copy, restore the complete backup at the configured path, and restart. Check `/health`, existing trace links, annotations, and artifact downloads.
- **Move:** preserve IDs and copy the database and artifacts together. Update configuration and any recorded absolute artifact locations; verify the copied store before cutover.
- **Concurrent use:** browsing and annotations can continue during an import. Initially, run **one bulk import at a time** to limit SQLite contention. Never overlap imports of the same source run into the same experiment.

See MLflow's [server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) and [CLI settings](https://mlflow.org/docs/latest/api_reference/cli.html#server).
