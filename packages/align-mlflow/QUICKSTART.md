# Start an align-mlflow server

Use this guide to start a fresh server for inspecting imported Align runs. To point the importer at a run directory, see the [researcher quickstart](RESEARCHER-QUICKSTART.md).

## Install in your own checkout

Install Git and `uv`. This revision reproduces the current ITM prototype with MLflow **3.16.1** and Python **3.13.5**:

```bash
git clone --branch open-world-traces https://github.com/PaulHax/align-tools.git
cd align-tools
git checkout 3581a2772e752d082735709ece9ca8b4eb64bf65
uv sync --frozen --dev --python 3.13.5
```

For the patched session UI, put Node **24.14.0** on `PATH`, then run:

```bash
./packages/align-mlflow/scripts/install-ui.sh
```

The UI is installed into this checkout's environment, not a global MLflow installation. Node is needed for the UI build; importers do not need it.

## Start a fresh local store

From the checkout, choose a new persistent directory outside Git. The example uses your home directory and port **5001**, leaving ITM's existing server on port 5000 alone:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$HOME/.local/share/align-mlflow-demo/mlflow.db"
export MLFLOW_EXPERIMENT_NAME='Research - demo'
export MLFLOW_HOST=127.0.0.1
export MLFLOW_PORT=5001
./packages/align-mlflow/scripts/server.sh
```

Open <http://127.0.0.1:5001>. Check it with `curl -fsS http://127.0.0.1:5001/health`. The process runs in the foreground; stop it with **Ctrl-C**. Restart with the same settings to reuse the store. This launcher uses SQLite and local artifacts, suitable for an importer on the same computer with filesystem access.

In another terminal, set the same tracking URI and experiment and follow the [directory import recipe](RESEARCHER-QUICKSTART.md). For a new server that accepts complete imports from other computers, use the [HTTP artifact-proxy setup](SERVER.md#share-a-new-store-with-a-team).

## Existing shared server on ITM

The existing instance is <http://10.50.57.47:5000>. It runs as `paul.elliott` under the enabled user service `open-world-mlflow.service`, independently of a tmux session or Codex window. Its unit is `~/.config/systemd/user/open-world-mlflow.service`; its SQLite database and artifacts currently live in Paul's private workspace.

These management commands run as Paul on ITM. Another user's `systemctl --user` controls that user's own services:

```bash
systemctl --user status open-world-mlflow.service --no-pager
systemctl --user start open-world-mlflow.service
systemctl --user stop open-world-mlflow.service
systemctl --user restart open-world-mlflow.service
tail -n 50 -f /home/local/KHQ/paul.elliott/src/mlflow/logs/server.log
```

Do not start another foreground server on port 5000. Lingering is disabled, so boot/logout independence is not established. There is currently no authentication; access is through the intended internal network/VPN.

## Proposed shared storage on this host

For researchers importing `/data/shared` runs from their own accounts, the proposed state directory is `/data/shared/mlflow`, containing the database, `artifacts/`, and `source-archives/`. On ITM, `/data/shared` is on local ext4 storage and is already shared through the `itm` group. Use group read/write permissions, setgid directories, and a matching service/importer umask such as `0007` so new files remain accessible to the group. Source folders must also be readable by the importing user. Researchers can keep their code and Python environments in their own checkouts.

Configure the storage root once in the admin/server tools. A proposed wrapper option such as `--data-dir /data/shared/mlflow` would translate into MLflow's native backend and artifact settings; it is not an existing MLflow option or an implemented tool here. Researchers importing into the shared instance configure the server URL and experiment, then supply their source directory. They should not need the SQLite URI or Paul's account.

| Configuration | Where it belongs |
| --- | --- |
| `/data/shared/mlflow` state directory, permissions, and storage mode | Admin/server deployment configuration |
| `sqlite:////data/shared/mlflow/mlflow.db` backend URI | Server configuration |
| Artifact destination and whether uploads are proxied | Server configuration |
| HTTP tracking URL, experiment, and `/data/shared/...` source folder | Researcher importer |

For proxied uploads, the server owns artifact writes and researchers need only source-read and HTTP access. For the current direct-file mode, researchers also need artifact filesystem permissions. MLflow's tracking REST API manages tracking objects; it does not provide a general operation to relocate the server's database and artifact storage. Keep storage reconfiguration in the admin deployment tools and rehearse migrations before changing the service configuration.

This storage change has **not** been performed. Moving the SQLite file alone is insufficient: existing experiment and run artifact URIs point into Paul's home. A migration must back up the database and artifacts together, copy all state, update recorded artifact locations, and verify IDs, annotations, saved views, links, and artifact hashes on an isolated copy before cutover. Preserve existing IDs and retain the original store as a rollback copy.

See [server and storage details](SERVER.md) for artifact configuration and maintenance.
