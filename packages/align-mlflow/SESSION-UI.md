# Session UI

Adds action names and movable, hideable columns to MLflow's grouped session table. Installed on the server for everyone using its UI.

Requires Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`. From the repository root, before starting the server:

```bash
./packages/align-mlflow/scripts/setup.sh
```

Setup installs the locked **MLflow 3.16.1** environment and this UI by default. The first build downloads dependencies and needs several GB of disk space.

To reinstall only the UI, run `./packages/align-mlflow/scripts/install-ui.sh` in the existing locked environment. Stop the server before setup or reinstallation, then restart it and reload the browser. See [Start a shared MLflow server](START-SHARED-SERVER.md) for the setup steps.
