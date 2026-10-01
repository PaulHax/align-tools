# Session UI

- **Adds:** action names and movable, hideable columns in grouped sessions.
- **Install location:** the server; everyone using its UI gets the patch.

**Requires:** Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`. From the repository root, before starting the server:

```bash
./packages/align-mlflow/scripts/setup.sh
```

- **Default setup:** installs locked MLflow **3.16.1** and the UI together.
- **First build:** downloads dependencies and needs several GB of disk space.

**Reinstall only the UI:** `./packages/align-mlflow/scripts/install-ui.sh`. Stop the server before setup or reinstallation, then restart and reload browsers. See [server setup](SERVER.md#quick-start).
