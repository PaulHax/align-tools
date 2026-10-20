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

After the first import, install the comparison views for the chosen experiment using the [server setup step](SERVER.md#quick-start). Select **Session comparison** from the Views menu to show scenario, ADM, KDMA alignment target, driver, model, and run together. The filter control accepts the same recorded tags. In **Display > Columns**, hide or reorder columns and save the layout as another named view.

Older imports retain their recorded `adm`, `llm`, `scenario_id`, alignment target, and run values. A missing driver or model stays missing; installing a view does not infer metadata or rewrite traces. Source version shows the recorded version string; a dirty version does not identify the exact executed code.
