# Frontend development

The integration is maintained in `sessions.patch` against the checksum-pinned
MLflow source in `install.py`. Keep editable upstream sources and build output
outside the repository checkout.

Use Node 24.14 or later within 24.x. From the repository root, install the locked
Python environment and build the current frontend once:

```bash
uv sync --frozen --dev
./packages/align-mlflow/scripts/install-ui.sh
```

Create a separate editable source tree from the downloaded archive, with the
current patch applied:

```bash
uv run --no-sync python packages/align-mlflow/ui/update_patch.py prepare \
  --archive ~/.cache/align-mlflow/ui/3.17.2/mlflow-v3.17.2.tar.gz \
  --source /tmp/open-world-ui
```

Edit TypeScript under `/tmp/open-world-ui/mlflow/server/js/src/`. Run that
frontend's bundled Yarn install and build commands with Node 24 on `PATH`:

```bash
cd /tmp/open-world-ui/mlflow/server/js
node yarn/releases/yarn-4.12.0.cjs install --immutable
node yarn/releases/yarn-4.12.0.cjs build
```

From the repository root, regenerate the maintained patch. The command verifies
the source archive, applies the candidate patch in a temporary tree, and checks
that it reproduces every changed source file before writing the output:

```bash
uv run --no-sync python packages/align-mlflow/ui/update_patch.py update \
  --archive ~/.cache/align-mlflow/ui/3.17.2/mlflow-v3.17.2.tar.gz \
  --source /tmp/open-world-ui
```

Run the [browser tests](e2e/README.md) against the production build. Commit the
maintained patch, related documentation, and reusable tests together. Reinstall
the frontend and restart the server when deploying a changed patch.
