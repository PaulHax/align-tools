# Dashboard browser tests

These tests serve the installed, patched MLflow production UI and route its API
requests to deterministic examples. They do not start MLflow or read a database,
researcher files, or a shared server.

Use Node.js 24.14 or later within 24.x. From the repository root:

```bash
./packages/align-mlflow/scripts/setup.sh
cd packages/align-mlflow/ui/e2e
npm ci
npx playwright install chromium
npm test
```

The asset server discovers the installed UI through the repository's Python
environment. To test a separate production build, set `MLFLOW_UI_BUILD_DIR` to
its directory when running `npm test`.

The normal example contains eight runs and 192 episodes, including unaligned
baselines and one requested/evaluated target mismatch. A 100-run example checks
natural expanded height, keyboard collapsing, horizontal scroll containment, and full metadata access without relying on pixel snapshots.
Unexpected API requests fail the test so new server dependencies remain explicit.

A three-session example includes tied and backward timestamps to check consistent
decision ordering and explicit selection in individual and comparison views.

The UI workflow builds the maintained patch, installs Chromium, and runs this
suite. Playwright reports and failure traces are saved as workflow artifacts.
