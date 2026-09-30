# align-mlflow

Import align-system open-world runs into MLflow for episode inspection and ADM comparison. This package owns the importer, server scripts, and setup guides. Keep runtime data outside the checkout.

Start with the short guides:

- [Import an Align run directory](RESEARCHER-QUICKSTART.md).
- [Start a shared server](QUICKSTART.md).

For a local server, run `./packages/align-mlflow/scripts/setup.sh` from the repository root to install dependencies and the session UI, then:

1. [Set `MLFLOW_TRACKING_URI` to choose storage](SERVER.md#configure-storage).
2. [Import a run or sweep](INGESTION.md).
3. [Start the server and restore views](SERVER.md#start-and-stop).

Each source folder becomes a run, each episode a session, and each action a trace. Hydra files and metadata are copied as artifacts; source files stay unchanged.

The included [session UI](SESSION-UI.md) adds movable columns and action names to grouped sessions.
