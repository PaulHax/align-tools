# align-mlflow

Import align-system open-world runs into MLflow for episode inspection and ADM comparison. This package owns the importer, server scripts, and setup guides. Keep runtime data outside the checkout.

From the repository root, run `uv sync --dev`, then:

1. [Set `MLFLOW_TRACKING_URI` to choose storage](SERVER.md#configure-storage).
2. [Import a run or sweep](INGESTION.md).
3. [Start the server and restore views](SERVER.md#start-and-stop).

Each source folder becomes a run, each episode a session, and each action a trace. Hydra files and metadata are copied as artifacts; source files stay unchanged.

See the [adapter reference](REFERENCE.md) for data mapping, saved views, and recovery behavior.
