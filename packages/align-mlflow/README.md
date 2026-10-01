# align-mlflow

Import align-system open-world runs into MLflow for episode inspection and ADM comparison. This package owns the importer, server scripts, and setup guides. Keep runtime data outside the checkout.

The two guides start with a Quick Start:

- [Import Align runs](INGESTION.md): ingest a run or study directory and open the results.
- [Manage a shared MLflow server](SERVER.md): install, configure, start, stop, upgrade, and back up.

Each source folder becomes a run, each episode a session, and each action a trace. Hydra files and metadata are copied as artifacts; source files stay unchanged.

The included [session UI](SESSION-UI.md) adds movable columns and action names to grouped sessions.
