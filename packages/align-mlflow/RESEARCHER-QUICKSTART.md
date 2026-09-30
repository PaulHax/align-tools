# Import an Align run directory

Point the importer at one run folder or a parent containing several runs. It creates MLflow runs, decision traces grouped into episode sessions, and available TA3 alignment scores. No changes to the research code are required.

## Install the importer

On ITM, use your own checkout and environment. Follow the [installation commands](QUICKSTART.md#install-in-your-own-checkout), stopping after `uv sync --frozen --dev --python 3.13.5`. No Node installation or UI build is needed for ingestion.

The importer recognizes open-world runs with `input_output.json` and `.hydra/config.yaml` selecting an open-world driver. Keep the whole run folder, including `.hydra/`, `meta.json`, and `raw_align_system.log` when present. Other Align formats are skipped; this is not a generic importer for every Align output format.

## Choose the destination and import

**Current ITM prerequisite:** complete imports currently require the operator account. Self-service imports from researchers' own accounts need the proposed [group-accessible storage migration](QUICKSTART.md#proposed-shared-storage-on-this-host) or a separately configured artifact-proxy destination. Neither change has been made yet.

For the shared ITM instance, the intended command is:

```bash
export MLFLOW_TRACKING_URI='http://10.50.57.47:5000'
export MLFLOW_EXPERIMENT_NAME='Research - project-name'
./packages/align-mlflow/scripts/ingest.sh /data/shared/project-name/run-folder
```

Replace the project name and directory. Run the command from your checkout. A parent directory works too: the importer searches recursively. Use a project-specific experiment rather than mixing new research imports into the existing **Open World Phase 2** baseline.

The importer must be able to read the source and write the experiment's artifact files. The ITM server currently records local `file://` artifact paths under Paul's private home; using its HTTP URL sends metadata through the server but does not proxy these file uploads. A shared database path alone would not change those stored artifact locations.

For a private store you own, use an absolute SQLite URI instead, then run the same importer command:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$HOME/.local/share/align-mlflow-demo/mlflow.db"
export MLFLOW_EXPERIMENT_NAME='Research - project-name'
./packages/align-mlflow/scripts/ingest.sh /data/shared/project-name/run-folder
```

The script creates local storage and an adjacent `artifacts/` directory. A private server need not be running during direct SQLite ingestion; [start it afterward](QUICKSTART.md#start-a-fresh-local-store) using this same tracking URI.

Use the package script above. The separate ITM workspace wrapper at `/home/local/KHQ/paul.elliott/src/mlflow/scripts/ingest.sh` targets a fixed Phase 2 dataset and runs its archival workflow; it is not the researcher directory importer.

## Inspect and repeat

Each successful run prints its number of new steps and episode scores. A failed run prints `not synced (...)` and the command exits nonzero. `No open-world runs found` means the folder contained no recognized runs, even though the command can exit successfully; check the source path and Hydra driver configuration.

For the shared instance, open <http://10.50.57.47:5000>, select your experiment, and open **Traces**. Choose **All time** if the recorded runs are older. Group by session to inspect episodes. For a private instance, use its local URL instead.

To add the comparison views to a new experiment, with the same destination and experiment still selected:

```bash
./packages/align-mlflow/scripts/views.sh http://10.50.57.47:5000
```

For a private server on port 5001, replace the URL with `http://127.0.0.1:5001`. This optional command prints view links and updates matching named presets; it is not required to import data.

Repeat the import to resume an interruption or add appended records. Confirmed traces retain their IDs and assessments. Keep one importer active at a time for the destination, and preserve run folder names and provenance files when copying runs. Import revised source data into a separate experiment.

The importer leaves source files unchanged. It preserves action records in traces and copies available Hydra files and `meta.json` as artifacts; it does **not** archive every original file. Full-folder archival in the ITM prototype is a separate, dataset-specific operation. See [ingestion details](INGESTION.md) and the [adapter reference](REFERENCE.md) for coverage and recovery behavior.
