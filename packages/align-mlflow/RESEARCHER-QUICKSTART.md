# Import an Align run directory

Use the URL of a configured shared MLflow server.

1. **Install the importer once** (requires Git and `uv`):

   ```bash
   git clone --branch open-world-traces https://github.com/PaulHax/align-tools.git
   cd align-tools
   uv sync --frozen --dev
   ```

2. **Import your directory.** From your `align-tools` checkout, replace the server URL, experiment name, and source path:

   ```bash
   MLFLOW_TRACKING_URI='http://YOUR_HOST:5001' \
   MLFLOW_EXPERIMENT_NAME='Research - my-project' \
   ./packages/align-mlflow/scripts/ingest.sh /path/to/align-run
   ```

   Supply one run folder or a parent directory containing several runs.

3. **View the results.** Open your server URL, select your experiment, and open **Traces**. Group by **Session**; choose **All time** for older runs.

Rerun step 2 to resume or add records. For the initial SQLite setup, run one bulk import at a time; browsing and annotations can continue. See [ingestion details](INGESTION.md) for supported formats and troubleshooting.
