# Start a shared MLflow server

Choose a new, writable data directory and an unused port. This creates a fresh store; it does not migrate existing data.

1. **Install** (requires Git and `uv`):

   ```bash
   git clone --branch open-world-traces https://github.com/PaulHax/align-tools.git
   cd align-tools
   uv sync --frozen --dev
   ```

2. **Start the server.** Replace the directory and hostname with your own:

   ```bash
   store_dir='/absolute/path/to/mlflow-store'
   public_host='mlflow.example.internal'
   port=5001
   mkdir -p "$store_dir/artifacts"
   uv run --no-sync mlflow server \
     --backend-store-uri "sqlite:///$store_dir/mlflow.db" \
     --artifacts-destination "$store_dir/artifacts" \
     --default-artifact-root mlflow-artifacts:/ \
     --serve-artifacts --host 0.0.0.0 --port "$port" --workers 1 \
     --allowed-hosts "$public_host:$port,localhost:$port" \
     --cors-allowed-origins "http://$public_host:$port,http://localhost:$port"
   ```

3. **Open `http://YOUR_HOST:5001`** and follow the [directory import steps](RESEARCHER-QUICKSTART.md). Stop the foreground server with **Ctrl-C**; restart with the same settings to reuse its data.

This example has no authentication; use the intended private network. See [server options](SERVER.md) for storage and access configuration, or [install the session UI](SESSION-UI.md).
