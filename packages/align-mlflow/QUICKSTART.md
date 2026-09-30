# Start a shared MLflow server

This starts a fresh store; it does not migrate existing data.

1. **Install the server and session UI** (requires Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`):

   ```bash
   git clone --branch open-world-traces https://github.com/PaulHax/align-tools.git
   cd align-tools
   ./packages/align-mlflow/scripts/setup.sh
   ```

2. **Choose your configuration.** Copy the [example](examples/server.env.example) to `.env` and edit it as needed. It uses `/data/shared/mlflow` on ITM and port **5001**:

   ```bash
   cp packages/align-mlflow/examples/server.env.example .env
   ```

   The server account needs write access to the chosen directory. On another host, update both host and browser-origin allowlists.

3. **Start the server:**

   ```bash
   uv run --no-sync mlflow --env-file .env server
   ```

   Open <http://10.50.57.47:5001> for the example configuration and follow the [directory import steps](RESEARCHER-QUICKSTART.md). Stop with **Ctrl-C**; restart with the same command to reuse its data. To select another configuration file, change `--env-file`.

This example has no authentication; use the intended private network. See [server options](SERVER.md) for storage and access configuration.
