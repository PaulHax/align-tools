"""Install the example views using MLflow 3.16.1's internal saved-view format.

This is an optional UI recipe, independent of ingestion. It updates only the
named views in views.json. The server version must match the verified format.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from mlflow import MlflowClient

PREFIX = "mlflow.tracesV4ViewState."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--ui-url", required=True)
    parser.add_argument("--experiment", default="align-system open world")
    args = parser.parse_args()
    ui_url = args.ui_url.rstrip("/")
    with urlopen(f"{ui_url}/version", timeout=10) as response:
        version = response.read().decode().strip()
    if version != "3.16.1":
        parser.error(
            f"Server is MLflow {version}; this recipe is verified for 3.16.1. "
            "Use REFERENCE.md's manual column/filter recipe for other versions."
        )

    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        parser.error(f"Experiment {args.experiment!r} does not exist; import first")
    saved = {}
    for key, value in experiment.tags.items():
        if key.startswith(PREFIX):
            saved[key] = json.loads(value)

    views = json.loads(Path(__file__).with_name("views.json").read_text())
    for view in views:
        matches = [key for key, value in saved.items() if value["name"] == view["name"]]
        if len(matches) > 1:
            parser.error(f"Multiple views named {view['name']!r}; rename duplicates")
        key = matches[0] if matches else PREFIX + view["id"]
        previous = saved.get(key, {})
        state = json.dumps(view["state"], separators=(",", ":"))
        if previous.get("state") != state:
            now = int(time.time() * 1000)
            client.set_experiment_tag(
                experiment.experiment_id,
                key,
                json.dumps(
                    {
                        "name": view["name"],
                        "createdAt": previous.get("createdAt", now),
                        "updatedAt": now,
                        "state": state,
                    }
                ),
            )
        query = urlencode(
            {"startTimeLabel": "ALL", "traceViewShareKey": key.removeprefix(PREFIX)}
        )
        print(
            f"{view['name']}: "
            f"{ui_url}/#/experiments/{experiment.experiment_id}/traces?{query}"
        )


if __name__ == "__main__":
    main()
