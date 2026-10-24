"""Install the example views using MLflow 3.17.2's internal saved-view format.

This is an optional UI recipe, independent of ingestion. It updates only the
named presets and, with --by-adm, comparison views for recorded ADM labels.
The server version must match the verified format.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

from mlflow import MlflowClient
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import http_request

PREFIX = "mlflow.tracesV4ViewState."


def adm_views(client: MlflowClient, experiment_id: str, comparison: dict) -> list[dict]:
    labels: set[str] = set()
    token = None
    while True:
        page = client.search_runs([experiment_id], max_results=1000, page_token=token)
        labels.update(label for run in page if (label := run.data.tags.get("adm")))
        token = page.token
        if not token:
            break
    return [
        {
            "id": "open-world-adm-" + hashlib.sha256(label.encode()).hexdigest()[:12],
            "name": f"ADM: {label}",
            "state": {
                **comparison,
                "single": {**comparison["single"], "startTimeLabel": "ALL"},
                "filters": [
                    {"field": "tag", "key": "adm", "operator": "=", "value": label}
                ],
            },
        }
        for label in sorted(labels)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--ui-url", required=True)
    parser.add_argument("--experiment", default="align-system open world")
    parser.add_argument(
        "--by-adm",
        action="store_true",
        help="Also create a Session comparison view for each recorded run tag 'adm'",
    )
    args = parser.parse_args()
    ui_url = args.ui_url.rstrip("/")
    response = http_request(get_default_host_creds(ui_url), "/version", "GET")
    response.raise_for_status()
    version = response.text.strip()
    if version != "3.17.2":
        parser.error(
            f"Server is MLflow {version}; this recipe is verified for 3.17.2. "
            "Configure columns and filters manually in the UI for other versions."
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
    if args.by_adm:
        comparison = next(
            view["state"]
            for view in views
            if view["id"] == "open-world-session-comparison"
        )
        views.extend(adm_views(client, experiment.experiment_id, comparison))
    destinations = []
    for view in views:
        matches = [key for key, value in saved.items() if value["name"] == view["name"]]
        if len(matches) > 1:
            parser.error(f"Multiple views named {view['name']!r}; rename duplicates")
        key = matches[0] if matches else PREFIX + view["id"]
        if key in saved and saved[key]["name"] != view["name"]:
            parser.error(
                f"View ID {view['id']!r} belongs to {saved[key]['name']!r}; "
                f"rename it to {view['name']!r} to restore this preset, or copy "
                "the custom view and delete the conflicting original. No views changed."
            )
        destinations.append((view, key))

    for view, key in destinations:
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
            {
                **{
                    name: value
                    for name, value in view["state"]["single"].items()
                    if name != "cols"
                },
                **view["state"].get("multi", {}),
                "traceViewShareKey": key.removeprefix(PREFIX),
            },
            doseq=True,
        )
        print(
            f"{view['name']}: "
            f"{ui_url}/#/experiments/{experiment.experiment_id}/traces?{query}"
        )


if __name__ == "__main__":
    main()
