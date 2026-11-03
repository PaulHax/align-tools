"""Inspect compact proximity and clinical changes in persisted decision cards."""

import copy
import json

from mlflow.entities import Span
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from open_world_fixtures import episode_records
from test_generic_evidence import command, import_traces, read_traces, root


def test_import_and_refresh_group_proximity_without_repeating_unchanged_patients(
    tmp_path,
):
    records = episode_records(None)[:4]
    patients = [
        {
            "id": "Patient 1",
            "unstructured": "Patient 1 has a visible injury.",
            "nearby": False,
            "vitals": {},
        },
        {
            "id": "Patient 2",
            "unstructured": "Patient 2 has another visible injury.",
            "nearby": True,
            "vitals": {},
        },
        {
            "id": "Alpha",
            "name": "Casey",
            "unstructured": "Alpha is awaiting assessment.",
            "nearby": None,
        },
        {"id": "Bravo", "unstructured": "Bravo is awaiting assessment."},
        {
            "id": "Charlie",
            "unstructured": "Charlie is awaiting assessment.",
            "nearby": "false",
        },
        {
            "id": "Unseen patient",
            "unstructured": "Hidden description.",
            "nearby": True,
            "unseen": True,
        },
    ]
    for index, record in enumerate(records):
        state = record["input"]["full_state"]
        record["input"]["state"] = state["unstructured"] = (
            "Several casualties await assessment."
        )
        state["characters"] = copy.deepcopy(patients)
        if index:
            state["characters"][0]["nearby"] = True
            state["characters"][1]["nearby"] = False
        if index >= 2:
            state["characters"][0]["vitals"] = {"heart_rate": "FAST"}
            state["characters"][0]["tag"] = "IMMEDIATE"
            state["characters"][1]["vitals"] = {"AVPU": "ALERT"}
        if index == 3:
            state["characters"][0]["unstructured"] = (
                "Patient 1's visible bleeding is controlled.\nRecheck capillary refill."
            )
    _, uri, client, eid, traces = import_traces(
        tmp_path, records, "No action details recorded.\n"
    )
    summaries = [root(trace).inputs["input"] for trace in traces]
    first_roster = "**Now nearby:** Patient 2 · **Not nearby:** Patient 1 · **Unknown proximity:** Alpha (Casey), Bravo, Charlie"
    moved_roster = "**Now nearby:** Patient 1 · **Not nearby:** Patient 2 · **Unknown proximity:** Alpha (Casey), Bravo, Charlie"
    assert first_roster in summaries[0]
    assert "Patient 1 has a visible injury." in summaries[0]
    assert "Patient 2 has another visible injury." in summaries[0]
    for summary in summaries[1:]:
        assert moved_roster in summary
        assert "Nearby:" not in summary.replace(moved_roster, "")
        assert "Patient 1 has a visible injury." not in summary
        assert "Patient 2 has another visible injury." not in summary
        assert "Unseen patient" not in summary
        assert "Hidden description." not in summary
    assert summaries[1] == "**Situation**\n\nUnchanged.\n\n" + moved_roster
    assert "Patient updates since previous input" not in summaries[1]
    assert (
        "- **Patient 1:** Vitals / Heart rate: not recorded → FAST; Triage tag: not recorded → IMMEDIATE\n- **Patient 2:** Vitals / AVPU: not recorded → ALERT"
        in summaries[2]
    )
    assert (
        "- **Patient 1:** Patient 1's visible bleeding is controlled. Recheck capillary refill."
        in summaries[3]
    )
    assert "Heart rate" not in summaries[3]
    assert "Triage tag" not in summaries[3]

    changed = copy.deepcopy(root(traces[1]).to_dict())
    changed["attributes"][SpanAttributeKey.INPUTS] = json.dumps(
        {**root(traces[1]).inputs, "input": "**Patient 1**\n\n- Nearby: false → true"}
    )
    TracingClient(tracking_uri=uri).log_spans(eid, [Span.from_dict(changed)])
    before = read_traces(client, eid)
    result = command(uri, "refresh-cards")
    assert result.returncode == 0, result.stderr
    after = read_traces(client, eid)
    assert [root(trace).inputs["input"] for trace in after] == summaries
    assert [trace.info.to_dict() for trace in after] == [
        trace.info.to_dict() for trace in before
    ]
    assert [root(trace).outputs for trace in after] == [
        root(trace).outputs for trace in before
    ]
    assert [
        {"input": root(trace).inputs["source"], **root(trace).outputs["source"]}
        for trace in after
    ] == records
    repeated = command(uri, "refresh-cards")
    assert repeated.returncode == 0, repeated.stderr
    assert "0 cards refreshed, 4 already current" in repeated.stdout
