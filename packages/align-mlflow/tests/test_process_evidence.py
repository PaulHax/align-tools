"""Inspect process evidence after import, including immutable payload references."""

import copy
import json

import pytest
from align_utils.open_world import OpenWorldRecord
from mlflow.entities import Span
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from open_world_fixtures import SCENARIO, episode_records
from test_generic_evidence import action_log, command, import_traces, read_traces, root

from align_mlflow.cards import card_outputs
from align_mlflow.evidence import read_decision_logs


def activities(process):
    return [activity for phase in process["phases"] for activity in phase["activities"]]


def event(role, direction, body, *, action=0, call=1, tool=None, detail=False):
    fields = (
        f"scenario={SCENARIO} | scene=treat_and_tag | action={action} | call={call}"
    )
    if tool:
        fields += f" | tool_call={tool}"
    return f"=== {role} | {direction}{' | DETAIL' if detail else ''} | {fields} ===\n{body}\n"


def submission(record):
    return "\n\n".join(
        key
        + ":\n"
        + (
            value
            if isinstance(value, str)
            else json.dumps(value, indent=2, ensure_ascii=False)
        )
        for key, value in record["output"]["action"].items()
    )


def correlated_action(
    record, action=0, *, request="messages:\nCaptured system and user context."
):
    return (
        event("DECISION", "INPUT", "messages=2 tools=1 phase=immediate", action=action)
        + event("DECISION", "INPUT", request, action=action, detail=True)
        + event(
            "DECISION",
            "OUTPUT",
            "messages:\nVisible justification and a tool call.",
            action=action,
            detail=True,
        )
        + event(
            "TOOL", "INPUT", "name=check_vitals", action=action, tool=f"tool-{action}"
        )
        + event(
            "TOOL",
            "INPUT",
            '{"name":"check_vitals","arguments":{"character_name":"Patient 1"}}',
            action=action,
            tool=f"tool-{action}",
            detail=True,
        )
        + event(
            "ENVIRONMENT",
            "SUBMIT",
            submission(record),
            action=action,
            tool=f"tool-{action}",
            detail=True,
        )
        + event(
            "ENVIRONMENT",
            "EXECUTED",
            "action=CHECK_VITALS patient=Patient 1 intent=False scenario_complete=False",
            action=action,
            tool=f"tool-{action}",
        )
        + event(
            "TOOL",
            "OUTPUT",
            "Action executed. Updated environment.",
            action=action,
            tool=f"tool-{action}",
            detail=True,
        )
    )


def test_pipeline_process_retains_cache_and_component_evidence_in_persisted_trace(
    tmp_path,
):
    records = episode_records(None)[:2]
    prompt = "Recorded prompt " * 600
    log = action_log(
        records[0],
        "Cache miss for `direct_regression_adm_component` ..\n"
        "[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]\n" + prompt + "\n"
        "[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]\n"
        '{"score":0.9,"reasoning":"Recorded visible explanation."}\n',
    )
    log += action_log(
        records[1], "Cache hit for `direct_regression_adm_component` ..\n"
    )
    _, _, _, _, traces = import_traces(tmp_path, records, log)
    first, second = [root(trace).outputs["evidence"]["process"] for trace in traces]
    assert first["schema_version"] == "decision-process-v1"
    assert first["producer"] == "pipeline"
    regression = next(
        phase
        for phase in first["phases"]
        if phase["label"] == "OWDirectRegressionADMComponent"
    )
    assert any(
        item["kind"] == "cache" and item["status"] == "miss"
        for item in regression["activities"]
    )
    request = next(
        item["inputs"] for item in regression["activities"] if "inputs" in item
    )
    assert request["truncated"] is True
    assert len(request["preview"]) <= 2048
    assert request["source"]["artifact_path"].endswith("/raw_align_system.log")
    assert any(
        item["kind"] == "cache" and item["status"] == "hit"
        for item in activities(second)
    )
    assert not any("inputs" in item for item in activities(second))
    assert root(traces[0]).outputs["evidence"]["log"]["text_truncated"] is True
    assert len(root(traces[0]).outputs["evidence"]["log"]["text"]) <= 4096


def test_langgraph_process_exposes_attempts_and_validation_without_invented_calls(
    tmp_path,
):
    records = episode_records(None)[:1]
    records[0]["choice_info"] = {
        "agent": {
            "observation": "Two visible casualties.",
            "history": ["Checked vitals."],
            "choices": ["Tag a casualty"],
            "proposal_history": [
                {
                    "action_choice": "invalid",
                    "detailed_reasoning": "Recorded explanation.",
                },
                {
                    "action_choice": "valid",
                    "detailed_reasoning": "Recorded revised explanation.",
                },
            ],
            "feedback": ["invalid: action unavailable"],
            "attempts": 2,
        }
    }
    log = action_log(
        records[0], "[bold]*AGENT PROPOSE PROMPT*[/bold]\nRecorded proposal context.\n"
    )
    _, uri, client, eid, traces = import_traces(tmp_path, records, log)
    process = root(traces[0]).outputs["evidence"]["process"]
    assert process["producer"] == "langgraph"
    attempts = [
        item for item in activities(process) if item["id"].startswith("proposal-")
    ]
    assert [item["label"] for item in attempts] == [
        "Proposal attempt 1",
        "Proposal attempt 2",
    ]
    assert (
        attempts[1]["outputs"]["source"]["json_pointer"]
        == "/choice_info/agent/proposal_history/1"
    )
    assert not any(item["kind"] == "model" for item in attempts)
    assert any(
        item["kind"] == "validation"
        and "action unavailable" in item["outputs"]["preview"]
        for item in activities(process)
    )
    assert any("total model-call count" in gap for gap in process["gaps"])
    before = root(traces[0]).outputs
    refreshed = command(uri, "refresh-cards")
    assert refreshed.returncode == 0, refreshed.stderr
    assert root(read_traces(client, eid)[0]).outputs == before


def test_legacy_agent_tool_sections_are_visible_but_uncaptured_results_stay_missing(
    tmp_path,
):
    records = episode_records(None)[:1]
    records[0]["choice_info"] = {
        "langchain_agent": {
            "justification": "Recorded rationale",
            "n_actions_taken_in_scenario": 0,
        }
    }
    log = action_log(
        records[0],
        "[bold]*AGENT SYSTEM PROMPT*[/bold]\nRecorded system instructions.\n"
        '[bold]*AGENT TOOL CALL*[/bold]: check_vitals({"character_name":"Patient 1"})\n',
    )
    _, _, _, _, traces = import_traces(tmp_path, records, log)
    process = root(traces[0]).outputs["evidence"]["process"]
    assert process["producer"] == "langchain"
    logged = next(phase for phase in process["phases"] if phase["id"] == "logged")
    assert [item["label"] for item in logged["activities"]] == [
        "AGENT SYSTEM PROMPT",
        "AGENT TOOL CALL",
    ]
    assert "check_vitals" in logged["activities"][1]["outputs"]["preview"]
    assert any("not comprehensively captured" in gap for gap in process["gaps"])


def test_correlated_import_pairs_exact_payloads_and_keeps_large_log_in_one_artifact(
    tmp_path,
):
    records = episode_records(None)[:1]
    records[0]["choice_info"] = {
        "langchain_agent": {"justification": "Recorded rationale"},
        "multi_agent": {
            "proposal": records[0]["output"]["action"],
            "advice": {"recommendation": "keep"},
            "review_context": {"observation": "Recorded reviewer context"},
            "final_action": records[0]["output"]["action"],
            "review_status": "reviewed",
            "weighting": {
                "configuration": {"influence": 0.5, "swap_threshold": 0.15},
                "strength": 0.0,
                "suggestion": "keep",
                "followed_guidance": True,
            },
        },
    }
    request = (
        "messages:\nCaptured é system context.\n" + "Long captured context. " * 1000
    )
    log = correlated_action(records[0], request=request)
    _, uri, client, eid, traces = import_traces(tmp_path, records, log)
    output = root(traces[0]).outputs
    assert output["evidence"]["log"]["status"] == "exact_correlated_action_sequence"
    process = output["evidence"]["process"]
    assert process["producer"] == "reviewer"
    model = next(item for item in activities(process) if item["kind"] == "model")
    assert model["status"] == "captured"
    payload = model["inputs"]
    assert payload["truncated"] is True
    assert len(payload["preview"]) == 2048
    source = payload["source"]
    start, end = source["byte_range"]
    assert log.encode()[start:end].decode() == request + "\n"
    assert (
        "\n".join(log.splitlines()[source["lines"][0] - 1 : source["lines"][1]])
        == request
    )
    run = client.search_runs([eid])[0]
    artifact = (
        tmp_path / "artifacts" / run.info.run_id / "artifacts" / source["artifact_path"]
    )
    assert artifact.read_text() == log
    assert len(json.dumps(output)) < len(log)
    tool = next(item for item in activities(process) if item["kind"] == "tool")
    assert "arguments" in tool["inputs"]["preview"]
    assert "Updated environment" in tool["outputs"]["preview"]
    assert process["review"]["followed_guidance"] is True
    assert process["review"]["swap_threshold"] == 0.15
    refreshed = command(uri, "refresh-cards")
    assert refreshed.returncode == 0, refreshed.stderr
    assert root(read_traces(client, eid)[0]).outputs == output


@pytest.mark.parametrize(
    "damage",
    [
        "mismatch",
        "missing_submit",
        "duplicate_execution",
        "wrong_occurrence",
        "partial",
    ],
)
def test_uncertain_correlated_actions_do_not_shift_evidence(tmp_path, damage):
    records = episode_records(None)[:2]
    first = correlated_action(records[0])
    next_record = copy.deepcopy(records[1])
    if damage == "mismatch":
        next_record["output"]["action"]["character_id"] = "Different casualty"
    suffix = correlated_action(
        next_record, action=2 if damage == "wrong_occurrence" else 1
    )
    if damage == "missing_submit":
        submit = event(
            "ENVIRONMENT",
            "SUBMIT",
            submission(next_record),
            action=1,
            tool="tool-1",
            detail=True,
        )
        suffix = suffix.replace(submit, "")
    elif damage == "duplicate_execution":
        suffix += event(
            "ENVIRONMENT",
            "EXECUTED",
            "Duplicate execution summary.",
            action=1,
            tool="tool-1",
        )
    elif damage == "partial":
        suffix = suffix[: suffix.index("=== ENVIRONMENT | EXECUTED")]
    models = [OpenWorldRecord.model_validate(record) for record in records]
    logs = read_decision_logs(tmp_path, models, log_text=first + suffix)
    assert logs[0].status == "exact_correlated_action_sequence"
    assert not logs[1].events
    assert logs[1].status == (
        "action_not_yet_logged"
        if damage == "partial"
        else "unverified_correlated_action_at_record_1"
    )
    process = card_outputs(models[1], record_index=1, log=logs[1])["evidence"][
        "process"
    ]
    assert not any(item["kind"] == "model" for item in activities(process))
    assert any("association unavailable" in gap for gap in process["gaps"])


def test_failed_correlated_attempt_remains_distinct_from_executed_action(tmp_path):
    record = episode_records(None)[0]
    model = OpenWorldRecord.model_validate(record)
    failed = event(
        "ENVIRONMENT", "SUBMIT", submission(record), tool="failed", detail=True
    )
    failed += event("ENVIRONMENT", "REJECTED", "status=400", tool="failed")
    log = read_decision_logs(
        tmp_path, [model], log_text=failed + correlated_action(record)
    )[0]
    assert log.status == "exact_correlated_action_sequence"
    process = card_outputs(model, log=log)["evidence"]["process"]
    failed_activity = next(
        item for item in activities(process) if item.get("status") == "failed"
    )
    assert "rejected" in failed_activity["label"]
    assert any(item.get("status") == "executed" for item in activities(process))


def test_repeated_case_episodes_reset_action_occurrence_without_merging_calls(tmp_path):
    records = episode_records(None)[:2]
    records[0]["input"]["full_state"]["elapsed_time"] = 10
    records[1]["input"]["full_state"]["elapsed_time"] = 0
    models = [OpenWorldRecord.model_validate(record) for record in records]
    first = correlated_action(records[0], request="messages:\nFirst episode request.")
    second = event(
        "DECISION", "CONTEXT BUDGET", "Second episode budget."
    ) + correlated_action(records[1], request="messages:\nSecond episode request.")
    logs = read_decision_logs(tmp_path, models, log_text=first + second)
    assert [log.status for log in logs] == ["exact_correlated_action_sequence"] * 2
    processes = [
        card_outputs(model, log=log)["evidence"]["process"]
        for model, log in zip(models, logs)
    ]
    requests = [
        next(
            item["inputs"]["preview"]
            for item in activities(process)
            if item["kind"] == "model"
        )
        for process in processes
    ]
    assert requests == [
        "messages:\nFirst episode request.\n",
        "messages:\nSecond episode request.\n",
    ]
    assert "Second episode budget" not in json.dumps(processes[0])
    assert any(
        "Second episode budget" in item.get("outputs", {}).get("preview", "")
        for item in activities(processes[1])
    )


def test_unrecorded_action_extensions_do_not_become_an_exact_action_match(tmp_path):
    source = episode_records(None)[0]
    log_text = correlated_action(source)
    source["output"]["action"]["new_execution_parameter"] = "unlogged value"
    logs = read_decision_logs(
        tmp_path, [OpenWorldRecord.model_validate(source)], log_text=log_text
    )
    assert logs[0].status == "unverified_correlated_action_at_record_0"
    assert not logs[0].events


def test_malformed_correlated_header_does_not_hide_a_rejected_attempt(tmp_path):
    source = episode_records(None)[0]
    rejected = event("ENVIRONMENT", "REJECTED", "status=400", tool="failed")
    rejected = rejected.replace("tool_call=failed ===", "tool_call=failed | call=2 ===")
    log = rejected + correlated_action(source)
    parsed = read_decision_logs(
        tmp_path, [OpenWorldRecord.model_validate(source)], log_text=log
    )[0]
    assert parsed.status == "unverified_correlated_action_at_record_0"
    assert not parsed.events


def test_missing_call_identity_retains_each_unresolved_payload_without_pairing(
    tmp_path,
):
    source = episode_records(None)[0]
    source["choice_info"] = {"langchain_agent": {}}
    unresolved = "".join(
        event("TOOL", direction, text, detail=True)
        for direction, text in (
            ("INPUT", "First arguments"),
            ("OUTPUT", "First result"),
            ("INPUT", "Second arguments"),
            ("OUTPUT", "Second result"),
        )
    )
    original = correlated_action(source)
    original = "\n".join(
        line.replace(" | call=1", "") if line.startswith("=== DECISION") else line
        for line in original.split("\n")
    )
    model = OpenWorldRecord.model_validate(source)
    parsed = read_decision_logs(tmp_path, [model], log_text=unresolved + original)[0]
    assert parsed.status == "exact_correlated_action_sequence"
    process = card_outputs(model, log=parsed)["evidence"]["process"]
    uncorrelated = [
        item for item in activities(process) if item["label"].startswith("Uncorrelated")
    ]
    captured = [
        item[key]["preview"].strip()
        for item in uncorrelated
        for key in ("inputs", "outputs")
        if key in item
    ]
    assert captured[:4] == [
        "First arguments",
        "First result",
        "Second arguments",
        "Second result",
    ]
    assert not any("inputs" in item and "outputs" in item for item in uncorrelated)
    assert any("pairing is unresolved" in gap for gap in process["gaps"])


def test_unowned_pipeline_scores_are_labelled_as_final_context():
    source = episode_records(None)[0]
    source["choice_info"]["per_step_timing_stats"].append(
        {
            "step": "custom.ComparativeRegressionADMComponent",
            "step_num": 3,
            "elapsed_s": 1.0,
        }
    )
    process = card_outputs(OpenWorldRecord.model_validate(source))["evidence"][
        "process"
    ]
    snapshots = [
        item
        for item in activities(process)
        if item["label"].startswith("Final pipeline snapshot: predicted kdma values")
    ]
    assert len(snapshots) == 2
    assert all(item["kind"] == "context" for item in snapshots)


def test_unowned_log_sections_keep_source_evidence_without_inventing_calls(tmp_path):
    source = episode_records(None)[0]
    model = OpenWorldRecord.model_validate(source)
    text = action_log(
        source,
        "[bold]*UNFAMILIAR PROMPT*[/bold]\nCaptured unknown-component context.\n"
        "[bold]*UNFAMILIAR RESULT*[/bold]\nCaptured unknown-component result.\n",
    )
    log = read_decision_logs(tmp_path, [model], log_text=text)[0]
    process = card_outputs(model, log=log)["evidence"]["process"]
    unowned = next(phase for phase in process["phases"] if phase["id"] == "unassigned")
    assert "ownership is unknown" in unowned["summary"]
    assert [item["kind"] for item in unowned["activities"]] == ["recorded", "recorded"]
    payload = unowned["activities"][0]["inputs"]
    start, end = payload["source"]["byte_range"]
    assert "Captured unknown-component context." in text.encode()[start:end].decode()
    assert any("without call attribution" in gap for gap in process["gaps"])


def test_refresh_from_verified_original_source_enriches_older_trace_without_changing_identity(
    tmp_path,
):
    records = episode_records(None)[:1]
    records[0]["choice_info"] = {
        "langchain_agent": {"justification": "Recorded rationale"}
    }
    log = correlated_action(records[0], request="messages:\nOriginal é request.")
    path, uri, client, eid, traces = import_traces(tmp_path, records, log)
    imported = traces[0]
    old = copy.deepcopy(root(imported).to_dict())
    outputs = copy.deepcopy(root(imported).outputs)
    outputs["evidence"].pop("process")
    outputs["evidence"]["log"] = {
        "status": "no_action_markers",
        "source": {"file": "raw_align_system.log"},
    }
    old["attributes"][SpanAttributeKey.OUTPUTS] = json.dumps(outputs)
    TracingClient(tracking_uri=uri).log_spans(eid, [Span.from_dict(old)])
    before = read_traces(client, eid)[0]
    preview = command(uri, "refresh-cards", "--source", str(path), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert read_traces(client, eid)[0].to_dict() == before.to_dict()
    refreshed = command(uri, "refresh-cards", "--source", str(path))
    assert refreshed.returncode == 0, refreshed.stderr
    after = read_traces(client, eid)[0]
    assert after.info.to_dict() == before.info.to_dict()
    assert [
        (span.span_id, span.parent_id, span.start_time_ns, span.end_time_ns)
        for span in after.data.spans
    ] == [
        (span.span_id, span.parent_id, span.start_time_ns, span.end_time_ns)
        for span in before.data.spans
    ]
    process = root(after).outputs["evidence"]["process"]
    request = next(
        item["inputs"] for item in activities(process) if item["kind"] == "model"
    )
    start, end = request["source"]["byte_range"]
    assert log.encode()[start:end].decode() == "messages:\nOriginal é request.\n"
    assert root(after).outputs["evidence"]["record_index"] == 0
    repeated = command(uri, "refresh-cards", "--source", str(path))
    assert repeated.returncode == 0, repeated.stderr
    assert "0 cards refreshed, 1 already current" in repeated.stdout
    assert read_traces(client, eid)[0].to_dict() == after.to_dict()


def test_source_refresh_rejects_changed_original_json_before_modifying_trace(tmp_path):
    records = episode_records(None)[:1]
    path, uri, client, eid, traces = import_traces(
        tmp_path, records, action_log(records[0])
    )
    before = traces[0].to_dict()
    changed = copy.deepcopy(records)
    changed[0]["output"]["action"]["character_id"] = "Different casualty"
    (path / "input_output.json").write_text(json.dumps(changed))
    result = command(uri, "refresh-cards", "--source", str(path))
    assert result.returncode != 0
    assert "source record differs" in result.stderr
    assert read_traces(client, eid)[0].to_dict() == before


@pytest.mark.parametrize(
    "case", ["recorded_identity", "ambiguous_identity", "changed_provenance"]
)
def test_source_refresh_uses_verified_recorded_identity_when_display_labels_change(
    tmp_path, case
):
    records = episode_records(None)[:1]
    path, uri, client, eid, traces = import_traces(
        tmp_path, records, correlated_action(records[0])
    )
    source = client.search_runs([eid])[0]
    historical_key = "historical-import-key"
    client.set_tag(source.info.run_id, "align.run_key", historical_key)
    client.set_trace_tag(traces[0].info.trace_id, "align.run_key", historical_key)
    old = root(traces[0]).to_dict()
    outputs = copy.deepcopy(root(traces[0]).outputs)
    outputs["evidence"].pop("process")
    old["attributes"][SpanAttributeKey.OUTPUTS] = json.dumps(outputs)
    TracingClient(tracking_uri=uri).log_spans(eid, [Span.from_dict(old)])
    if case == "ambiguous_identity":
        client.create_run(
            eid,
            tags={"align.run_key": "another-import", "align.source_path": str(path)},
        )
    elif case == "changed_provenance":
        config = path / ".hydra/config.yaml"
        config.write_text(config.read_text() + "\nnew_setting: changed\n")
    before = read_traces(client, eid)[0]
    result = command(uri, "refresh-cards", "--source", str(path))
    if case == "recorded_identity":
        assert result.returncode == 0, result.stderr
        after = read_traces(client, eid)[0]
        assert after.info.to_dict() == before.info.to_dict()
        process = root(after).outputs["evidence"]["process"]
        assert any(
            item["kind"] == "model" and "byte_range" in item["inputs"]["source"]
            for item in activities(process)
        )
        assert after.info.tags["align.run_key"] == historical_key
    else:
        assert result.returncode != 0
        assert (
            "Ambiguous imported source identity"
            if case == "ambiguous_identity"
            else "Source provenance differs"
        ) in result.stderr
        assert read_traces(client, eid)[0].to_dict() == before.to_dict()
