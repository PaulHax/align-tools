"""Inspect generic saved evidence through the public trace outputs."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
from align_utils.open_world import OpenWorldRecord
from mlflow import MlflowClient
from mlflow.entities import Span
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from open_world_fixtures import episode_records, write_run

from align_mlflow.cards import card_outputs
from align_mlflow.evidence import read_decision_logs

EXPERIMENT = "generic evidence"
TARGET = "ADEPT-June2025-merit-0.4"
MARKER = "[bold]*ACTION BEING TAKEN*[/bold]\n"


def command(uri, *arguments):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "align_mlflow",
            *arguments,
            "--tracking-uri",
            uri,
            "--experiment",
            EXPERIMENT,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )


def import_traces(tmp_path, records, log_text):
    path = write_run(tmp_path / "2026-08-31__09-17-11", records)
    if log_text is None:
        (path / "raw_align_system.log").unlink()
    else:
        (path / "raw_align_system.log").write_text(log_text)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    result = command(uri, "sync", str(path))
    assert result.returncode == 0, result.stderr
    traces = read_traces(client, experiment_id)
    assert len(traces) == len(records)
    assert [
        {"input": root(trace).inputs["source"], **root(trace).outputs["source"]}
        for trace in traces
    ] == records
    return path, uri, client, experiment_id, traces


def read_traces(client, experiment_id):
    return sorted(
        client.search_traces(locations=[experiment_id]),
        key=lambda trace: int(trace.info.tags["align.record_index"]),
    )


def root(trace):
    return next(span for span in trace.data.spans if span.parent_id is None)


def action_log(record, text=""):
    return text + MARKER + json.dumps(record["output"]["action"], indent=2) + "\n"


def rejection_block(status=400):
    reason = "BAD REQUEST" if status == 400 else "INTERNAL SERVER ERROR"
    return (
        f"({status})\nReason: {reason}\n"
        "HTTP response headers: HTTPHeaderDict({})\n"
        'HTTP response body: "rejected"\n\n\n'
    )


def test_generic_evidence_and_plain_logs_remain_visible_after_refresh(tmp_path):
    original = episode_records(TARGET)
    records = [original[0], original[1], original[5]]
    unknown = {
        "unfamiliar": {"nested": [{"value": None}, False, "é"]},
        "list": [1, {"unfamiliar": [None, "raw"]}],
        "nothing": None,
        "flag": False,
        "count": 7,
        "weight": 0.5,
        "text": "Recorded context " * 100,
    }
    records[0]["choice_info"] = unknown
    records[2]["choice_info"] = copy.deepcopy(unknown)
    fragments = [
        "Unfamiliar log syntax: {'raw': [None, False]}\n",
        "Cache miss for `direct_regression_adm_component` ..\n"
        "[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]\n"
        "Preserve the measured pipeline prompt.\n"
        "[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]\n"
        '{"score": 97, "reasoning": "Preserve the measured pipeline response."}\n',
        "Driver selected a scene transition.\n",
    ]
    log_text = "".join(
        action_log(record, fragment) for record, fragment in zip(records, fragments)
    )
    _, uri, client, eid, traces = import_traces(tmp_path, records, log_text)
    first, pipeline, driver = map(root, traces)

    evidence = first.outputs["evidence"]
    assert evidence["record_index"] == 0
    info = evidence["choice_info"]
    assert info["source"] == "source.choice_info"
    assert info["present"] is True
    assert info["applies_to_action"] is True
    assert set(info["fields"]) == set(unknown)
    assert {name: value["type"] for name, value in info["fields"].items()} == {
        "unfamiliar": "object",
        "list": "array",
        "nothing": "null",
        "flag": "boolean",
        "count": "number",
        "weight": "number",
        "text": "string",
    }
    assert info["fields"]["unfamiliar"]["preview"] == json.dumps(
        unknown["unfamiliar"], ensure_ascii=False
    )
    assert info["fields"]["list"]["items"] == 2
    assert info["fields"]["text"]["preview_truncated"] is True
    assert len(info["fields"]["text"]["preview"]) <= 256
    assert unknown["text"] == first.outputs["source"]["choice_info"]["text"]
    assert len(traces[0].data.spans) == len(traces[2].data.spans) == 1
    assert driver.outputs["evidence"]["choice_info"]["applies_to_action"] is False
    assert driver.get_attribute("align.choice_info_applies_to_action") is False

    digest = hashlib.sha256(log_text.encode()).hexdigest()
    for span, fragment in zip((first, pipeline, driver), fragments):
        log = span.outputs["evidence"]["log"]
        assert log["status"] == "exact_action_sequence"
        assert log["text"] == fragment.rstrip("\n")
        source = log["source"]
        assert source["file"] == "raw_align_system.log"
        assert source["snapshot_sha256"] == digest
        assert source["sha256"] == hashlib.sha256(log["text"].encode()).hexdigest()
        assert (
            "\n".join(
                log_text.splitlines()[source["lines"][0] - 1 : source["lines"][1]]
            )
            == log["text"]
        )
    regression = next(
        span
        for span in traces[1].data.spans
        if span.name == "OWDirectRegressionADMComponent"
    )
    assert "Preserve the measured pipeline prompt." in regression.inputs["input"]
    assert "Preserve the measured pipeline response." in regression.outputs["response"]

    before = [copy.deepcopy(span.outputs) for span in (first, pipeline, driver)]
    altered = copy.deepcopy(first.to_dict())
    altered["attributes"][SpanAttributeKey.OUTPUTS] = json.dumps(
        {**first.outputs, "response": "Older action summary"}
    )
    TracingClient(tracking_uri=uri).log_spans(eid, [Span.from_dict(altered)])
    refreshed = command(uri, "refresh-cards")
    assert refreshed.returncode == 0, refreshed.stderr
    assert [root(trace).outputs for trace in read_traces(client, eid)] == before
    repeated = command(uri, "refresh-cards")
    assert repeated.returncode == 0, repeated.stderr
    assert "0 cards refreshed, 3 already current" in repeated.stdout


@pytest.mark.parametrize(
    "case", ["missing", "truncated", "mismatch", "malformed", "repeated"]
)
def test_optional_uncertain_logs_do_not_block_records_or_shift_evidence(tmp_path, case):
    records = episode_records(TARGET)[:3]
    prefix = action_log(records[0], "First verified evidence.\n")
    if case == "missing":
        log_text = None
    elif case == "truncated":
        log_text = prefix + "Unfinished evidence.\n" + MARKER + '{"action_id": '
    elif case == "malformed":
        log_text = prefix + "Unassigned malformed evidence.\n" + MARKER + "invalid\n"
        log_text += action_log(records[2], "Later evidence must remain unassigned.\n")
    else:
        unexpected = copy.deepcopy(records[1])
        unexpected["output"]["action"]["character_id"] = "Unexpected patient"
        log_text = prefix + action_log(unexpected, "Unassigned rejected evidence.\n")
        if case == "repeated":
            log_text += action_log(
                records[0], "Repeated action must not recover attribution.\n"
            )
        log_text += action_log(records[1], "Uncertain later evidence.\n")
        log_text += action_log(records[2], "Later evidence must remain unassigned.\n")
    _, _, _, _, traces = import_traces(tmp_path, records, log_text)
    logs = [root(trace).outputs["evidence"]["log"] for trace in traces]
    if case == "missing":
        assert all(log["status"] == "missing_log" for log in logs)
    else:
        assert logs[0]["status"] == "exact_action_sequence"
        assert logs[0]["text"] == "First verified evidence."
        expected = {
            "truncated": "action_not_yet_logged",
            "malformed": "invalid_action_at_record_1",
            "mismatch": "action_mismatch_at_record_1",
            "repeated": "action_mismatch_at_record_1",
        }[case]
        assert all(log["status"] == expected for log in logs[1:])
    for trace, log in zip(
        traces[1:] if case != "missing" else traces,
        logs[1:] if case != "missing" else logs,
    ):
        assert "text" not in log
        assert "lines" not in log["source"]
        for span in trace.data.spans:
            if span.parent_id is not None:
                assert not any(
                    source["file"] == "raw_align_system.log"
                    for source in span.get_attribute("align.evidence_sources")
                )


def test_explicit_log_snapshot_controls_visible_evidence(tmp_path):
    source = episode_records(TARGET)[:1]
    records = tuple(OpenWorldRecord.model_validate(record) for record in source)
    snapshot = action_log(source[0], "Captured evidence.\n")
    (tmp_path / "raw_align_system.log").write_text("Changed after capture.\n")

    log = read_decision_logs(tmp_path, records, log_text=snapshot)[0]
    outputs = card_outputs(records[0], log=log)
    assert outputs["evidence"]["log"]["text"] == "Captured evidence."
    assert (
        outputs["evidence"]["log"]["source"]["snapshot_sha256"]
        == hashlib.sha256(snapshot.encode()).hexdigest()
    )
    missing = read_decision_logs(tmp_path, records, log_text=None)[0]
    assert card_outputs(records[0], log=missing)["evidence"]["log"] == {
        "status": "missing_log",
        "source": {"file": "raw_align_system.log"},
    }


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("annotation", ["Recorded é.".encode(), b"Recorded \xff."])
def test_original_log_bytes_identify_readable_excerpts(tmp_path, newline, annotation):
    source = episode_records(TARGET)[:1]
    records = tuple(OpenWorldRecord.model_validate(record) for record in source)
    raw_log = (
        b"First recorded line."
        + newline
        + annotation
        + newline
        + action_log(source[0]).encode().replace(b"\n", newline)
    )
    log_path = tmp_path / "raw_align_system.log"
    log_path.write_bytes(raw_log)
    digest = hashlib.sha256(raw_log).hexdigest()
    from_file = card_outputs(records[0], log=read_decision_logs(tmp_path, records)[0])[
        "evidence"
    ]["log"]
    assert from_file["status"] == "exact_action_sequence"
    assert from_file["text"] == (
        "First recorded line." + newline.decode() + annotation.decode(errors="replace")
    )
    assert from_file["source"]["lines"] == [1, 2]
    assert from_file["source"]["snapshot_sha256"] == digest
    assert (
        from_file["source"]["sha256"]
        == hashlib.sha256(from_file["text"].encode()).hexdigest()
    )

    log_path.write_bytes(b"Changed after capture.\n")
    from_snapshot = card_outputs(
        records[0],
        log=read_decision_logs(
            tmp_path,
            records,
            log_text=raw_log.decode(errors="replace"),
            snapshot_sha256=digest,
        )[0],
    )["evidence"]["log"]
    assert from_snapshot == from_file


@pytest.mark.parametrize("status", [400, 500])
@pytest.mark.parametrize("same_action", [False, True])
def test_rejected_attempts_do_not_supply_evidence_for_saved_actions(
    tmp_path, status, same_action
):
    records = episode_records(TARGET)[:3]
    rejected = copy.deepcopy(records[1])
    if not same_action:
        rejected["output"]["action"]["character_id"] = "Rejected candidate"
    failed_text = (
        "Cache miss for `direct_regression_adm_component` ..\n"
        "[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]\n"
        "Failed proposal evidence.\n"
        "[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]\n"
        '{"reasoning": "Failed proposal evidence."}\n'
    )
    fresh_text = (
        "Cache miss for `direct_regression_adm_component` ..\n"
        "[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]\n"
        "Fresh retry evidence.\n"
        "[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]\n"
        '{"reasoning": "Fresh retry evidence."}\n'
    )
    log_text = (
        action_log(records[0], "Verified first evidence.\n")
        + action_log(rejected, failed_text)
        + rejection_block(status)
        + action_log(records[1], fresh_text)
        + action_log(records[2], "Following saved action.\n")
    )
    _, _, _, _, traces = import_traces(tmp_path, records, log_text)
    logs = [root(trace).outputs["evidence"]["log"] for trace in traces]
    assert all(log["status"] == "exact_action_sequence" for log in logs)
    assert logs[0]["text"] == "Verified first evidence."
    assert logs[1]["text"] == fresh_text.rstrip("\n")
    assert logs[1]["skipped_rejected_attempts"] == 1
    assert "skipped_rejected_attempts" not in logs[0]
    assert "skipped_rejected_attempts" not in logs[2]
    assert logs[2]["text"] == "Following saved action."
    for span in traces[1].data.spans:
        displayed = json.dumps({"inputs": span.inputs, "outputs": span.outputs})
        assert "Failed proposal evidence." not in displayed
        assert "HTTP response" not in displayed
        assert '"rejected"' not in displayed
    regression = next(
        span
        for span in traces[1].data.spans
        if span.name == "OWDirectRegressionADMComponent"
    )
    assert "Fresh retry evidence." in regression.inputs["input"]
    assert "Fresh retry evidence." in regression.outputs["response"]
    source = logs[1]["source"]
    assert "\n".join(
        log_text.splitlines()[source["lines"][0] - 1 : source["lines"][1]]
    ) == fresh_text.rstrip("\n")


@pytest.mark.parametrize(
    "case",
    [
        "unknown_status",
        "missing_headers",
        "invalid_headers",
        "invalid_header_dictionary",
        "missing_body",
        "incomplete_body",
        "missing_terminator",
        "delayed_exception",
        "duplicate_exception",
    ],
)
def test_unverified_rejection_suffix_stops_subsequent_evidence(tmp_path, case):
    source = episode_records(TARGET)[:3]
    records = tuple(OpenWorldRecord.model_validate(record) for record in source)
    block = rejection_block()
    if case == "unknown_status":
        block = block.replace("(400)", "(403)")
    elif case == "missing_headers":
        block = block.replace("HTTP response headers: HTTPHeaderDict({})\n", "")
    elif case == "invalid_headers":
        block = block.replace("HTTPHeaderDict({})", "HTTPHeaderDict(incomplete)")
    elif case == "invalid_header_dictionary":
        block = block.replace("HTTPHeaderDict({})", "HTTPHeaderDict({broken})")
    elif case == "missing_body":
        block = block.replace('HTTP response body: "rejected"\n', "")
    elif case == "incomplete_body":
        block = block.replace('"rejected"', '"rejected')
    elif case == "missing_terminator":
        block = block.rstrip("\n") + "\n"
    elif case == "delayed_exception":
        block = "\n" + block
    elif case == "duplicate_exception":
        block += rejection_block(500)
    log_text = (
        action_log(source[0], "Verified prefix.\n")
        + action_log(source[1], "Uncertain proposal evidence.\n")
        + block
        + action_log(source[1], "Uncertain retry evidence.\n")
        + action_log(source[2], "Uncertain following evidence.\n")
    )
    logs = read_decision_logs(tmp_path, records, log_text=log_text)
    outputs = [
        card_outputs(record, log=log)["evidence"]["log"]
        for record, log in zip(records, logs)
    ]
    assert outputs[0]["status"] == "exact_action_sequence"
    assert outputs[0]["text"] == "Verified prefix."
    for output in outputs[1:]:
        assert output["status"] == "unverified_action_outcome_at_record_1"
        assert "text" not in output
        assert "lines" not in output["source"]
        assert "skipped_rejected_attempts" not in output


@pytest.mark.parametrize("suffix", ["(", "(4", "(40", "(400", "(500"])
def test_partial_rejection_status_does_not_supply_saved_action_evidence(
    tmp_path, suffix
):
    records = episode_records(TARGET)[:3]
    log_text = (
        action_log(records[0], "Verified prefix.\n")
        + action_log(records[1], "Uncertain rejected proposal evidence.\n")
        + suffix
    )
    _, _, _, _, traces = import_traces(tmp_path, records, log_text)
    logs = [root(trace).outputs["evidence"]["log"] for trace in traces]
    assert logs[0]["status"] == "exact_action_sequence"
    assert logs[0]["text"] == "Verified prefix."
    for trace, log in zip(traces[1:], logs[1:]):
        assert log["status"] == "unverified_action_outcome_at_record_1"
        assert "text" not in log
        assert "lines" not in log["source"]
        assert "skipped_rejected_attempts" not in log
        for span in trace.data.spans:
            assert "Uncertain rejected proposal evidence." not in json.dumps(
                {"inputs": span.inputs, "outputs": span.outputs}
            )
            if span.parent_id is not None:
                assert not any(
                    source["file"] == "raw_align_system.log"
                    for source in span.get_attribute("align.evidence_sources")
                )


def test_rejected_identical_action_waits_for_a_verified_retry(tmp_path):
    source = episode_records(TARGET)[:2]
    records = tuple(OpenWorldRecord.model_validate(record) for record in source)
    log_text = (
        action_log(source[0], "Verified prefix.\n")
        + action_log(source[1], "Rejected identical action evidence.\n")
        + rejection_block()
    )
    logs = read_decision_logs(tmp_path, records, log_text=log_text)
    assert card_outputs(records[0], log=logs[0])["evidence"]["log"]["text"] == (
        "Verified prefix."
    )
    pending = card_outputs(records[1], log=logs[1])["evidence"]["log"]
    assert pending["status"] == "action_not_yet_logged"
    assert "text" not in pending
    assert "lines" not in pending["source"]


@pytest.mark.parametrize("action", [None, [], {}, {"action_id": "incomplete"}])
def test_rejection_block_does_not_make_partial_json_an_action(tmp_path, action):
    source = episode_records(TARGET)[:2]
    records = tuple(OpenWorldRecord.model_validate(record) for record in source)
    log_text = (
        action_log(source[0], "Verified prefix.\n")
        + MARKER
        + json.dumps(action)
        + "\n"
        + rejection_block()
        + action_log(source[1], "Uncertain later evidence.\n")
    )
    logs = read_decision_logs(tmp_path, records, log_text=log_text)
    assert logs[0].status == "exact_action_sequence"
    pending = card_outputs(records[1], log=logs[1])["evidence"]["log"]
    assert pending["status"] == "invalid_action_at_record_1"
    assert "text" not in pending
    assert "skipped_rejected_attempts" not in pending
