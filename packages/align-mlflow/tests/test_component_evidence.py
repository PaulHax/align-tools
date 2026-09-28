"""Exercise component evidence through import and maintenance commands."""

import json
import os
import sqlite3
import subprocess
import sys

import mlflow
import pytest
from mlflow import MlflowClient
from open_world_fixtures import completion_line, episode_records, write_run

EXPERIMENT = "component evidence"
TARGET = "ADEPT-June2025-merit-0.4"
REGRESSION = "OWDirectRegressionADMComponent"
SELECTION = "OWChoiceToActionADMComponent"
PARAMETER = "OWActionParameterCompletionADMComponent"


def command(uri, path, name="sync", *options):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "align_mlflow",
            name,
            str(path),
            "--tracking-uri",
            uri,
            "--experiment",
            EXPERIMENT,
            *options,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    return result


def setup(tmp_path):
    records = episode_records(TARGET)
    records = records[:2] + [records[5]]
    for record in records[:2]:
        timings = record["choice_info"]["per_step_timing_stats"]
        for component in (SELECTION, PARAMETER):
            timings.append(
                {
                    "step": f"align_system.algorithms.open_world_components.{component}",
                    "step_num": len(timings),
                    "elapsed_s": 0.125,
                }
            )
        record["choice_info"][REGRESSION] = {
            "unknown": [None, False, "é"],
            "source": ["arbitrary", "field"],
        }
    run = write_run(
        tmp_path / "2026-08-31__09-17-11", records, [completion_line(TARGET, 0.8)]
    )
    fragments = [
        """Cache miss for `direct_regression_adm_component` ..
[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]
Patient context with ``` literal backticks.
[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]
{"score": 97, "reasoning": "Recorded regression reasoning"}
{0: 1.0, 1: 0.0}
Max vote keys: [0]
Cache miss for `outlines_baseline_adm_component` ..
[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]
Select an action for the chosen patient.
[bold]*VOTES*[/bold]
{"Check vitals": 0.9, "Tag patient": 0.1}
Cache miss for `ow_action_parameter_completion_adm_component` ..
[bold]*TREATMENT FOLLOWUP PROMPT*[/bold]
Choose a supply.
[bold]*TREATMENT FOLLOWUP RESPONSE*[/bold]
{"treatment": "Tourniquet", "reasoning": "Recorded parameter reasoning"}
""",
        "Cache hit for `direct_regression_adm_component` returning cached output\n",
        "",
    ]
    log = "".join(
        fragment
        + "[bold]*ACTION BEING TAKEN*[/bold]\n"
        + json.dumps(record["output"]["action"], indent=2)
        + "\n"
        for fragment, record in zip(fragments, records)
    )
    (run / "raw_align_system.log").write_text(log + completion_line(TARGET, 0.8) + "\n")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    return run, records, uri, client, experiment_id


def traces(client, experiment_id):
    return sorted(
        client.search_traces(locations=[experiment_id]),
        key=lambda t: int(t.info.tags["align.record_index"]),
    )


def component(trace, short):
    return next(
        s
        for s in trace.data.spans
        if (s.get_attribute("component") or "").endswith("." + short)
    )


def root(trace):
    return next(s for s in trace.data.spans if s.parent_id is None)


def assert_evidence(imported, records, path):
    first, cached, driver = imported
    reg = component(first, REGRESSION)
    assert "Patient context with ```" in reg.inputs["input"]
    assert "Recorded regression reasoning" in reg.outputs["response"]
    assert "Select an action" not in reg.inputs["input"]
    assert (
        reg.outputs["recorded_choice_info"][REGRESSION]
        == records[0]["choice_info"][REGRESSION]
    )
    select = component(first, SELECTION)
    assert "Select an action" in select.inputs["input"]
    assert "Check vitals" in select.outputs["response"]
    assert "Recorded regression reasoning" not in select.outputs["response"]
    param = component(first, PARAMETER)
    assert "Choose a supply" in param.inputs["input"]
    assert "Recorded parameter reasoning" in param.outputs["response"]
    align = component(first, "MultinomialWeightedMidpointAlignmentADMComponent")
    assert align.inputs["alignment_target_id"] == TARGET
    assert (
        align.inputs["recorded_predicted_kdma_values"]
        == records[0]["choice_info"]["predicted_kdma_values"]
    )
    assert align.outputs["recorded_choice_info"]["alignment_info"]["votes"] == {
        "0": 1.0,
        "1": 0.0,
    }
    assert "Max vote keys: [0]" in align.outputs["response"]
    world = component(first, "WorldStateTrackerADMComponent")
    assert world.inputs["observed_state"] == records[0]["input"]["full_state"]
    assert "context" not in world.inputs
    cached_reg = component(cached, REGRESSION)
    assert cached_reg.get_attribute("align.cache_status") == ["hit"]
    assert "No fresh prompt" in cached_reg.inputs["input"]
    assert "response" not in cached_reg.outputs
    assert len(driver.data.spans) == 1
    log_lines = (path / "raw_align_system.log").read_text().splitlines()
    for ref in reg.get_attribute("align.evidence_sources"):
        if ref["file"] == "raw_align_system.log":
            assert (
                ref["title"] == "cache"
                or log_lines[ref["lines"][0] - 1] == f"[bold]*{ref['title']}*[/bold]"
            )
    assert all(
        root(t).inputs["source"] == r["input"] for t, r in zip(imported, records)
    )


def test_import_records_component_evidence_with_checked_log_sources(tmp_path):
    path, records, uri, client, eid = setup(tmp_path)
    result = command(uri, path)
    assert result.returncode == 0, result.stderr
    assert_evidence(traces(client, eid), records, path)


def test_refresh_preserves_decisions_ids_timings_and_reviews(tmp_path):
    path, records, uri, client, eid = setup(tmp_path)
    result = command(uri, path)
    assert result.returncode == 0, result.stderr
    imported = traces(client, eid)
    mlflow.set_tracking_uri(uri)
    mlflow.log_feedback(
        trace_id=imported[0].info.trace_id,
        name="developer_review",
        value=True,
        rationale="Keep this review",
    )
    client.set_trace_tag(imported[0].info.trace_id, "review_note", "Retain this tag")
    with sqlite3.connect(tmp_path / "mlflow.db") as db:
        for tid, sid, content in db.execute(
            "SELECT trace_id,span_id,content FROM spans WHERE parent_span_id IS NOT NULL"
        ).fetchall():
            data = json.loads(content)
            data["attributes"] = {
                k: v
                for k, v in data["attributes"].items()
                if not k.startswith("align.")
                and k not in ("mlflow.spanInputs", "mlflow.spanOutputs")
            }
            db.execute(
                "UPDATE spans SET content=? WHERE trace_id=? AND span_id=?",
                (json.dumps(data), tid, sid),
            )
    before = traces(client, eid)
    preview = command(uri, path, "refresh-components", "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert "10 components would refresh" in preview.stdout
    assert [t.to_dict() for t in traces(client, eid)] == [t.to_dict() for t in before]
    refreshed = command(uri, path, "refresh-components")
    assert refreshed.returncode == 0, refreshed.stderr
    after = traces(client, eid)
    assert_evidence(after, records, path)
    assert [t.info.to_dict() for t in after] == [t.info.to_dict() for t in before]
    assert [root(t).to_dict() for t in after] == [root(t).to_dict() for t in before]
    for old, new in zip(before, after):
        assert {
            (s.span_id, s.parent_id, s.start_time_ns, s.end_time_ns)
            for s in old.data.spans
        } == {
            (s.span_id, s.parent_id, s.start_time_ns, s.end_time_ns)
            for s in new.data.spans
        }
    repeated = command(uri, path, "refresh-components")
    assert repeated.returncode == 0, repeated.stderr
    assert "0 components refreshed" in repeated.stdout
    assert [t.to_dict() for t in traces(client, eid)] == [t.to_dict() for t in after]


def test_mismatched_action_log_is_not_attributed_to_components(tmp_path):
    path, records, uri, client, eid = setup(tmp_path)
    log = path / "raw_align_system.log"
    log.write_text(
        log.read_text().replace(
            '"character_id": "Patient 1"', '"character_id": "Different patient"', 1
        )
    )
    result = command(uri, path)
    assert result.returncode == 0, result.stderr
    for trace in traces(client, eid):
        for span in trace.data.spans:
            if span.parent_id is None:
                continue
            assert (
                span.get_attribute("align.log_match") == "action_mismatch_at_record_0"
            )
            assert not any(
                r["file"] == "raw_align_system.log"
                for r in span.get_attribute("align.evidence_sources")
            )
            assert "PROMPT" not in str(span.inputs)


@pytest.mark.parametrize("relevance_count", [1, 2])
def test_relevance_logs_are_not_assigned_to_regression(tmp_path, relevance_count):
    records = episode_records(TARGET)[:1]
    info = records[0]["choice_info"]
    info["predicted_relevance"] = {"medical": [1], "merit": [0]}
    timings = info["per_step_timing_stats"]
    for _ in range(relevance_count):
        timings.append(
            {
                "step": "align_system.algorithms.relevance_adm_component.PredictMostRelevantADMComponent",
                "step_num": len(timings),
                "elapsed_s": 0.1,
            }
        )
    path = write_run(tmp_path / "2026-08-31__09-17-11", records)
    (path / "raw_align_system.log").write_text(
        "Cache miss for `direct_regression_adm_component` ..\n"
        "[bold]*MEDICAL PREDICTION DIALOG PROMPT*[/bold]\nScore medical urgency.\n"
        '[bold]*MEDICAL PREDICTION RESPONSE (sample #0)*[/bold]\n{"score": 97}\n'
        "[bold]*MOST RELEVANT ATTRIBUTE PREDICTION DIALOG PROMPT*[/bold]\n"
        "Choose the most relevant attribute.\n"
        "[bold]*RELEVANCE PREDICTION RESPONSE (sample #0)*[/bold]\n"
        '{"most_relevant": "medical"}\n'
        "[bold]*ACTION BEING TAKEN*[/bold]\n"
        + json.dumps(records[0]["output"]["action"])
        + "\n"
    )
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = MlflowClient(tracking_uri=uri)
    eid = client.create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    result = command(uri, path)
    assert result.returncode == 0, result.stderr
    imported = traces(client, eid)[0]
    regression = component(imported, REGRESSION)
    assert "Score medical urgency" in regression.inputs["input"]
    assert "most relevant" not in str(regression.inputs)
    assert "most_relevant" not in str(regression.outputs)
    relevance = component(imported, "PredictMostRelevantADMComponent")
    assert relevance.outputs["recorded_choice_info"]["predicted_relevance"] == {
        "medical": [1],
        "merit": [0],
    }
    if relevance_count == 1:
        assert "Choose the most relevant" in relevance.inputs["input"]
        assert "most_relevant" in relevance.outputs["response"]
    else:
        assert relevance.inputs is None
        assert "response" not in relevance.outputs
