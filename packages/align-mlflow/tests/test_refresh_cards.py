"""Refresh an existing import through the CLI and inspect persisted traces."""

import copy
import json
import os
import sqlite3
import subprocess
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Span
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey
from open_world_fixtures import completion_line, episode_records, write_run

from align_mlflow.store import connect
from align_mlflow.sync import sync_run

EXPERIMENT = "card refresh"
TARGET = "ADEPT-June2025-merit-0.4"


def root_span(trace):
    return next(span for span in trace.data.spans if span.parent_id is None)


def read_traces(client, experiment_id):
    return sorted(
        client.search_traces(locations=[experiment_id]),
        key=lambda trace: trace.info.request_time,
    )


def original_format(root, record):
    data = copy.deepcopy(root.to_dict())
    data["attributes"].pop("align.card_format")
    data["attributes"][SpanAttributeKey.INPUTS] = json.dumps(record["input"])
    data["attributes"][SpanAttributeKey.OUTPUTS] = json.dumps(
        {key: value for key, value in record.items() if key != "input"}
    )
    return Span.from_dict(data)


def command(uri, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "align_mlflow",
            "refresh-cards",
            "--tracking-uri",
            uri,
            "--experiment",
            EXPERIMENT,
            *args,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )


def test_refresh_preserves_source_links_annotations_and_current_cards(tmp_path):
    records = episode_records(TARGET)
    records[0]["unknown_field"] = {"exact": [None, False, "é"]}
    records[0]["input"]["source"] = "Recorded field with a reserved display name"
    path = write_run(
        tmp_path / "2026-08-31__09-17-11",
        records,
        [completion_line(TARGET, 0.8)],
    )
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    MlflowClient(tracking_uri=uri).create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    experiment_id = connect(uri, EXPERIMENT)
    assert sync_run(experiment_id, path).error is None
    client = MlflowClient(tracking_uri=uri)
    imported = read_traces(client, experiment_id)
    older_cards = [
        original_format(root_span(trace), record)
        for trace, record in zip(imported[:5], records)
    ]
    readable = copy.deepcopy(root_span(imported[3]).to_dict())
    readable["attributes"][SpanAttributeKey.INPUTS] = json.dumps(
        {
            **root_span(imported[3]).inputs,
            "input": "**Scene:** treat_and_tag · **Elapsed:** 80s\n\nOlder summary",
        }
    )
    older_cards[3] = Span.from_dict(readable)
    TracingClient(tracking_uri=uri).log_spans(
        experiment_id,
        older_cards,
    )
    client.set_trace_tag(imported[0].info.trace_id, "review_note", "retain this")
    mlflow.log_feedback(
        trace_id=imported[0].info.trace_id,
        name="review_quality",
        value=0.7,
        rationale="Keep the review attached to this trace.",
    )
    client.delete_trace_tag(imported[4].info.trace_id, "align.import_complete")
    # Exported trace durations round differently from absolute span timestamps.
    with sqlite3.connect(tmp_path / "mlflow.db") as connection:
        connection.execute(
            "UPDATE trace_info SET execution_time_ms=execution_time_ms-1 "
            "WHERE request_id=?",
            (imported[0].info.trace_id,),
        )
    before = read_traces(client, experiment_id)

    preview = command(uri, "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert "5 cards would refresh, 1 already current" in preview.stdout
    assert [trace.to_dict() for trace in read_traces(client, experiment_id)] == [
        trace.to_dict() for trace in before
    ]

    result = command(uri)
    assert result.returncode == 0, result.stderr
    assert "5 cards refreshed, 1 already current" in result.stdout
    after = read_traces(client, experiment_id)
    assert [trace.info.to_dict() for trace in after] == [
        trace.info.to_dict() for trace in before
    ]
    for index, (old, new) in enumerate(zip(before, after)):
        if index in (4, 6):
            assert new.to_dict() == old.to_dict()
            continue
        root = root_span(new)
        assert "**Situation**" in root.inputs["input"]
        assert "**Scene:**" not in root.inputs["input"]
        assert "**Elapsed:**" not in root.inputs["input"]
        assert "**Justification**" in root.outputs["response"]
        assert {"input": root.inputs["source"], **root.outputs["source"]} == records[
            index
        ]
        assert {span.span_id for span in new.data.spans} == {
            span.span_id for span in old.data.spans
        }
        assert [span.to_dict() for span in new.data.spans if span.parent_id] == [
            span.to_dict() for span in old.data.spans if span.parent_id
        ]
    assert "**Patients**" in root_span(after[5]).inputs["input"]
    assert "Patient 1 is injured" in root_span(after[5]).inputs["input"]

    repeated = command(uri)
    assert repeated.returncode == 0, repeated.stderr
    assert "0 cards refreshed, 6 already current" in repeated.stdout
    assert [trace.to_dict() for trace in read_traces(client, experiment_id)] == [
        trace.to_dict() for trace in after
    ]


def test_refresh_rolls_back_an_invalid_page(tmp_path):
    records = episode_records(TARGET)[:2]
    path = write_run(tmp_path / "2026-08-31__09-17-11", records)
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    MlflowClient(tracking_uri=uri).create_experiment(
        EXPERIMENT, artifact_location=str(tmp_path / "artifacts")
    )
    experiment_id = connect(uri, EXPERIMENT)
    assert sync_run(experiment_id, path).error is None
    client = MlflowClient(tracking_uri=uri)
    imported = read_traces(client, experiment_id)
    records[0]["output"] = {"choice": 0}
    TracingClient(tracking_uri=uri).log_spans(
        experiment_id,
        [
            original_format(root_span(trace), record)
            for trace, record in zip(imported, records)
        ],
    )
    before = [trace.to_dict() for trace in read_traces(client, experiment_id)]
    result = command(uri)
    assert result.returncode != 0
    assert "output.action" in result.stderr
    assert [trace.to_dict() for trace in read_traces(client, experiment_id)] == before
