"""Observe captured evidence and recovery through persisted imports."""

import hashlib
import json
from pathlib import Path

import pytest
from align_utils.open_world import load_run
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from open_world_fixtures import episode_records, write_run

from align_mlflow.store import connect
from align_mlflow.provenance import preserve_run, read_provenance
from align_mlflow.sync import run_start_ns, sync_run


def store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    eid = client.create_experiment(
        "captured evidence", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    connect(uri, "captured evidence")
    return client, eid


def downloaded(client, run_id, path, destination):
    destination.mkdir()
    return Path(client.download_artifacts(run_id, path, str(destination))).read_bytes()


def test_log_bytes_and_changed_record_index_have_independent_snapshots(tmp_path):
    records = episode_records(None)[:2]
    run = write_run(tmp_path / "source", records[:1], [])
    original_log = b"Unrecognized diagnostic\r\nwith original bytes: \xff\r\n"
    (run / "raw_align_system.log").write_bytes(original_log)
    client, eid = store(tmp_path)

    first = sync_run(eid, run)
    assert first.error is None
    source = client.search_runs([eid])[0]
    reference = json.loads(source.data.tags["align.log_snapshot"])
    assert reference["sha256"] == hashlib.sha256(original_log).hexdigest()
    assert (
        downloaded(
            client, source.info.run_id, reference["artifact_path"], tmp_path / "log"
        )
        == original_log
    )
    first_index = downloaded(
        client, source.info.run_id, reference["index_path"], tmp_path / "index-before"
    )
    assert len(json.loads(first_index)["records"]) == 1

    (run / "input_output.json").write_text(json.dumps(records))
    second = sync_run(eid, run)
    assert second.error is None
    assert second.new_steps == 1
    updated = json.loads(
        client.get_run(source.info.run_id).data.tags["align.log_snapshot"]
    )
    assert updated["artifact_path"] == reference["artifact_path"]
    assert updated["index_path"] != reference["index_path"]
    assert (
        downloaded(
            client,
            source.info.run_id,
            reference["index_path"],
            tmp_path / "index-original",
        )
        == first_index
    )
    assert (
        len(
            json.loads(
                downloaded(
                    client,
                    source.info.run_id,
                    updated["index_path"],
                    tmp_path / "index-after",
                )
            )["records"]
        )
        == 2
    )
    assert sync_run(eid, run).new_steps == 0
    assert len(client.search_traces(locations=[eid])) == 2


def test_missing_raw_log_imports_actions_and_retains_a_later_available_log(tmp_path):
    run = write_run(tmp_path / "source", episode_records(None)[:1], [])
    log_path = run / "raw_align_system.log"
    log_path.unlink()
    client, eid = store(tmp_path)

    first = sync_run(eid, run)
    assert first.error is None
    assert first.new_steps == 1
    source = client.search_runs([eid])[0]
    assert "align.log_snapshot" not in source.data.tags
    original_trace = client.search_traces(locations=[eid])[0].to_dict()

    content = b"Late diagnostic\r\n"
    log_path.write_bytes(content)
    second = sync_run(eid, run)
    assert second.error is None
    assert second.new_steps == 0
    reference = json.loads(
        client.get_run(source.info.run_id).data.tags["align.log_snapshot"]
    )
    assert (
        downloaded(
            client, source.info.run_id, reference["artifact_path"], tmp_path / "log"
        )
        == content
    )

    log_path.unlink()
    third = sync_run(eid, run)
    assert third.error is None
    assert third.new_steps == 0
    assert (
        json.loads(client.get_run(source.info.run_id).data.tags["align.log_snapshot"])
        == reference
    )
    assert client.search_traces(locations=[eid])[0].to_dict() == original_trace


def test_incomplete_timing_artifact_recovers_only_original_snapshot(tmp_path):
    records = episode_records(None)[:2]
    for record in records:
        record["choice_info"] = {"unfamiliar": [None, True]}
    run = write_run(tmp_path / "source", records, [])
    (run / "timing.json").write_text(
        json.dumps({"scenarios": [], "raw_times_s": [1.25, 2.5]})
    )
    client, eid = store(tmp_path)
    assert sync_run(eid, run).error is None
    before = {
        t.info.trace_id: t.to_dict() for t in client.search_traces(locations=[eid])
    }
    source = client.search_runs([eid])[0]
    reference = json.loads(source.data.tags["align.timeline"])
    assert reference["basis"] == "raw_times_s"
    saved = (
        tmp_path
        / "artifacts"
        / source.info.run_id
        / "artifacts"
        / reference["artifact_path"]
    )
    original = saved.read_bytes()

    saved.write_bytes(b"incomplete")
    recovered = sync_run(eid, run)
    assert recovered.error is None
    assert recovered.new_steps == 0
    assert saved.read_bytes() == original
    assert {
        t.info.trace_id: t.to_dict() for t in client.search_traces(locations=[eid])
    } == before

    saved.write_bytes(b"incomplete")
    (run / "timing.json").write_text(
        json.dumps({"scenarios": [], "raw_times_s": [9, 9]})
    )
    unresolved = sync_run(eid, run)
    assert unresolved.error is None
    assert "Captured timing was unavailable" in unresolved.warning
    assert saved.read_bytes() == b"incomplete"
    assert {
        t.info.trace_id: t.to_dict() for t in client.search_traces(locations=[eid])
    } == before


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_lost_timing_resumes_after_confirmed_root_with_frozen_unknown_durations(
    tmp_path, damage
):
    records = episode_records(None)[:3]
    for record in records:
        record["choice_info"] = {}
    run = write_run(tmp_path / "source", records[:1], [])
    timing = run / "timing.json"
    timing.write_text(json.dumps({"scenarios": [], "raw_times_s": [1.234567891]}))
    client, eid = store(tmp_path)
    assert sync_run(eid, run).error is None
    first = client.search_traces(locations=[eid])[0]
    first_before = first.to_dict()
    first_root = next(span for span in first.data.spans if span.parent_id is None)
    source = client.search_runs([eid])[0]
    reference = json.loads(source.data.tags["align.timeline"])
    artifact = (
        tmp_path
        / "artifacts"
        / source.info.run_id
        / "artifacts"
        / reference["artifact_path"]
    )
    if damage == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"unavailable")
    records[1]["choice_info"] = {
        "per_step_timing_stats": [
            {"step": "recorded.component", "step_num": 0, "elapsed_s": 1.5}
        ]
    }
    (run / "input_output.json").write_text(json.dumps(records[:2]))
    timing.write_text(json.dumps({"scenarios": [], "raw_times_s": [99, 2.5]}))
    source_bytes = timing.read_bytes()

    resumed = sync_run(eid, run)

    assert resumed.error is None
    assert resumed.new_steps == 1
    assert "Captured timing was unavailable" in resumed.warning
    traces = sorted(
        client.search_traces(locations=[eid]),
        key=lambda trace: int(trace.info.tags["align.record_index"]),
    )
    assert traces[0].to_dict() == first_before
    second_root = next(span for span in traces[1].data.spans if span.parent_id is None)
    assert second_root.start_time_ns == first_root.end_time_ns + 1_000_000
    assert second_root.end_time_ns - second_root.start_time_ns == 1_000
    assert traces[1].info.tags["align.timing_source"] == "unknown"
    assert len(traces[1].data.spans) == 1
    assert second_root.outputs["source"]["choice_info"] == records[1]["choice_info"]
    assert timing.read_bytes() == source_bytes
    if damage == "missing":
        assert not artifact.exists()
    else:
        assert artifact.read_bytes() == b"unavailable"
    before = {trace.info.trace_id: trace.to_dict() for trace in traces}
    assert sync_run(eid, run).new_steps == 0
    assert {
        trace.info.trace_id: trace.to_dict()
        for trace in client.search_traces(locations=[eid])
    } == before

    (run / "input_output.json").write_text(json.dumps(records))
    timing.unlink()
    appended = sync_run(eid, run)
    assert appended.error is None
    assert appended.new_steps == 1
    assert "unknown" in appended.warning
    traces = sorted(
        client.search_traces(locations=[eid]),
        key=lambda trace: int(trace.info.tags["align.record_index"]),
    )
    assert {trace.info.trace_id: trace.to_dict() for trace in traces[:2]} == before
    third_root = next(span for span in traces[2].data.spans if span.parent_id is None)
    assert third_root.start_time_ns == second_root.end_time_ns + 1_000_000
    assert third_root.end_time_ns - third_root.start_time_ns == 1_000
    before = {trace.info.trace_id: trace.to_dict() for trace in traces}
    assert sync_run(eid, run).new_steps == 0
    assert {
        trace.info.trace_id: trace.to_dict()
        for trace in client.search_traces(locations=[eid])
    } == before


def test_lost_timing_without_confirmed_traces_uses_recorded_origin(tmp_path):
    records = episode_records(None)[:1]
    records[0]["choice_info"] = {}
    run = write_run(tmp_path / "2026-09-24__16-05-25", records, [])
    timing = run / "timing.json"
    timing.write_text(json.dumps({"scenarios": [], "raw_times_s": [1.25]}))
    client, eid = store(tmp_path)
    source_id = preserve_run(eid, load_run(run), read_provenance(run), origin_ns=1)
    source = client.get_run(source_id)
    reference = json.loads(source.data.tags["align.timeline"])
    artifact = (
        tmp_path / "artifacts" / source_id / "artifacts" / reference["artifact_path"]
    )
    artifact.unlink()
    timing.write_text(json.dumps({"scenarios": [], "raw_times_s": [99]}))

    resumed = sync_run(eid, run)

    assert resumed.error is None
    assert resumed.new_steps == 1
    assert "Captured timing was unavailable" in resumed.warning
    trace = client.search_traces(locations=[eid])[0]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.start_time_ns == run_start_ns(run)
    assert root.end_time_ns - root.start_time_ns == 1_000
    assert trace.info.tags["align.timing_source"] == "unknown"


@pytest.mark.parametrize("durations", [[9e9], [4e9, 4e9]])
def test_component_durations_outside_timestamp_range_keep_actions_and_evidence(
    tmp_path, durations
):
    records = episode_records(None)[:1]
    records[0]["choice_info"]["per_step_timing_stats"] = [
        {
            "step": f"recorded.component_{index}",
            "step_num": index,
            "elapsed_s": duration,
        }
        for index, duration in enumerate(durations)
    ]
    run = write_run(tmp_path / "2026-09-24__16-05-25", records, [])
    client, eid = store(tmp_path)

    imported = sync_run(eid, run)

    assert imported.error is None
    assert imported.new_steps == 1
    assert "unknown" in imported.warning
    trace = client.search_traces(locations=[eid])[0]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert trace.info.tags["align.timing_source"] == "unknown"
    assert len(trace.data.spans) == 1 + len(durations)
    assert all(
        root.start_time_ns
        <= span.start_time_ns
        <= span.end_time_ns
        <= root.end_time_ns
        < 2**63
        for span in trace.data.spans
    )
    assert root.outputs["source"]["choice_info"] == records[0]["choice_info"]


def test_unfamiliar_optional_context_does_not_prevent_action_import(tmp_path):
    contexts = [
        None,
        ["a different implementation", {"detail": False}],
        {
            "alignment_info": ["unfamiliar"],
            "per_step_timing_stats": "not measured",
            "other": {"value": 1},
        },
    ]
    records = episode_records(None)[: len(contexts)]
    for record, context in zip(records, contexts):
        record["choice_info"] = context
    run = write_run(tmp_path / "source", records, [])
    client, eid = store(tmp_path)
    result = sync_run(eid, run)
    assert result.error is None
    assert result.new_steps == len(contexts)
    traces = sorted(
        client.search_traces(locations=[eid]),
        key=lambda t: int(t.info.tags["align.record_index"]),
    )
    for trace, context in zip(traces, contexts):
        assert len(trace.data.spans) == 1
        root = trace.data.spans[0]
        assert root.outputs["source"]["choice_info"] == context
        assert root.outputs["evidence"]["choice_info"]["present"] is True


def test_interrupted_raw_log_upload_resumes_with_original_bytes(tmp_path, monkeypatch):
    run = write_run(tmp_path / "source", episode_records(None)[:1], [])
    original = b"Captured diagnostic\r\n"
    (run / "raw_align_system.log").write_bytes(original)
    client, eid = store(tmp_path)
    upload = MlflowClient.log_artifact
    interrupted = False

    def interrupt_once(self, run_id, local_path, artifact_path=None):
        nonlocal interrupted
        if Path(local_path).name == "raw_align_system.log" and not interrupted:
            interrupted = True
            destination = (
                tmp_path
                / "artifacts"
                / run_id
                / "artifacts"
                / artifact_path
                / Path(local_path).name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"partial upload")
            raise MlflowException("Interrupted artifact upload")
        return upload(self, run_id, local_path, artifact_path)

    monkeypatch.setattr(MlflowClient, "log_artifact", interrupt_once)
    assert sync_run(eid, run).error is not None
    assert not client.search_traces(locations=[eid])
    resumed = sync_run(eid, run)
    assert resumed.error is None
    assert resumed.new_steps == 1
    source = client.search_runs([eid])[0]
    reference = json.loads(source.data.tags["align.log_snapshot"])
    assert (
        downloaded(
            client,
            source.info.run_id,
            reference["artifact_path"],
            tmp_path / "download",
        )
        == original
    )
