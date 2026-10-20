"""Observe captured evidence and recovery through persisted imports."""

import hashlib
import json
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from open_world_fixtures import episode_records, write_run

from align_mlflow.store import connect
from align_mlflow.sync import sync_run


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

    first = sync_run(eid, run, include_log=True)
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
    second = sync_run(eid, run, include_log=True)
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
    assert sync_run(eid, run, include_log=True).new_steps == 0
    assert len(client.search_traces(locations=[eid])) == 2


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
    failed = sync_run(eid, run)
    assert failed.error is not None
    assert "digest" in failed.error
    assert saved.read_bytes() == b"incomplete"
    assert {
        t.info.trace_id: t.to_dict() for t in client.search_traces(locations=[eid])
    } == before


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
    assert sync_run(eid, run, include_log=True).error is not None
    assert not client.search_traces(locations=[eid])
    resumed = sync_run(eid, run, include_log=True)
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
