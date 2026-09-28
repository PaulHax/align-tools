"""Enrich confirmed decision traces in a local MLflow 3.16.1 SQLite store."""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import mlflow
from align_utils.open_world import find_open_world_runs, load_run
from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey
from sqlalchemy.engine import make_url

from .cards import CARD_FORMAT
from .evidence import EVIDENCE_VERSION, read_decision_logs
from .traces import component_spans, run_key


def refresh_components(
    tracking_uri: str, experiment_name: str, root: Path, dry_run: bool = False
):
    if mlflow.__version__ != "3.16.1":
        raise ValueError("Component refresh is verified with MLflow 3.16.1 only")
    url = make_url(tracking_uri)
    if url.drivername != "sqlite" or not url.database or url.query:
        raise ValueError("Component refresh requires a local sqlite:/// tracking URI")
    database = Path(url.database).resolve()
    if not database.is_file():
        raise ValueError(f"Tracking database does not exist: {database}")
    mode = "ro" if dry_run else "rw"
    with closing(sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True)) as db:
        experiment = db.execute(
            "SELECT experiment_id FROM experiments WHERE name=? AND lifecycle_stage='active'",
            (experiment_name,),
        ).fetchone()
        if experiment is None:
            raise ValueError(f"Active experiment {experiment_name!r} does not exist")
        seen = set()
        for path in find_open_world_runs(root):
            run = load_run(path)
            key = run_key(run)
            if key in seen:
                continue
            seen.add(key)
            records = run.records
            logs = read_decision_logs(path, records)
            traces = db.execute(
                "SELECT run.request_id,record.value FROM trace_tags run "
                "JOIN trace_tags record ON record.request_id=run.request_id AND record.key='align.record_index' "
                "JOIN trace_tags complete ON complete.request_id=run.request_id AND complete.key='align.import_complete' "
                "AND complete.value='source-record-v1' "
                "JOIN trace_info t ON t.request_id=run.request_id AND t.experiment_id=? "
                "WHERE run.key='align.run_key' AND run.value=?",
                (experiment[0], key),
            ).fetchall()
            changed = 0
            for offset in range(0, len(traces), 50):
                with db:
                    db.execute("BEGIN" if dry_run else "BEGIN IMMEDIATE")
                    for trace_id, index_text in traces[offset : offset + 50]:
                        index = int(index_text)
                        if not 0 <= index < len(records):
                            raise ValueError(
                                f"{trace_id}: source record {index} is absent"
                            )
                        stored = db.execute(
                            "SELECT span_id,content FROM spans WHERE trace_id=? ORDER BY start_time_unix_nano,span_id",
                            (trace_id,),
                        ).fetchall()
                        spans = [
                            Span.from_dict(json.loads(content)) for _, content in stored
                        ]
                        roots = [span for span in spans if span.parent_id is None]
                        if len(roots) != 1:
                            raise ValueError(f"{trace_id}: expected one decision root")
                        parent = roots[0]
                        source = (
                            {
                                "input": parent.inputs["source"],
                                **parent.outputs["source"],
                            }
                            if parent.get_attribute("align.card_format") == CARD_FORMAT
                            else {"input": parent.inputs, **parent.outputs}
                        )
                        if source != records[index].source:
                            raise ValueError(
                                f"{trace_id}: source record differs from the confirmed import"
                            )
                        children = [
                            span for span in spans if span.parent_id is not None
                        ]
                        expected = component_spans(
                            records[index], parent.start_time_ns, index, logs[index]
                        )
                        if len(children) != len(expected) or any(
                            child.parent_id != parent.span_id
                            or child.get_attribute("component")
                            != spec.attributes["component"]
                            for child, spec in zip(children, expected)
                        ):
                            raise ValueError(
                                f"{trace_id}: stored pipeline differs from the source"
                            )
                        location = db.execute(
                            "SELECT value FROM trace_tags WHERE request_id=? AND key='mlflow.trace.spansLocation'",
                            (trace_id,),
                        ).fetchone()
                        if location != ("TRACKING_STORE",):
                            raise ValueError(f"{trace_id}: spans are archived")
                        contents = {sid: json.loads(content) for sid, content in stored}
                        trace_changed = False
                        for child, spec in zip(children, expected):
                            if (
                                child.inputs is not None or child.outputs is not None
                            ) and child.get_attribute(
                                "align.component_evidence"
                            ) != EVIDENCE_VERSION:
                                raise ValueError(
                                    f"{trace_id}/{child.span_id}: unrecognized component payload; refusing to replace it"
                                )
                            content = contents[child.span_id]
                            before = json.dumps(content)
                            attrs = content["attributes"]
                            for name in (
                                "align.component_evidence",
                                "align.log_match",
                                "align.evidence_sources",
                                "align.evidence_notes",
                                "align.cache_status",
                            ):
                                attrs.pop(name, None)
                            attrs.update(
                                {
                                    name: json.dumps(value, ensure_ascii=False)
                                    for name, value in spec.attributes.items()
                                }
                            )
                            for name, value in (
                                (SpanAttributeKey.INPUTS, spec.inputs),
                                (SpanAttributeKey.OUTPUTS, spec.outputs),
                            ):
                                if value is None:
                                    attrs.pop(name, None)
                                else:
                                    attrs[name] = json.dumps(value, ensure_ascii=False)
                            if json.dumps(content) == before:
                                continue
                            changed += 1
                            trace_changed = True
                            if not dry_run:
                                payload = json.dumps(content, ensure_ascii=False)
                                db.execute(
                                    "UPDATE spans SET content=? WHERE trace_id=? AND span_id=?",
                                    (payload, trace_id, child.span_id),
                                )
                                if db.execute(
                                    "SELECT content FROM spans WHERE trace_id=? AND span_id=?",
                                    (trace_id, child.span_id),
                                ).fetchone() != (payload,):
                                    raise ValueError(
                                        f"{trace_id}: component read-back did not match"
                                    )
                        if trace_changed and not dry_run:
                            db.execute(
                                "UPDATE trace_info SET db_payload_generation=COALESCE(db_payload_generation,0)+1 WHERE request_id=?",
                                (trace_id,),
                            )
            yield (
                path,
                changed,
                dict(Counter(logs[int(index)].status for _, index in traces)),
            )
