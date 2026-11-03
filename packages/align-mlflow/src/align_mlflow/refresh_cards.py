"""Refresh imported cards in an MLflow 3.17.2 SQLite store."""

import copy
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterator, Optional

import mlflow
from align_utils.open_world import (
    RAW_LOG_FILE,
    OpenWorldRecord,
    find_open_world_runs,
    load_run,
)
from mlflow import MlflowClient
from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey
from sqlalchemy.engine import make_url

from .cards import CARD_FORMAT, card_inputs, card_outputs
from .artifacts import preserve_log
from .evidence import DecisionLog, Section, _sections, read_decision_logs
from .process import PROCESS_VERSION
from .provenance import read_provenance
from .traces import run_key


def _source_record(root: Span) -> OpenWorldRecord:
    card_format = root.get_attribute("align.card_format")
    if card_format not in (None, CARD_FORMAT):
        raise ValueError(f"{root.trace_id}: unsupported card format {card_format!r}")
    if not isinstance(root.inputs, dict) or not isinstance(root.outputs, dict):
        raise ValueError(f"{root.trace_id}: expected preserved source objects")
    source = (
        {"input": root.inputs["source"], **root.outputs["source"]}
        if card_format == CARD_FORMAT
        else {"input": root.inputs, **root.outputs}
    )
    return OpenWorldRecord.model_validate(source)


def refreshed_root(
    root: Span,
    previous: Optional[OpenWorldRecord] = None,
    *,
    source_log: Optional[DecisionLog] = None,
    source_record_index: Optional[int] = None,
) -> Optional[Span]:
    record = _source_record(root)
    card_format = root.get_attribute("align.card_format")
    recorded_evidence = root.outputs.get("evidence")
    record_index = (
        recorded_evidence.get("record_index")
        if isinstance(recorded_evidence, dict)
        else None
    )
    if source_record_index is not None:
        record_index = source_record_index
    captured = (
        recorded_evidence.get("log") if isinstance(recorded_evidence, dict) else None
    )
    log = None
    if isinstance(captured, dict):
        source = captured.get("source")
        if not isinstance(source, dict):
            source = {}
        text = captured.get("text")
        lines = source.get("lines")
        sections: tuple[Section, ...] = ()
        excerpt = None
        if (
            isinstance(text, str)
            and not captured.get("text_truncated")
            and isinstance(lines, list)
            and len(lines) == 2
            and all(isinstance(line, int) for line in lines)
        ):
            excerpt = Section(
                source.get("title", "Captured decision log"), text, lines[0], lines[1]
            )
            sections = _sections(text, lines[0])
        log = DecisionLog(
            captured.get("status", "not_loaded"),
            sections=sections,
            excerpt=excerpt,
            snapshot_sha256=source.get("snapshot_sha256"),
            skipped_rejected_attempts=captured.get("skipped_rejected_attempts", 0),
        )
    inputs, outputs = (
        card_inputs(record, previous),
        card_outputs(
            record,
            record_index=record_index,
            log=source_log if source_log is not None else log,
        ),
    )
    if (
        source_log is None
        and card_format == CARD_FORMAT
        and isinstance(recorded_evidence, dict)
    ):
        for key in ("record_index", "log"):
            if key in recorded_evidence:
                outputs["evidence"][key] = recorded_evidence[key]
        process = recorded_evidence.get("process")
        if (
            isinstance(process, dict)
            and process.get("schema_version") == PROCESS_VERSION
        ):
            outputs["evidence"]["process"] = process
        elif isinstance(captured, dict) and captured.get("text_truncated"):
            outputs["evidence"]["process"]["gaps"].append(
                "Only a bounded log preview was saved on this trace; full activities require the captured source artifact."
            )
    if card_format == CARD_FORMAT and root.inputs == inputs and root.outputs == outputs:
        return None
    data = copy.deepcopy(root.to_dict())
    data["attributes"].update(
        {
            SpanAttributeKey.INPUTS: json.dumps(inputs, ensure_ascii=False),
            SpanAttributeKey.OUTPUTS: json.dumps(outputs, ensure_ascii=False),
            "align.card_format": json.dumps(CARD_FORMAT),
        }
    )
    return Span.from_dict(data)


def _source_evidence(db, experiment_id, source_root, tracking_uri, dry_run):
    wanted = {
        row[0]
        for row in db.execute(
            "SELECT DISTINCT run.value FROM trace_tags run "
            "JOIN trace_info t ON t.request_id=run.request_id "
            "WHERE t.experiment_id=? AND run.key='align.run_key'",
            (experiment_id,),
        )
    }
    references = []
    sources = {}
    bindings = db.execute(
        "SELECT r.run_uuid,k.value,p.value,d.value FROM runs r "
        "JOIN tags k ON k.run_uuid=r.run_uuid AND k.key='align.run_key' "
        "LEFT JOIN tags p ON p.run_uuid=r.run_uuid AND p.key='align.source_path' "
        "LEFT JOIN tags d ON d.run_uuid=r.run_uuid AND d.key='align.provenance_sha256' "
        "WHERE r.experiment_id=? AND r.lifecycle_stage='active'",
        (experiment_id,),
    ).fetchall()
    bound_runs = {}
    for path in find_open_world_runs(source_root):
        run = load_run(path, log_text=None)
        current_key = run_key(run)
        candidates = [
            binding
            for binding in bindings
            if binding[1] == current_key
            or isinstance(binding[2], str)
            and Path(binding[2]).resolve() == path.resolve()
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous imported source identity for {path}")
        if not candidates or candidates[0][1] not in wanted:
            continue
        source_run_id, key, _, provenance = candidates[0]
        if key in sources:
            raise ValueError(f"Multiple source directories have run key {key}")
        if provenance is not None:
            expected = json.loads(provenance)
            captured = read_provenance(path)
            if (
                not isinstance(expected, dict)
                or not expected
                or any(
                    name not in captured
                    or hashlib.sha256(captured[name]).hexdigest() != digest
                    for name, digest in expected.items()
                )
            ):
                raise ValueError(
                    f"Source provenance differs for {path}; decision evidence was not attached"
                )
        elif key != current_key:
            raise ValueError(
                f"Recorded source provenance is required to resolve {path}"
            )
        bound_runs[key] = source_run_id
        rows = db.execute(
            "SELECT s.trace_id,s.content,record.value FROM spans s "
            "JOIN trace_tags run ON run.request_id=s.trace_id AND run.key='align.run_key' "
            "JOIN trace_tags record ON record.request_id=s.trace_id AND record.key='align.record_index' "
            "JOIN trace_tags complete ON complete.request_id=s.trace_id "
            "AND complete.key='align.import_complete' AND complete.value='source-record-v1' "
            "WHERE s.experiment_id=? AND s.parent_span_id IS NULL AND run.value=?",
            (experiment_id, key),
        )
        for trace_id, content, index in rows:
            index = int(index)
            if (
                index < 0
                or index >= len(run.records)
                or _source_record(Span.from_dict(json.loads(content))).source
                != run.records[index].source
            ):
                raise ValueError(
                    f"{trace_id}: source record differs; decision evidence was not attached"
                )
            references.append((trace_id, key, index))
        try:
            content = (path / RAW_LOG_FILE).read_bytes()
        except FileNotFoundError:
            content = None
        logs = read_decision_logs(
            path,
            run.records,
            log_text=content.decode("utf-8", errors="replace")
            if content is not None
            else None,
            snapshot_sha256=hashlib.sha256(content).hexdigest()
            if content is not None
            else None,
        )
        sources[key] = (logs, content)
    if not sources:
        raise ValueError(
            "No original source directories match this experiment's imported run keys"
        )
    if not dry_run:
        client = MlflowClient(tracking_uri=tracking_uri)
        for key, (logs, content) in sources.items():
            preserve_log(client, client.get_run(bound_runs[key]), content, logs)
    return {
        trace_id: (sources[key][0][int(index)], int(index))
        for trace_id, key, index in references
    }


def _previous_traces(db: sqlite3.Connection, experiment_id: int) -> dict[str, str]:
    rows = db.execute(
        "SELECT t.request_id,t.session_id,run.value,step.value,record.value "
        "FROM trace_info t "
        "JOIN trace_tags run ON run.request_id=t.request_id AND run.key='align.run_key' "
        "JOIN trace_tags step ON step.request_id=t.request_id AND step.key='step' "
        "JOIN trace_tags record ON record.request_id=t.request_id AND record.key='align.record_index' "
        "JOIN trace_tags complete ON complete.request_id=t.request_id AND complete.key='align.import_complete' "
        "AND complete.value='source-record-v1' "
        "WHERE t.experiment_id=? AND t.session_id IS NOT NULL",
        (experiment_id,),
    ).fetchall()
    by_step = defaultdict(list)
    for trace_id, session, run, step, record in rows:
        by_step[(session, run, int(step))].append((trace_id, int(record)))
    result = {}
    for (session, run, step), current in by_step.items():
        previous = by_step.get((session, run, step - 1), []) if step > 1 else []
        if len(current) == len(previous) == 1 and current[0][1] == previous[0][1] + 1:
            result[current[0][0]] = previous[0][0]
    return result


def refresh_cards(
    tracking_uri: str,
    experiment_name: str,
    dry_run: bool = False,
    source_root: Optional[Path] = None,
) -> Iterator[tuple[int, int]]:
    """Yield cumulative (updated, already current) counts after each page.

    Uses the 3.17.2 SQLite schema. Only root payloads and their cache generations
    are updated; IDs, timing, tags, child spans and assessments stay intact.
    Run with one writer while refreshing. Each page commits atomically.
    """
    if mlflow.__version__ != "3.17.2":
        raise ValueError("Card refresh is verified with MLflow 3.17.2 only")
    url = make_url(tracking_uri)
    if url.drivername != "sqlite" or not url.database or url.query:
        raise ValueError("Card refresh requires a local sqlite:/// tracking URI")
    database = Path(url.database).resolve()
    if not database.is_file():
        raise ValueError(f"Tracking database does not exist: {database}")
    mode = "ro" if dry_run else "rw"
    with closing(sqlite3.connect(database.as_uri() + f"?mode={mode}", uri=True)) as db:
        experiment = db.execute(
            "SELECT experiment_id FROM experiments "
            "WHERE name=? AND lifecycle_stage='active'",
            (experiment_name,),
        ).fetchone()
        if experiment is None:
            raise ValueError(f"Active experiment {experiment_name!r} does not exist")
        predecessors = _previous_traces(db, experiment[0])
        source_evidence = (
            _source_evidence(db, experiment[0], source_root, tracking_uri, dry_run)
            if source_root is not None
            else {}
        )
        updated = current = 0
        last_id = ""
        while True:
            with db:
                db.execute("BEGIN" if dry_run else "BEGIN IMMEDIATE")
                # The trace-ID index bounds each page without scanning unrelated
                # component spans through the experiment/time index.
                page = db.execute(
                    "SELECT s.trace_id,s.span_id,s.content,location.value "
                    "FROM spans s INDEXED BY sqlite_autoindex_spans_1 "
                    "JOIN trace_tags complete "
                    "ON complete.request_id=s.trace_id "
                    "AND complete.key='align.import_complete' "
                    "AND complete.value='source-record-v1' "
                    "LEFT JOIN trace_tags location ON location.request_id=s.trace_id "
                    "AND location.key='mlflow.trace.spansLocation' "
                    "WHERE s.experiment_id=? AND s.parent_span_id IS NULL "
                    "AND s.trace_id>? ORDER BY s.trace_id LIMIT 100",
                    (experiment[0], last_id),
                ).fetchall()
                if not page:
                    return
                for trace_id, span_id, serialized, location in page:
                    content = json.loads(serialized)
                    previous = None
                    if trace_id in predecessors:
                        prior_roots = db.execute(
                            "SELECT content FROM spans WHERE trace_id=? AND parent_span_id IS NULL",
                            (predecessors[trace_id],),
                        ).fetchall()
                        if len(prior_roots) == 1:
                            previous = _source_record(
                                Span.from_dict(json.loads(prior_roots[0][0]))
                            )
                    source_log, source_record_index = source_evidence.get(
                        trace_id, (None, None)
                    )
                    root = refreshed_root(
                        Span.from_dict(content),
                        previous,
                        source_log=source_log,
                        source_record_index=source_record_index,
                    )
                    if root is None:
                        current += 1
                        continue
                    if location != "TRACKING_STORE":
                        raise ValueError(f"{trace_id}: stored span is archived")
                    updated += 1
                    if dry_run:
                        continue
                    attributes = root.to_dict()["attributes"]
                    for key in (
                        SpanAttributeKey.INPUTS,
                        SpanAttributeKey.OUTPUTS,
                        "align.card_format",
                    ):
                        content["attributes"][key] = attributes[key]
                    payload = json.dumps(content, ensure_ascii=False)
                    # Span upserts also recalculate trace timing. Updating only the
                    # payload preserves sub-millisecond rounding in recorded durations.
                    db.execute(
                        "UPDATE spans SET content=? WHERE trace_id=? AND span_id=?",
                        (payload, trace_id, span_id),
                    )
                    # MLflow uses this generation to invalidate DB-backed payload caches.
                    db.execute(
                        "UPDATE trace_info SET db_payload_generation="
                        "COALESCE(db_payload_generation,0)+1 WHERE request_id=?",
                        (trace_id,),
                    )
                    stored = db.execute(
                        "SELECT content FROM spans WHERE trace_id=? AND span_id=?",
                        (trace_id, span_id),
                    ).fetchone()
                    if stored != (payload,):
                        raise ValueError(
                            f"{trace_id}: card refresh read-back did not match"
                        )
                last_id = page[-1][0]
            yield updated, current
