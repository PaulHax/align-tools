"""Refresh imported cards in an MLflow 3.16.1 SQLite store."""

import copy
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterator, Optional

import mlflow
from align_utils.open_world import OpenWorldRecord
from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey
from sqlalchemy.engine import make_url

from .cards import CARD_FORMAT, card_inputs, card_outputs


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
    root: Span, previous: Optional[OpenWorldRecord] = None
) -> Optional[Span]:
    record = _source_record(root)
    card_format = root.get_attribute("align.card_format")
    inputs, outputs = card_inputs(record, previous), card_outputs(record)
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


def _previous_traces(db: sqlite3.Connection, experiment_id: int) -> dict[str, str]:
    rows = db.execute(
        "SELECT t.request_id,session.value,run.value,step.value,record.value "
        "FROM trace_info t "
        "JOIN trace_request_metadata session ON session.request_id=t.request_id AND session.key='mlflow.trace.session' "
        "JOIN trace_tags run ON run.request_id=t.request_id AND run.key='align.run_key' "
        "JOIN trace_tags step ON step.request_id=t.request_id AND step.key='step' "
        "JOIN trace_tags record ON record.request_id=t.request_id AND record.key='align.record_index' "
        "JOIN trace_tags complete ON complete.request_id=t.request_id AND complete.key='align.import_complete' "
        "AND complete.value='source-record-v1' WHERE t.experiment_id=?",
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
    tracking_uri: str, experiment_name: str, dry_run: bool = False
) -> Iterator[tuple[int, int]]:
    """Yield cumulative (updated, already current) counts after each page.

    Uses the 3.16.1 SQLite schema. Only root payloads and their cache generations
    are updated; IDs, timing, tags, child spans and assessments stay intact.
    Run with one writer while refreshing. Each page commits atomically.
    """
    if mlflow.__version__ != "3.16.1":
        raise ValueError("Card refresh is verified with MLflow 3.16.1 only")
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
                    root = refreshed_root(Span.from_dict(content), previous)
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
