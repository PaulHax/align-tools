"""Describe open-world runs as MLflow traces without touching MLflow.

Each record becomes one trace (a turn) whose session is its episode, mirroring
how align-system calls the ADM once per step. The ADM pipeline components
reported in ``choice_info.per_step_timing_stats`` become child spans.

Records carry no wall-clock times, so spans are laid out on a synthetic
timeline: the run's start plus the cumulative component time of earlier
steps. Records are only ever appended, so a step's timestamps do not change
as the run grows.
"""

import hashlib
import re
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Dict, Mapping, Optional, Tuple

from align_utils.open_world import Episode, OpenWorldRecord, OpenWorldRun
from mlflow.entities import SpanType

from .cards import CARD_FORMAT, card_inputs, card_outputs
from .evidence import DecisionLog, component_evidence

SESSION_METADATA_KEY = "mlflow.trace.session"
SOURCE_RUN_METADATA_KEY = "mlflow.sourceRun"
RUN_KEY_TAG = "align.run_key"
RECORD_INDEX_TAG = "align.record_index"
SCORE_ASSESSMENT = "ta3_session_alignment_score"

# Components that report zero seconds still need a visible span.
_MIN_SPAN_NS = 1_000
# Keeps start times strictly increasing so the session view orders turns.
_STEP_GAP_NS = 1_000_000
_URL_UNSAFE = re.compile(r"[#?/]")


@dataclass(frozen=True)
class SpanSpec:
    name: str
    span_type: str
    start_ns: int
    end_ns: int
    inputs: Any = None
    outputs: Any = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepTrace:
    record_index: int
    session_id: str
    root: SpanSpec
    children: Tuple[SpanSpec, ...]
    tags: Mapping[str, str]


@dataclass(frozen=True)
class SessionScore:
    """TA3's score for an episode, logged as session feedback on its first step."""

    session_id: str
    first_record_index: int
    value: float
    rationale: str
    alignment_target_id: Optional[str]


def run_label(run: OpenWorldRun) -> str:
    """Readable run name built from the run itself, so it does not depend on
    which directory was synced or where the run was copied."""
    adm = run.config.adm.name if run.config else None
    return " ".join(filter(None, [adm, run.adm_profile, run.path.name]))


def run_key(run: OpenWorldRun) -> str:
    """Identity used to find a run's logged steps. Includes the TA3 session
    name when present, which is unique per live run."""
    identity = f"{run_label(run)}\n{run.username or ''}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def session_id(episode: Episode, label: str, key: str) -> str:
    target = episode.alignment_target_id or "unaligned"
    raw = (
        f"{episode.scenario_id} | {target} | {label} | "
        f"episode {episode.index + 1} | {key}"
    )
    # MLflow's session page puts the id in its URL without escaping these.
    return _URL_UNSAFE.sub("-", raw)


def component_spans(
    record: OpenWorldRecord,
    start_ns: int,
    record_index: int = 0,
    log: Optional[DecisionLog] = None,
) -> Tuple[SpanSpec, ...]:
    """Child spans for the pipeline components that chose this action.

    Driver-chosen steps get none: their choice_info may describe the previous
    step.
    """
    if record.chosen_by_driver:
        return ()
    timings = sorted(
        record.choice_info.per_step_timing_stats or [],
        key=lambda timing: timing.step_num,
    )
    durations = [max(round(timing.elapsed_s * 1e9), _MIN_SPAN_NS) for timing in timings]
    starts = accumulate(durations, initial=start_ns)
    return tuple(
        SpanSpec(
            name=timing.step.rsplit(".", 1)[-1],
            span_type=SpanType.CHAIN,
            start_ns=start,
            end_ns=start + duration,
            inputs=evidence[0],
            outputs=evidence[1],
            attributes={"component": timing.step, **evidence[2]},
        )
        for timing, start, duration in zip(timings, starts, durations)
        for evidence in [component_evidence(record, timing.step, record_index, log)]
    )


def step_duration_ns(record: OpenWorldRecord) -> int:
    if record.chosen_by_driver:
        return _MIN_SPAN_NS
    timings = record.choice_info.per_step_timing_stats or []
    return (
        sum(max(round(t.elapsed_s * 1e9), _MIN_SPAN_NS) for t in timings)
        or _MIN_SPAN_NS
    )


def step_traces(
    run: OpenWorldRun,
    run_start_ns: int,
    logs: Optional[Tuple[DecisionLog, ...]] = None,
) -> Tuple[StepTrace, ...]:
    if logs is not None and len(logs) != len(run.records):
        raise ValueError("Decision log count does not match source records")
    label, key = run_label(run), run_key(run)
    placed = [
        (episode, step, episode.first_record_index + step - 1, record)
        for episode in run.episodes
        for step, record in enumerate(episode.records, start=1)
    ]
    starts = accumulate(
        (step_duration_ns(record) + _STEP_GAP_NS for *_, record in placed),
        initial=run_start_ns,
    )
    return tuple(
        StepTrace(
            record_index=record_index,
            session_id=session_id(episode, label, key),
            root=_root_span(
                record, step, start, episode.records[step - 2] if step > 1 else None
            ),
            children=component_spans(
                record,
                start,
                record_index,
                logs[record_index] if logs is not None else None,
            ),
            tags=_step_tags(run, label, key, episode, step, record_index, record),
        )
        for (episode, step, record_index, record), start in zip(placed, starts)
    )


def _root_span(
    record: OpenWorldRecord,
    step: int,
    start_ns: int,
    previous: Optional[OpenWorldRecord] = None,
) -> SpanSpec:
    action = record.output.action
    return SpanSpec(
        name=f"{step:02d} {action.action_type} {action.character_id or ''}".rstrip(),
        span_type=SpanType.AGENT,
        start_ns=start_ns,
        end_ns=start_ns + step_duration_ns(record),
        inputs=card_inputs(record, previous),
        outputs=card_outputs(record),
        attributes={
            "align.choice_info_applies_to_action": not record.chosen_by_driver,
            "align.card_format": CARD_FORMAT,
        },
    )


def _step_tags(
    run: OpenWorldRun,
    label: str,
    key: str,
    episode: Episode,
    step: int,
    record_index: int,
    record: OpenWorldRecord,
) -> Dict[str, str]:
    action = record.output.action
    optional = {
        "alignment_target_id": record.input.alignment_target_id,
        "scene_id": record.input.full_state.meta_info.scene_id,
        "character_id": action.character_id,
        "action_detail": " ".join(str(v) for v in (action.parameters or {}).values()),
        "adm": run.config.adm.name if run.config else None,
        "llm": run.config.adm.llm_backbone if run.config else None,
        "align.source_version": run.version,
    }
    return {
        RUN_KEY_TAG: key,
        RECORD_INDEX_TAG: str(record_index),
        "run": label,
        "episode": str(episode.index + 1),
        "step": str(step),
        "scenario_id": record.input.scenario_id,
        "action_type": action.action_type,
        "chosen_by": "driver" if record.chosen_by_driver else "adm",
        **{name: value for name, value in optional.items() if value},
    }


def session_scores(run: OpenWorldRun) -> Tuple[SessionScore, ...]:
    label, key = run_label(run), run_key(run)
    return tuple(
        SessionScore(
            session_id=session_id(episode, label, key),
            first_record_index=episode.first_record_index,
            value=score,
            rationale=outcome.text,
            alignment_target_id=outcome.alignment_target_id,
        )
        for episode in run.episodes
        if (outcome := episode.outcome) is not None
        and (score := outcome.session_alignment_score) is not None
    )
