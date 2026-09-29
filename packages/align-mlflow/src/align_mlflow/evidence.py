"""Recorded decision evidence, with conservative attribution to pipeline steps."""

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from align_utils.open_world import OpenWorldRecord, RAW_LOG_FILE

EVIDENCE_VERSION = "choice-info-log-v1"
_ACTION = re.compile(r"^\[bold\]\*ACTION BEING TAKEN\*\[/bold\][ \t]*$", re.M)
_HEADER = re.compile(r"^\[bold\]\*(.*?)\*\[/bold\]$")
_CACHE = re.compile(r"^Cache (hit|miss) for `([^`]+)`")
_REGRESSION = {
    "DirectRegressionADMComponent",
    "OWDirectRegressionADMComponent",
    "ComparativeRegressionADMComponent",
}
_ACTION_SELECTION = {"OWChoiceToActionADMComponent", "OutlinesBaselineADMComponent"}
_RELEVANCE = {"PredictMostRelevantADMComponent", "BertRelevanceADMComponent"}
_PARAMETER = {
    "OWActionParameterCompletionADMComponent",
    "ActionParameterCompletionADMComponent",
}
_CACHE_COMPONENTS = {
    "direct_regression_adm_component": _REGRESSION
    - {"ComparativeRegressionADMComponent"},
    "comparative_regression_adm_component": {"ComparativeRegressionADMComponent"},
    "outlines_baseline_adm_component": _ACTION_SELECTION,
    "ow_action_parameter_completion_adm_component": _PARAMETER,
    "action_parameter_completion_adm_component": _PARAMETER,
}


@dataclass(frozen=True)
class Section:
    title: str
    text: str
    first_line: int
    last_line: int

    def reference(self) -> dict:
        return {
            "file": RAW_LOG_FILE,
            "lines": [self.first_line, self.last_line],
            "sha256": hashlib.sha256(self.text.encode()).hexdigest(),
            "title": self.title,
        }


@dataclass(frozen=True)
class DecisionLog:
    status: str
    sections: tuple[Section, ...] = ()


def _sections(text: str, first_line: int) -> tuple[Section, ...]:
    lines = text.splitlines()
    starts = []
    for i, line in enumerate(lines):
        if _CACHE.match(line):
            starts.append((i, "cache"))
        elif header := _HEADER.match(line):
            starts.append((i, header[1]))
        elif line.startswith("Max vote keys:"):
            # The preceding vector uses pipeline candidate order, not action indices.
            vector = i - 1 if i and lines[i - 1].startswith("{") else i
            starts.append((vector, "alignment_selection"))
    sections = []
    for n, (start, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        if title == "cache":
            end = start + 1
        sections.append(
            Section(
                title,
                "\n".join(lines[start:end]).rstrip(),
                first_line + start,
                first_line + end - 1,
            )
        )
    return tuple(sections)


def read_decision_logs(
    path: Path, records: Sequence[OpenWorldRecord]
) -> tuple[DecisionLog, ...]:
    """Match a log prefix against complete action objects before using any excerpts.

    A growing log can lag the JSON snapshot. Unmatched trailing records get no
    excerpts. Any mismatched or malformed complete action invalidates the join.
    """
    try:
        text = (path / RAW_LOG_FILE).read_text()
    except FileNotFoundError:
        return tuple(DecisionLog("missing_log") for _ in records)
    markers = list(_ACTION.finditer(text))
    empty = "no_action_markers" if not markers else "action_not_yet_logged"
    result = [DecisionLog(empty) for _ in records]
    newlines = [m.start() for m in re.finditer("\n", text)]
    decoder = json.JSONDecoder()
    previous_end = 0
    for index, marker in enumerate(markers[: len(records)]):
        start = marker.end()
        while start < len(text) and text[start].isspace():
            start += 1
        try:
            action, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            # A final action may still be in flight. Earlier matched records stand.
            if index == len(markers) - 1 and not text.endswith("\n"):
                break
            return tuple(
                DecisionLog(f"invalid_action_at_record_{index}") for _ in records
            )
        if action != records[index].source["output"]["action"]:
            return tuple(
                DecisionLog(f"action_mismatch_at_record_{index}") for _ in records
            )
        begin_line = bisect_right(newlines, previous_end - 1) + 1
        result[index] = DecisionLog(
            "exact_action_sequence",
            _sections(text[previous_end : marker.start()], begin_line),
        )
        previous_end = end
    return tuple(result)


def _owner_sections(
    record: OpenWorldRecord, log: DecisionLog
) -> dict[str, list[Section]]:
    components = [
        t.step.rsplit(".", 1)[-1]
        for t in record.choice_info.per_step_timing_stats or []
    ]

    def unique(candidates):
        found = [name for name in components if name in candidates]
        return found[0] if len(found) == 1 else None

    info = record.source.get("choice_info") or {}
    alignment = info.get("alignment_info") or {}
    alignment_source = alignment.get("source") if isinstance(alignment, dict) else None
    alignment_owner = (
        alignment_source.rsplit(".", 1)[-1] if isinstance(alignment_source, str) else ""
    )
    assigned: dict[str, list[Section]] = {}
    active = None
    for section in log.sections:
        title = section.title
        owner = None
        if title == "cache":
            match = _CACHE.match(section.text)
            active = (
                unique(_CACHE_COMPONENTS.get(match[2], set()))
                if match is not None
                else None
            )
            owner = active
        elif title == "alignment_selection":
            owner = unique({alignment_owner})
        elif "FOLLOWUP PROMPT" in title or "FOLLOWUP RESPONSE" in title:
            owner = unique(_PARAMETER)
        elif title == "TAG ADJUSTMENT":
            owner = unique({"OWTaggingAdjustmentADMComponent"})
        elif title == "VOTES":
            owner = active if active in _ACTION_SELECTION else None
        elif title.startswith(
            ("MOST RELEVANT ATTRIBUTE PREDICTION", "RELEVANCE PREDICTION")
        ):
            owner = unique({"PredictMostRelevantADMComponent"})
        elif "PREDICTION DIALOG PROMPT" in title or "PREDICTION RESPONSE" in title:
            candidates = _REGRESSION | (
                _ACTION_SELECTION if title.startswith("KDMA SCORE") else set()
            )
            owner = active if active in candidates else unique(candidates)
        if owner:
            assigned.setdefault(owner, []).append(section)
    return assigned


def _block(section: Section) -> str:
    # A fence longer than any source run prevents logged text from closing it.
    fence = "`" * max(
        3, 1 + max((len(m[0]) for m in re.finditer(r"`+", section.text)), default=0)
    )
    return f"**{section.title}** ({RAW_LOG_FILE}:{section.first_line}-{section.last_line})\n\n{fence}text\n{section.text}\n{fence}"


def component_evidence(
    record: OpenWorldRecord,
    component: str,
    record_index: int,
    log: Optional[DecisionLog] = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Display final JSON context separately from attributed log emissions."""
    short = component.rsplit(".", 1)[-1]
    info = record.source.get("choice_info") or {}
    log = log or DecisionLog("not_loaded")
    sections = _owner_sections(record, log).get(short, [])
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    fields = []
    snapshot = {}
    for key, value in info.items():
        explicitly_owned = key in {component, short} or (
            isinstance(value, dict)
            and (value.get("source") == component or value.get("source") == short)
        )
        selected = (
            explicitly_owned
            or (short in _REGRESSION and key == "predicted_kdma_values")
            or (short in _RELEVANCE and key == "predicted_relevance")
            or (short == "ICLADMComponent" and key == "icl_example_responses")
            or (short == "PopulateChoiceInfo" and key != "per_step_timing_stats")
        )
        if selected:
            snapshot[key] = value
            fields.append(f"choice_info.{key}")
    if snapshot:
        outputs["recorded_choice_info"] = snapshot
    alignment = info.get("alignment_info")
    if isinstance(alignment, dict) and alignment.get("source") in (component, short):
        inputs["alignment_target_id"] = record.source["input"].get(
            "alignment_target_id"
        )
        fields.append("input.alignment_target_id")
        for key in ("predicted_kdma_values", "predicted_relevance"):
            if key in info:
                inputs[f"recorded_{key}"] = info[key]
                fields.append(f"choice_info.{key}")
    if short == "WorldStateTrackerADMComponent":
        inputs["observed_state"] = record.source["input"]["full_state"]
        fields.append("input.full_state")
    elif short in {"OWFormatChoicesADMComponent", "ITMFormatChoicesADMComponent"}:
        inputs["available_actions"] = record.source["input"].get("choices", [])
        fields.append("input.choices")
    elif short == "JustificationFromReasonings":
        outputs["final_action_justification"] = record.source["output"]["action"].get(
            "justification"
        )
        fields.append("output.action.justification")
    elif short in _ACTION_SELECTION | _PARAMETER | {
        "OWTaggingAdjustmentADMComponent",
        "EnsureChosenActionADMComponent",
    }:
        outputs["final_action"] = record.source["output"]["action"]
        fields.append("output.action")
    prompts = [section for section in sections if "PROMPT" in section.title]
    responses = [
        section
        for section in sections
        if "PROMPT" not in section.title and section.title != "cache"
    ]
    if prompts:
        inputs = {"input": "\n\n".join(map(_block, prompts)), **inputs}
    if responses:
        outputs = {"response": "\n\n".join(map(_block, responses)), **outputs}
    cache_status = [
        match[1]
        for section in sections
        if section.title == "cache"
        and (match := _CACHE.match(section.text)) is not None
    ]
    attributes = {
        "align.component_evidence": EVIDENCE_VERSION,
        "align.log_match": log.status,
        "align.evidence_sources": [
            *(
                [
                    {
                        "file": "input_output.json",
                        "record_index": record_index,
                        "fields": fields,
                    }
                ]
                if fields
                else []
            ),
            *(section.reference() for section in sections),
        ],
    }
    if cache_status:
        attributes["align.cache_status"] = cache_status
        if all(status == "hit" for status in cache_status) and not prompts:
            inputs = {
                "input": "Cached execution. No fresh prompt was recorded for this component in this decision.",
                **inputs,
            }
    return inputs or None, outputs or None, attributes
