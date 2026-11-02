"""Bounded, source-linked decision activities from existing producer formats."""

import json
import re
from typing import Any, Optional

from align_utils.open_world import OpenWorldRecord

from .evidence import DecisionLog, _owner_sections

PROCESS_VERSION = "decision-process-v1"
PREVIEW_LIMIT = 2048


def _text(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2)
    )


def _payload(value, source, *, truncated=False):
    text = _text(value)
    return {
        "preview": text[:PREVIEW_LIMIT],
        "truncated": truncated or len(text) > PREVIEW_LIMIT,
        "source": source,
    }


def _json_source(pointer, record_index):
    return {
        "file": "input_output.json",
        "json_pointer": pointer,
        **({"record_index": record_index} if record_index is not None else {}),
    }


def _escape(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def _json_payload(value, pointer, record_index):
    return _payload(value, _json_source(pointer, record_index))


def _log_payload(section, log):
    return _payload(
        section.text,
        section.reference(log.snapshot_sha256),
        truncated=section.truncated,
    )


def _activity(identifier, label, kind="recorded", **values):
    return {"id": identifier, "label": label, "kind": kind, **values}


def _phase(identifier, label, activities, summary=None):
    return {
        "id": identifier,
        "label": label,
        "activities": activities,
        **({"summary": summary} if summary else {}),
    }


def _structured(record, info, record_index, log):
    phases = []
    gaps = []

    def activity(identifier, label, key, value, kind="recorded", parent="/choice_info"):
        return _activity(
            identifier,
            label,
            kind,
            outputs=_json_payload(value, f"{parent}/{_escape(key)}", record_index),
        )

    multi = info.get("multi_agent")
    if isinstance(multi, dict):
        base = "/choice_info/multi_agent"
        proposal = []
        if "proposal" in multi:
            proposal.append(
                activity(
                    "proposal-snapshot",
                    "Recorded proposal",
                    "proposal",
                    multi["proposal"],
                    parent=base,
                )
            )
        if "proposal_validation" in multi:
            proposal.append(
                activity(
                    "proposal-validation",
                    "Proposal validation",
                    "proposal_validation",
                    multi["proposal_validation"],
                    "validation",
                    base,
                )
            )
        phases.append(_phase("proposal", "Proposal", proposal))
        review = []
        if "advice" in multi:
            item = activity(
                "review-snapshot",
                "Recorded reviewer advice",
                "advice",
                multi["advice"],
                "review",
                base,
            )
            if "review_context" in multi:
                item["inputs"] = _json_payload(
                    multi["review_context"], base + "/review_context", record_index
                )
            review.append(item)
        review_attempts = multi.get("review_attempts")
        for index, attempt in enumerate(
            review_attempts if isinstance(review_attempts, list) else []
        ):
            item = activity(
                f"review-attempt-{index}",
                f"Review attempt {index + 1}",
                str(index),
                attempt,
                "review",
                base + "/review_attempts",
            )
            if isinstance(attempt, dict) and isinstance(attempt.get("status"), str):
                item["status"] = attempt["status"]
            review.append(item)
        phases.append(
            _phase(
                "review",
                "Review",
                review,
                str(multi.get("review_status") or "Review status not recorded"),
            )
        )
        final = []
        for key, label in (
            ("final_review_response", "Response to reviewer"),
            ("final_review_response_validation", "Final response validation"),
            ("final_action", "Recorded final choice"),
        ):
            if key in multi:
                item = activity(
                    f"final-{key}",
                    label,
                    key,
                    multi[key],
                    "validation" if "validation" in key else "recorded",
                    base,
                )
                if key == "final_review_response" and "final_review_context" in multi:
                    item["inputs"] = _json_payload(
                        multi["final_review_context"],
                        base + "/final_review_context",
                        record_index,
                    )
                final.append(item)
        phases.append(
            _phase(
                "final", "Final selection", final, str(multi.get("disposition") or "")
            )
        )
        return "reviewer", phases, gaps
    agent = info.get("agent")
    if isinstance(agent, dict):
        observe = []
        for key in ("observation", "history", "choices"):
            if key in agent:
                observe.append(
                    activity(
                        f"observe-{key}",
                        key.capitalize(),
                        key,
                        agent[key],
                        "context",
                        "/choice_info/agent",
                    )
                )
        phases.append(_phase("observe", "Observe", observe))
        proposals = agent.get("proposal_history")
        attempts = [
            activity(
                f"proposal-{index}",
                f"Proposal attempt {index + 1}",
                str(index),
                value,
                parent="/choice_info/agent/proposal_history",
            )
            for index, value in enumerate(
                proposals if isinstance(proposals, list) else []
            )
        ]
        if log is not None:
            attempts.extend(
                _activity(
                    f"prompt-line-{section.first_line}",
                    "Recorded proposal prompt",
                    "context",
                    inputs=_log_payload(section, log),
                )
                for section in log.sections
                if section.title == "AGENT PROPOSE PROMPT"
            )
        phases.append(_phase("proposal", "Propose", attempts))
        feedback = []
        if "feedback" in agent:
            feedback.append(
                activity(
                    "validation-feedback",
                    "Recorded validation feedback",
                    "feedback",
                    agent["feedback"],
                    "validation",
                    "/choice_info/agent",
                )
            )
        phases.append(_phase("validation", "Validate", feedback))
        gaps.append(
            "Proposal attempts do not establish the total model-call count; parameter completion can make additional calls."
        )
        return "langgraph", phases, gaps
    timings = record.choice_info.per_step_timing_stats or []
    if timings:
        owners = _owner_sections(record, log) if log else {}
        for index, timing in enumerate(sorted(timings, key=lambda item: item.step_num)):
            short = timing.step.rsplit(".", 1)[-1]
            activities = [
                activity(
                    f"component-{index}",
                    short,
                    "per_step_timing_stats",
                    info.get("per_step_timing_stats"),
                    "component",
                )
            ]
            for section in owners.get(short, []):
                payload = _log_payload(section, log)
                if section.title == "cache":
                    match = re.match(r"Cache (hit|miss)", section.text)
                    activities.append(
                        _activity(
                            f"cache-line-{section.first_line}",
                            "Application cache",
                            "cache",
                            status=match[1] if match else "unknown",
                            outputs=payload,
                        )
                    )
                else:
                    activities.append(
                        _activity(
                            f"section-line-{section.first_line}",
                            section.title,
                            "recorded",
                            **{
                                "inputs"
                                if "PROMPT" in section.title
                                else "outputs": payload
                            },
                        )
                    )
            for key, value in info.items():
                explicit = (
                    key in {short, timing.step}
                    or isinstance(value, dict)
                    and value.get("source") in (short, timing.step)
                )
                selected = (
                    explicit
                    or ("Regression" in short and key == "predicted_kdma_values")
                    or (short == "ICLADMComponent" and key == "icl_example_responses")
                )
                if selected:
                    activities.append(
                        activity(
                            f"component-{index}-{key}",
                            key.replace("_", " ")
                            if explicit
                            else f"Final pipeline snapshot: {key.replace('_', ' ')}",
                            key,
                            value,
                            "recorded" if explicit else "context",
                        )
                    )
            phases.append(_phase(f"component-{index}", short, activities))
        return "pipeline", phases, gaps
    if isinstance(info.get("langchain_agent"), dict):
        phases.append(
            _phase(
                "decision",
                "Decision agent",
                [
                    activity(
                        "decision-metadata",
                        "Recorded decision metadata",
                        "langchain_agent",
                        info["langchain_agent"],
                    )
                ],
            )
        )
        return "langchain", phases, gaps
    if info:
        phases.append(
            _phase(
                "recorded",
                "Recorded decision evidence",
                [
                    activity(f"metadata-{key}", str(key), key, value)
                    for key, value in info.items()
                ],
            )
        )
    return "unknown", phases, gaps


def _correlated(phases, log, gaps):
    by_phase = {phase["id"]: phase for phase in phases}
    current = "proposal"
    calls = {}
    previous_event = previous_key = None
    for event in log.events:
        if event.role == "REVIEWER":
            current = "review"
        elif event.role == "ENVIRONMENT":
            current = "execution"
        elif event.role == "DECISION":
            if (
                event.direction == "FINAL SELECTION"
                or "phase=final" in event.section.text
            ):
                current = "final"
            elif (
                event.direction == "PROPOSAL" or "phase=proposal" in event.section.text
            ):
                current = "proposal"
            elif event.direction == "INPUT" and current in {"review", "execution"}:
                current = "final"
        call = event.fields.get("call")
        tool = event.fields.get("tool_call")
        model = event.role in {"DECISION", "REVIEWER"} and event.direction in {
            "INPUT",
            "OUTPUT",
            "FAILED",
            "FAILURE",
        }
        tool_event = event.role == "TOOL" and event.direction in {"INPUT", "OUTPUT"}
        identity = (event.role, event.direction, dict(event.fields))
        correlated = (
            model
            and call is not None
            or tool_event
            and call is not None
            and tool is not None
        )
        if correlated:
            key = (event.role, call, tool if tool_event else None)
        elif not (model or tool_event) and event.detail and previous_event == identity:
            key = previous_key
        else:
            key = ("event", event.section.first_line)
        previous_event = None if event.detail else identity
        previous_key = key
        phase = by_phase.get(current)
        if phase is None:
            phase = _phase(
                current,
                {
                    "proposal": "Proposal",
                    "review": "Review",
                    "final": "Final selection",
                    "execution": "Environment execution",
                }.get(current, current),
                [],
            )
            phases.append(phase)
            by_phase[current] = phase
        item = calls.get(key)
        if item is None:
            kind = (
                "model"
                if model
                else "tool"
                if tool_event
                else "action"
                if event.role == "ENVIRONMENT"
                else "validation"
                if "VALIDAT" in event.direction
                else "context"
                if "CONTEXT" in event.direction
                else "recorded"
            )
            name = re.search(r"\bname=([^\s|]+)", event.section.text)
            label = (
                f"{event.role.title()} model call {call}"
                if model
                else f"Tool {name[1] if name else tool}"
                if tool_event
                else f"{event.role.title()}: {event.direction.lower()}"
            )
            if (model or tool_event) and not correlated:
                kind = "recorded"
                label = f"Uncorrelated {event.role.lower()} {event.direction.lower()}"
                gaps.append(
                    f"{label.capitalize()}: stable call identifiers were not captured; input/output pairing is unresolved."
                )
            item = _activity(
                f"event-{event.section.first_line}",
                label,
                kind,
                source=event.section.reference(log.snapshot_sha256),
            )
            phase["activities"].append(item)
            calls[key] = item
        if event.detail:
            item["inputs" if event.direction == "INPUT" else "outputs"] = _log_payload(
                event.section, log
            )
        elif not (model or tool_event):
            item["outputs"] = _log_payload(event.section, log)
        if event.direction in {"FAILED", "FAILURE", "REJECTED", "INVALID RESPONSE"}:
            item["status"] = "failed"
        elif event.direction == "EXECUTED":
            item["status"] = "executed"
        elif model or tool_event:
            if event.detail and item.get("status") != "failed":
                item["status"] = "captured"
            else:
                item.setdefault("status", "summary_only")
    for item in calls.values():
        if item["kind"] in {"model", "tool"}:
            for key in ("inputs", "outputs"):
                if key not in item:
                    gaps.append(
                        f"{item['label']}: {key[:-1]} payload was not captured."
                    )
    model_calls = {
        event.fields.get("call")
        for event in log.events
        if event.role in {"DECISION", "REVIEWER"} and event.direction == "INPUT"
    }
    for event in log.events:
        if event.role == "TOOL" and event.fields.get("call") not in model_calls:
            gaps.append(
                f"Model call {event.fields.get('call')}: no request boundary is present in this decision's captured events."
            )


def _legacy(phases, log, gaps):
    if not log or not log.excerpt:
        return
    sections = log.sections
    activities = []
    for section in sections:
        if (
            section.title.startswith("AGENT")
            or "TOOLKIT" in section.title
            or "KALEIDO" in section.title
        ):
            activities.append(
                _activity(
                    f"legacy-{section.first_line}",
                    section.title,
                    "context"
                    if "PROMPT" in section.title or "CONTEXT" in section.title
                    else "recorded",
                    **{
                        "inputs"
                        if "PROMPT" in section.title
                        else "outputs": _log_payload(section, log)
                    },
                )
            )
    if activities:
        phases.append(_phase("logged", "Recorded agent activity", activities))
    gaps.append(
        "Exact per-call request windows and generic tool results are not comprehensively captured by this producer."
    )


def _action_label(value):
    if not isinstance(value, dict):
        return _text(value)[:256]
    return (
        " ".join(
            str(value[key]) for key in ("action_type", "character_id") if value.get(key)
        )
        or "Action not recorded"
    )


def _review(info):
    multi = info.get("multi_agent")
    if not isinstance(multi, dict):
        return None
    result = {}
    for key in ("proposal", "final_action"):
        if key in multi:
            result[key] = _action_label(multi[key])
    advice = multi.get("advice")
    if isinstance(advice, dict) and isinstance(advice.get("recommendation"), str):
        result["reviewer_recommendation"] = advice["recommendation"]
    if isinstance(multi.get("review_response_contract"), str):
        result["review_mode"] = multi["review_response_contract"]
    weighting = multi.get("weighting")
    if isinstance(weighting, dict):
        config = weighting.get("configuration")
        if isinstance(config, dict):
            for key in ("influence", "swap_threshold"):
                if isinstance(config.get(key), (int, float)) and not isinstance(
                    config[key], bool
                ):
                    result[key] = config[key]
        for key in ("confidence", "estimated_improvement", "strength"):
            if isinstance(weighting.get(key), (int, float)) and not isinstance(
                weighting[key], bool
            ):
                result[key] = weighting[key]
        if isinstance(weighting.get("suggestion"), str):
            result["computed_suggestion"] = weighting["suggestion"]
        if (
            isinstance(weighting.get("followed_guidance"), bool)
            or weighting.get("followed_guidance") is None
        ):
            result["followed_guidance"] = weighting.get("followed_guidance")
    return result


def decision_process(
    record: OpenWorldRecord,
    *,
    record_index: Optional[int] = None,
    log: Optional[DecisionLog] = None,
) -> dict[str, Any]:
    raw = record.source.get("choice_info")
    info = raw if isinstance(raw, dict) else {}
    if record.chosen_by_driver:
        producer, phases, gaps = (
            "driver",
            [],
            [
                "The driver selected this action; retained choice metadata may describe an earlier decision."
            ],
        )
    else:
        producer, phases, gaps = _structured(record, info, record_index, log)
    if log and log.events:
        if producer == "unknown":
            producer = "correlated"
        _correlated(phases, log, gaps)
    elif producer == "langchain":
        _legacy(phases, log, gaps)
    if log and not log.events:
        owned = {
            item[key]["source"].get("lines", [None])[0]
            for phase in phases
            for item in phase["activities"]
            for key in ("inputs", "outputs")
            if key in item and item[key]["source"].get("file") == "raw_align_system.log"
        }
        unowned = [
            section for section in log.sections if section.first_line not in owned
        ]
        if not log.sections and log.excerpt:
            unowned = [log.excerpt]
        if unowned:
            phases.append(
                _phase(
                    "unassigned",
                    "Other recorded log sections",
                    [
                        _activity(
                            f"unassigned-{section.first_line}",
                            section.title,
                            "recorded",
                            **{
                                "inputs"
                                if "PROMPT" in section.title
                                else "outputs": _log_payload(section, log)
                            },
                        )
                        for section in unowned
                    ],
                    "Component ownership is unknown; these sections are not attributed to individual calls.",
                )
            )
            gaps.append(
                "Some log sections have no verified component owner and are shown without call attribution."
            )
    if not log or log.status not in {
        "exact_action_sequence",
        "exact_correlated_action_sequence",
    }:
        gaps.append(
            "Decision log association unavailable: "
            + (log.status if log else "not loaded")
            + "."
        )
    if log and log.skipped_rejected_attempts:
        gaps.append(
            f"{log.skipped_rejected_attempts} rejected action attempts were excluded from saved-action evidence."
        )
    phases.append(
        _phase(
            "saved-action",
            "Saved environment action",
            [
                _activity(
                    "saved-action",
                    _action_label(record.source["output"]["action"]),
                    "action",
                    outputs=_json_payload(
                        record.source["output"]["action"],
                        "/output/action",
                        record_index,
                    ),
                    status="recorded",
                )
            ],
        )
    )
    review = _review(info) if not record.chosen_by_driver else None
    return {
        "schema_version": PROCESS_VERSION,
        "producer": producer,
        "phases": phases,
        "gaps": list(dict.fromkeys(gaps)),
        **({"review": review} if review else {}),
    }
