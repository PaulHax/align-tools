"""Summarize observations without using the decision made from them."""

import json
from typing import Any, Optional

from align_utils.open_world import OpenWorldRecord, starts_episode

_MISSING = object()


def _value(value: Any) -> str:
    if value is _MISSING:
        return "not recorded"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _label(key: str) -> str:
    return {
        "avpu": "AVPU",
        "unstructured": "Description",
        "scene_id": "Scene",
        "tag": "Triage tag",
    }.get(
        key,
        key.replace("_", " ") if key.isupper() else key.replace("_", " ").capitalize(),
    )


def _changes(previous: dict, current: dict, prefix: str = "") -> list[str]:
    result = []
    for key in dict.fromkeys([*current, *previous]):
        old, new = previous.get(key, _MISSING), current.get(key, _MISSING)
        if old == new:
            continue
        label = f"{prefix}{_label(key)}"
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(_changes(old, new, f"{label} / "))
        elif isinstance(new, dict) and old is _MISSING:
            result.extend(_changes({}, new, f"{label} / "))
        else:
            result.append(f"- {label}: {_value(old)} → {_value(new)}")
    return result


def _patient_label(patient: dict) -> str:
    identifier, name = patient["id"], patient.get("name")
    return f"{identifier} ({name})" if name and name != identifier else identifier


def _patient_snapshot(patient: dict) -> str:
    sections = [f"**{_patient_label(patient)}**"]
    if patient.get("unstructured"):
        sections.append(patient["unstructured"])
    details = [
        f"- {_label(key)}: {_value(patient[key])}"
        for key in ("nearby", "tag", "vitals")
        if key in patient and patient[key] is not None and patient[key] != {}
    ]
    if details:
        sections.append("\n".join(details))
    return "\n\n".join(sections)


def _patient_updates(previous: list[dict], current: list[dict]) -> list[str]:
    old = {patient["id"]: patient for patient in previous}
    new = {patient["id"]: patient for patient in current}
    if len(old) != len(previous) or len(new) != len(current):
        return [_patient_snapshot(p) for p in current if p.get("unseen") is not True]
    updates = []
    for identifier in dict.fromkeys([*new, *old]):
        before, after = old.get(identifier), new.get(identifier)
        was_visible = before is not None and before.get("unseen") is not True
        is_visible = after is not None and after.get("unseen") is not True
        if not is_visible:
            if was_visible:
                status = (
                    "No longer visible."
                    if after is not None
                    else "Not present in this observation."
                )
                updates.append(f"**{_patient_label(before)}**\n\n{status}")
            continue
        if not was_visible:
            updates.append(_patient_snapshot(after))
            continue
        details = []
        if after.get("unstructured", _MISSING) != before.get("unstructured", _MISSING):
            details.append(
                after.get("unstructured") or "Description no longer recorded."
            )
        ignored = {"id", "unseen", "unstructured"}
        fields = _changes(
            {k: v for k, v in before.items() if k not in ignored},
            {k: v for k, v in after.items() if k not in ignored},
        )
        if fields:
            details.append("\n".join(fields))
        if details:
            updates.append(f"**{_patient_label(after)}**\n\n" + "\n\n".join(details))
    return updates


def _supply_updates(previous: list[dict], current: list[dict]) -> list[str]:
    if all(isinstance(item.get("type"), str) for item in [*previous, *current]):
        old = {
            item["type"]: {k: v for k, v in item.items() if k != "type"}
            for item in previous
        }
        new = {
            item["type"]: {k: v for k, v in item.items() if k != "type"}
            for item in current
        }
        if len(old) == len(previous) and len(new) == len(current):
            return _changes(old, new)
    return [f"{_value(previous)} → {_value(current)}"] if previous != current else []


def input_summary(
    record: OpenWorldRecord, previous: Optional[OpenWorldRecord] = None
) -> str:
    if previous is not None and starts_episode(previous, record):
        previous = None
    source = record.source["input"]
    state = source["full_state"]
    narrative = record.input.state or record.input.full_state.unstructured
    prior_narrative = (
        (previous.input.state or previous.input.full_state.unstructured)
        if previous
        else None
    )
    situation = (
        "Unchanged."
        if previous is not None and narrative == prior_narrative
        else narrative or "No situation text recorded."
    )
    sections = [f"**Situation**\n\n{situation}"]
    patients = state.get("characters", [])
    if previous is None:
        visible = [
            _patient_snapshot(p) for p in patients if p.get("unseen") is not True
        ]
        if visible:
            sections.append("**Patients**\n\n" + "\n\n".join(visible))
        return "\n\n".join(sections)

    prior_source = previous.source["input"]
    prior_state = prior_source["full_state"]
    updates = _patient_updates(prior_state.get("characters", []), patients)
    if updates:
        sections.append(
            "**Patient updates since previous input**\n\n" + "\n\n".join(updates)
        )
    supplies = _supply_updates(
        prior_state.get("supplies", []), state.get("supplies", [])
    )
    if supplies:
        sections.append("**Supplies**\n\n" + "\n".join(supplies))
    ignored_state = {"characters", "supplies", "elapsed_time", "unstructured"}
    if record.input.state or previous.input.state:
        if (
            state.get("unstructured") != narrative
            or prior_state.get("unstructured") != prior_narrative
        ):
            ignored_state.remove("unstructured")
    other_state = _changes(
        {k: v for k, v in prior_state.items() if k not in ignored_state},
        {k: v for k, v in state.items() if k not in ignored_state},
    )
    if other_state:
        sections.append("**Other observation changes**\n\n" + "\n".join(other_state))
    other_inputs = _changes(
        {k: v for k, v in prior_source.items() if k not in {"state", "full_state"}},
        {k: v for k, v in source.items() if k not in {"state", "full_state"}},
    )
    if other_inputs:
        sections.append("**Other input changes**\n\n" + "\n".join(other_inputs))
    if len(sections) == 1:
        sections.append("No other changes to summarize.")
    return "\n\n".join(sections)
