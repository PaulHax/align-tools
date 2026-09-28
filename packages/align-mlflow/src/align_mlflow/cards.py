"""Readable session cards derived from the recorded state and chosen action."""

import json
from typing import Any

from align_utils.open_world import OpenWorldRecord


def input_summary(record: OpenWorldRecord) -> str:
    state = record.input.full_state
    context = [f"**Elapsed:** {state.elapsed_time:g}s"]
    if state.meta_info.scene_id:
        context.insert(0, f"**Scene:** {state.meta_info.scene_id}")
    sections = [" · ".join(context)]
    patient = next(
        (
            character
            for character in state.characters
            if character.id == record.output.action.character_id
        ),
        None,
    )
    if patient is not None:
        sections.append(
            f"**Patient context: {patient.name or patient.id}**\n\n"
            f"{patient.unstructured or 'No patient description recorded.'}"
        )
    narrative = record.input.state or state.unstructured
    sections.append(f"**Situation**\n\n{narrative or 'No situation text recorded.'}")
    return "\n\n".join(sections)


def _display_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def output_summary(record: OpenWorldRecord) -> str:
    action = record.output.action
    description = action.unstructured.strip() or next(
        (
            choice.unstructured.strip()
            for choice in record.input.choices
            if choice.action_id == action.action_id and choice.unstructured.strip()
        ),
        action.action_type.replace("_", " ").capitalize(),
    )
    details = [action.character_id] if action.character_id else []
    details.extend(
        f"{name}: {_display_value(value)}"
        for name, value in (action.parameters or {}).items()
    )
    sections = [description]
    if details:
        sections.append(" · ".join(details))
    if record.chosen_by_driver:
        sections.append("**Chosen by the driver.**")
    sections.append(
        "**Justification**\n\n" + (action.justification or "No justification recorded.")
    )
    return "\n\n".join(sections)
