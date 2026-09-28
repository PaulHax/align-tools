"""Readable session cards derived from the recorded state and chosen action."""

import json
import re
from typing import Any, Optional

from align_utils.open_world import OpenWorldRecord

from .observations import input_summary

CARD_FORMAT = "situation-action-v1"


def card_inputs(
    record: OpenWorldRecord, previous: Optional[OpenWorldRecord] = None
) -> dict[str, Any]:
    return {"input": input_summary(record, previous), "source": record.source["input"]}


def card_outputs(record: OpenWorldRecord) -> dict[str, Any]:
    return {
        "response": output_summary(record),
        "source": {
            key: value for key, value in record.source.items() if key != "input"
        },
    }


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
    details = (
        [action.character_id]
        if action.character_id
        and not re.search(
            rf"(?<!\w){re.escape(action.character_id)}(?!\w)", description
        )
        else []
    )
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
