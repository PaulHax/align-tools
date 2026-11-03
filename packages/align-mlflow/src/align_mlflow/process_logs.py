"""Read correlated agent presentation without inferring uncaptured exchanges."""

import hashlib
import json
import re
from typing import Optional, Sequence

from align_utils.open_world import OpenWorldRecord, split_episodes

from .evidence import DecisionLog, LogEvent, Section

_EVENT = re.compile(
    r"^=== ([A-Z][A-Z/ ]*?) \| ([A-Z][A-Z ]*?)((?: \| [^\r\n]*)?) ===\r?$", re.M
)
_ACTION_FIELDS = {
    "action_id",
    "action_type",
    "character_id",
    "parameters",
    "unstructured",
    "justification",
    "intent_action",
}
# The correlated producer explicitly omits evaluation labels from its action view.
_EVALUATION_FIELDS = {"kdma_association"}
_LIMIT = 2048


def _render_value(value):
    if not isinstance(value, str):
        return json.dumps(value, indent=2, ensure_ascii=False)
    lines = []
    for line in value.split("\n"):
        try:
            decoded = json.loads(line)
        except (ValueError, RecursionError):
            decoded = None
        lines.append(
            json.dumps(decoded, indent=2, ensure_ascii=False)
            if isinstance(decoded, (dict, list))
            else line
        )
    rendered = "\n".join(lines)
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        rendered = json.dumps(decoded, indent=2, ensure_ascii=False)
    return "".join(
        f"\\x{ord(char):02x}"
        if (ord(char) < 32 and char not in "\n\t") or 127 <= ord(char) <= 159
        else char
        for char in rendered
    )


def _matches_action(body: str, action: dict) -> bool:
    """Match every visible field literally, independent of dictionary order."""
    if action.keys() - _ACTION_FIELDS - _EVALUATION_FIELDS:
        return False
    pending = {key: value for key, value in action.items() if key in _ACTION_FIELDS}
    if not {"action_id", "action_type", "unstructured"} <= pending.keys():
        return False
    rest = body.rstrip("\r\n")
    while pending:
        key, separator, _ = rest.partition(":\n")
        if not separator or key not in pending:
            return False
        block = key + ":\n" + _render_value(pending.pop(key))
        if not rest.startswith(block):
            return False
        rest = rest[len(block) :]
        if pending:
            if not rest.startswith("\n\n"):
                return False
            rest = rest[2:]
    return not rest


def _scope(fields):
    return tuple(fields.get(key) for key in ("scenario", "target", "scene", "action"))


def read_correlated_logs(
    text: str,
    records: Sequence[OpenWorldRecord],
    digest: str,
) -> Optional[tuple[DecisionLog, ...]]:
    matches = list(_EVENT.finditer(text))
    if not any(match[1] == "ENVIRONMENT" for match in matches):
        return None
    result = [
        DecisionLog("action_not_yet_logged", snapshot_sha256=digest) for _ in records
    ]
    groups: list[list[tuple[LogEvent, int, int]]] = []
    group: list[tuple[LogEvent, int, int]] = []
    scope = None
    group_executed = False
    execution_fields: dict[str, str] = {}
    malformed_at = None
    line = 1
    cursor = byte_cursor = 0
    byte_offsets = "\ufffd" not in text
    for index, match in enumerate(matches):
        line += text.count("\n", cursor, match.start())
        if byte_offsets:
            byte_cursor += len(text[cursor : match.start()].encode("utf-8"))
        fields = {}
        detail = False
        valid = True
        for token in match[3].split(" | "):
            if not token:
                continue
            if token == "DETAIL":
                if detail:
                    valid = False
                detail = True
            else:
                key, separator, value = token.partition("=")
                if not separator or key in fields:
                    valid = False
                    break
                fields[key] = value
        start = match.end()
        if text[start : start + 2] == "\r\n":
            start += 2
        elif text[start : start + 1] == "\n":
            start += 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        body_bytes = body.encode("utf-8")
        byte_start = byte_cursor + len(text[match.start() : start].encode("utf-8"))
        section = Section(
            f"{match[1]} {match[2]}",
            body[:_LIMIT],
            line + 1,
            line + max(1, body.count("\n") + (not body.endswith("\n"))),
            (byte_start, byte_start + len(body_bytes)) if byte_offsets else None,
            len(body) > _LIMIT,
            hashlib.sha256(body_bytes).hexdigest(),
        )
        line += text.count("\n", match.start(), end)
        cursor = end
        byte_cursor = byte_start + len(body_bytes)
        if not valid:
            malformed_at = match.start() if malformed_at is None else malformed_at
            continue
        if group_executed:
            completed_tool_result = (
                match[1] == "TOOL"
                and match[2] == "OUTPUT"
                and all(
                    fields.get(key) is not None
                    and fields.get(key) == execution_fields.get(key)
                    for key in ("call", "tool_call")
                )
            )
            duplicate_execution = match[1] == "ENVIRONMENT" and match[2] == "EXECUTED"
            if not completed_tool_result and not duplicate_execution:
                groups.append(group)
                group = []
                group_executed = False
        if "action" not in fields:
            continue
        current_scope = _scope(fields)
        if group and current_scope != scope:
            groups.append(group)
            group = []
            group_executed = False
        scope = current_scope
        group.append(
            (LogEvent(match[1], match[2], fields, detail, section), start, end)
        )
        if match[1] == "ENVIRONMENT" and match[2] == "EXECUTED":
            group_executed = True
            execution_fields = fields
    if group:
        groups.append(group)

    record_index = 0
    occurrences = {
        episode.first_record_index + offset: offset
        for episode in split_episodes(records)
        for offset in range(len(episode.records))
    }
    for group in groups:
        if record_index >= len(records):
            break
        executed = [
            item
            for item in group
            if item[0].role == "ENVIRONMENT"
            and item[0].direction == "EXECUTED"
            and not item[0].detail
        ]
        if not executed:
            # A complete failed action attempt cannot become a saved record.
            if any(
                item[0].role == "ENVIRONMENT" and item[0].direction == "REJECTED"
                for item in group
            ):
                continue
            continue
        record = records[record_index]
        episode = (record.input.scenario_id, record.input.alignment_target_id)
        event = executed[0][0]
        expected_scope = (
            episode[0],
            episode[1],
            record.input.full_state.meta_info.scene_id,
            str(occurrences[record_index]),
        )
        submissions = [
            item
            for item in group
            if item[1] < executed[0][1]
            and item[0].role == "ENVIRONMENT"
            and item[0].direction == "SUBMIT"
            and item[0].detail
            and all(
                item[0].fields.get(key) == event.fields.get(key)
                for key in ("call", "tool_call")
            )
        ]
        verified = (
            len(executed) == 1
            and _scope(event.fields) == expected_scope
            and bool(submissions)
            and (malformed_at is None or malformed_at >= group[-1][2])
        )
        if verified:
            submitted, start, end = submissions[-1]
            verified = not any(
                item[1] > start
                and item[1] < executed[0][1]
                and item[0].role == "ENVIRONMENT"
                and item[0].direction == "REJECTED"
                for item in group
            ) and _matches_action(
                text[start:end].replace("\r\n", "\n"), record.source["output"]["action"]
            )
        if not verified:
            status = f"unverified_correlated_action_at_record_{record_index}"
            result[record_index:] = [
                DecisionLog(status, snapshot_sha256=digest)
                for _ in records[record_index:]
            ]
            break
        events = tuple(item[0] for item in group)
        first, last = events[0].section, events[-1].section
        begin, end = group[0][1], group[-1][2]
        block = text[begin:end]
        excerpt = Section(
            "Correlated decision events",
            block[:_LIMIT],
            first.first_line,
            last.last_line,
            (first.byte_range[0], last.byte_range[1])
            if first.byte_range and last.byte_range
            else None,
            len(block) > _LIMIT,
            hashlib.sha256(block.encode("utf-8")).hexdigest(),
        )
        result[record_index] = DecisionLog(
            "exact_correlated_action_sequence",
            excerpt=excerpt,
            snapshot_sha256=digest,
            events=events,
        )
        record_index += 1
    return tuple(result)
