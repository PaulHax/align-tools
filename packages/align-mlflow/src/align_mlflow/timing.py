"""Select a stable synthetic timeline from recorded action durations."""

import json
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from align_utils.open_world import OpenWorldRun

TIMELINE_TAG = "align.timeline"
COMPONENT_TIMING_SOURCE = "choice_info.per_step_timing_stats"
RAW_TIMING_SOURCE = "timing.json.raw_times_s"
SYNTHETIC_TIMING_SOURCE = "synthetic_minimum"
UNKNOWN_TIMING_SOURCE = "unknown"
# Zero or unknown durations still need visible spans and strictly ordered turns.
MIN_SPAN_NS = 1_000
STEP_GAP_NS = 1_000_000
_MAX_TIMESTAMP_NS = 2**63 - 1


@dataclass(frozen=True)
class TimingPlan:
    """The captured basis and origin used for every trace in a source run."""

    basis: Literal["component_durations", "raw_times_s"]
    origin_ns: int
    raw_times_s: Tuple[Optional[float], ...] = ()
    resume_index: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "basis": self.basis,
                "origin_ns": self.origin_ns,
                "raw_times_s": self.raw_times_s,
                **({"resume_index": self.resume_index} if self.resume_index else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "TimingPlan":
        try:
            value = json.loads(text)
            basis = value["basis"]
            origin = value["origin_ns"]
            durations = value["raw_times_s"]
            resume_index = value.get("resume_index", 0)
            if (
                basis not in ("component_durations", "raw_times_s")
                or type(origin) is not int
                or origin < 0
                or not isinstance(durations, list)
                or type(resume_index) is not int
                or not 0 <= resume_index <= len(durations)
                or any(
                    duration is not None
                    and (
                        type(duration) not in (int, float)
                        or duration_ns(duration) is None
                    )
                    for duration in durations
                )
                or (basis == "component_durations" and (durations or resume_index))
            ):
                raise ValueError("Invalid captured timing plan")
            return cls(basis, origin, tuple(durations), resume_index)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid captured timing plan") from error

    def raw_duration_ns(self, record_index: int, *, start_ns: int = 0) -> Optional[int]:
        if record_index >= len(self.raw_times_s):
            return None
        return duration_ns(self.raw_times_s[record_index], start_ns=start_ns)


def duration_ns(duration: Optional[float], *, start_ns: int = 0) -> Optional[int]:
    """Return a representable measured duration, or unknown."""
    if duration is None or not math.isfinite(duration * 1e9) or duration < 0:
        return None
    nanoseconds = round(duration * 1e9)
    return nanoseconds if nanoseconds <= _MAX_TIMESTAMP_NS - start_ns else None


def _capture_raw_times(
    durations: Tuple[Optional[float], ...], origin_ns: int, resume_index: int = 0
) -> Tuple[Optional[float], ...]:
    captured = []
    start = origin_ns
    for index, duration in enumerate(durations):
        if index < resume_index:
            captured.append(duration)
            continue
        nanoseconds = duration_ns(duration)
        if nanoseconds is None or start + nanoseconds > _MAX_TIMESTAMP_NS:
            duration, nanoseconds = None, MIN_SPAN_NS
        captured.append(duration)
        start += nanoseconds + STEP_GAP_NS
    return tuple(captured)


def resolve_timing_plan(
    run: OpenWorldRun,
    origin_ns: int,
    *,
    existing: Optional[TimingPlan] = None,
    legacy: bool = False,
) -> TimingPlan:
    """Capture action durations, using null for unresolved measurements.

    Captured durations and origins remain authoritative. Appended actions may
    use raw timing only when the complete file retains every measured prefix.
    Sources imported before plans were recorded retain component timing.
    """
    raw_times = tuple(run.timing.raw_times_s) if run.timing is not None else ()
    aligned = len(raw_times) == len(run.records)
    if existing is not None:
        if existing.basis != "raw_times_s" or len(run.records) <= len(
            existing.raw_times_s
        ):
            return existing
        aligned = aligned and all(
            measured is None or measured == current
            for measured, current in zip(existing.raw_times_s, raw_times)
        )
        candidates = tuple(raw_times) if aligned else (None,) * len(run.records)
        durations = existing.raw_times_s + candidates[len(existing.raw_times_s) :]
        return TimingPlan(
            "raw_times_s",
            existing.origin_ns,
            _capture_raw_times(durations, existing.origin_ns, existing.resume_index),
            existing.resume_index,
        )
    has_components = any(
        record.choice_info.per_step_timing_stats and not record.chosen_by_driver
        for record in run.records
    )
    if not legacy and not has_components:
        return TimingPlan(
            "raw_times_s",
            origin_ns,
            _capture_raw_times(
                tuple(raw_times) if aligned else (None,) * len(run.records), origin_ns
            ),
        )
    return TimingPlan("component_durations", origin_ns)
