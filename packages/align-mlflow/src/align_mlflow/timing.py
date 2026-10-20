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


@dataclass(frozen=True)
class TimingPlan:
    """The captured basis and origin used for every trace in a source run."""

    basis: Literal["component_durations", "raw_times_s"]
    origin_ns: int
    raw_times_s: Tuple[float, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "basis": self.basis,
                "origin_ns": self.origin_ns,
                "raw_times_s": self.raw_times_s,
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
            if (
                basis not in ("component_durations", "raw_times_s")
                or type(origin) is not int
                or origin < 0
                or not isinstance(durations, list)
                or any(
                    type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                    for duration in durations
                )
                or (basis == "component_durations" and durations)
            ):
                raise ValueError("Invalid captured timing plan")
            return cls(basis, origin, tuple(durations))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid captured timing plan") from error

    def raw_duration_ns(self, record_index: int) -> int:
        if record_index >= len(self.raw_times_s):
            raise ValueError(
                f"Captured timing plan has no raw duration for record {record_index}; "
                "new actions cannot use a different timeline."
            )
        try:
            return round(self.raw_times_s[record_index] * 1e9)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Invalid captured raw duration for record {record_index}"
            ) from error


def resolve_timing_plan(
    run: OpenWorldRun,
    origin_ns: int,
    *,
    existing: Optional[TimingPlan] = None,
    legacy: bool = False,
) -> TimingPlan:
    """Choose raw action durations only for a new run with complete timing.

    An existing plan is authoritative even when auxiliary timing files change.
    Sources imported before plans were recorded retain component timing.
    """
    if existing is not None:
        return existing
    raw_times = tuple(run.timing.raw_times_s) if run.timing is not None else ()
    has_components = any(
        record.choice_info.per_step_timing_stats and not record.chosen_by_driver
        for record in run.records
    )
    if (
        not legacy
        and not has_components
        and raw_times
        and len(raw_times) == len(run.records)
        and all(math.isfinite(duration) and duration >= 0 for duration in raw_times)
    ):
        return TimingPlan("raw_times_s", origin_ns, raw_times)
    return TimingPlan("component_durations", origin_ns)
