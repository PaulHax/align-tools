"""Models and loaders for align-system open-world run directories.

An open-world run is one align-system invocation driven by an
``itm_open_world*`` driver. The driver appends one record per action to
``input_output.json`` and rewrites the whole file after every action, so a run
directory can be loaded while the run is still in progress.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .models import Action, ChoiceInfo, ExperimentConfig, TimingData

INPUT_OUTPUT_FILE = "input_output.json"
RAW_LOG_FILE = "raw_align_system.log"
TIMING_FILE = "timing.json"
META_FILE = "meta.json"
CONFIG_FILE = Path(".hydra") / "config.yaml"
FINAL_STATE_GLOB = "*.final_state_unstructured.json"
OPEN_WORLD_DRIVER_PREFIX = "align_system.drivers.itm_open_world"

T = TypeVar("T")

# Justifications the driver writes when it picks an action itself instead of
# asking the ADM. When it ends a scene this way it leaves the previous step's
# choice_info on the record, so that choice_info does not describe the action.
DRIVER_JUSTIFICATION_PREFIXES = (
    "All patients have been tagged and treated",
    "Random action chosen due to component failure",
)

COMPLETION_MARKER = "*Final state unstructured*: "
_COMPLETION_TEXT = re.compile(
    r"Scenario (?P<scenario_id>\S+) complete for target (?P<target>\S+)\.(?:\s|$)"
    r"(?:Session alignment score = (?P<score>\S+))?"
)


class Character(BaseModel):
    """A character as the ADM saw it; unseen or distant ones carry less detail."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    name: Optional[str] = None
    unstructured: Optional[str] = None
    unseen: Optional[bool] = None
    nearby: Optional[bool] = None
    tag: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None


class MetaInfo(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    scene_id: Optional[str] = None


class WorldState(BaseModel):
    """The TA3 state returned before the action; not an omniscient view."""

    model_config = ConfigDict(extra="allow", frozen=True)

    unstructured: Optional[str] = None
    elapsed_time: float = 0.0
    scenario_complete: Optional[bool] = None
    meta_info: MetaInfo = Field(default_factory=MetaInfo)
    characters: Tuple[Character, ...] = ()
    supplies: Tuple[Dict[str, Any], ...] = ()
    events: Tuple[Dict[str, Any], ...] = ()


class OpenWorldInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str
    alignment_target_id: Optional[str] = None
    full_state: WorldState
    state: Optional[str] = None
    choices: Tuple[Action, ...] = ()


class OpenWorldOutput(BaseModel):
    """The action taken.

    ``input.choices`` are generic (one per action type) while ``action`` names
    the character and parameters. ``choice`` is the index of the first choice
    sharing the action's ``action_id``.
    """

    model_config = ConfigDict(frozen=True)

    choice: Optional[int] = None
    action: Action


class OpenWorldRecord(BaseModel):
    """One ``input_output.json`` entry: the state the ADM saw and the action taken."""

    model_config = ConfigDict(frozen=True)

    input: OpenWorldInput
    output: OpenWorldOutput
    choice_info: ChoiceInfo = Field(default_factory=ChoiceInfo)
    label: Tuple[Dict[str, Any], ...] = ()

    @property
    def chosen_by_driver(self) -> bool:
        justification = self.output.action.justification or ""
        return justification.startswith(DRIVER_JUSTIFICATION_PREFIXES)


class EpisodeOutcome(BaseModel):
    """The line the driver logs when TA3 reports a scenario session complete.

    ``alignment_target_id`` is the target TA3 scored against, which is known
    even when the ADM ran unaligned and recorded no target.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    scenario_id: Optional[str] = None
    alignment_target_id: Optional[str] = None
    session_alignment_score: Optional[float] = None


class Episode(BaseModel):
    """Consecutive records from one scenario session.

    ``outcome`` is None until TA3 reports the session complete, and for runs
    without ``raw_align_system.log``.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    first_record_index: int
    records: Tuple[OpenWorldRecord, ...]
    outcome: Optional[EpisodeOutcome] = None

    @property
    def scenario_id(self) -> str:
        return self.records[0].input.scenario_id

    @property
    def alignment_target_id(self) -> Optional[str]:
        """Target the ADM aligned to; None for unaligned ADMs such as the baseline."""
        return self.records[0].input.alignment_target_id


class OpenWorldRun(BaseModel):
    """An open-world run directory, possibly still being written.

    ``adm_profile`` is the TA3 profile that chose the scenario set, and
    ``username`` is the TA3 session name, unique per live run.
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    episodes: Tuple[Episode, ...]
    config: Optional[ExperimentConfig] = None
    adm_profile: Optional[str] = None
    version: Optional[str] = None
    username: Optional[str] = None
    timing: Optional[TimingData] = None

    @property
    def records(self) -> Tuple[OpenWorldRecord, ...]:
        return tuple(record for episode in self.episodes for record in episode.records)


def parse_records(raw_records: Iterable[Dict[str, Any]]) -> Tuple[OpenWorldRecord, ...]:
    return tuple(OpenWorldRecord.model_validate(raw) for raw in raw_records)


def starts_episode(previous: OpenWorldRecord, current: OpenWorldRecord) -> bool:
    """Whether ``current`` begins a new scenario session.

    input_output.json has no session marker. Unaligned ADMs record a null
    target for every session of a scenario, so TA3's clock restarting is what
    separates repeated sessions.
    """
    previous_key = (previous.input.scenario_id, previous.input.alignment_target_id)
    current_key = (current.input.scenario_id, current.input.alignment_target_id)
    return (
        current_key != previous_key
        or current.input.full_state.elapsed_time
        < previous.input.full_state.elapsed_time
    )


def split_episodes(records: Sequence[OpenWorldRecord]) -> Tuple[Episode, ...]:
    if not records:
        return ()
    starts = [0] + [
        index
        for index in range(1, len(records))
        if starts_episode(records[index - 1], records[index])
    ]
    ends = starts[1:] + [len(records)]
    return tuple(
        Episode(
            index=number, first_record_index=start, records=tuple(records[start:end])
        )
        for number, (start, end) in enumerate(zip(starts, ends))
    )


def parse_episode_outcomes(log_text: str) -> Tuple[EpisodeOutcome, ...]:
    """Completion lines from raw_align_system.log, in the order episodes finished.

    Only newline-terminated lines count, so a line still being written is not
    read with a truncated score.
    """
    complete_text = log_text[: log_text.rfind("\n") + 1]
    return tuple(
        _parse_outcome(line.split(COMPLETION_MARKER, 1)[1].strip())
        for line in complete_text.splitlines()
        if COMPLETION_MARKER in line
    )


def _parse_outcome(text: str) -> EpisodeOutcome:
    match = _COMPLETION_TEXT.match(text)
    if match is None:
        return EpisodeOutcome(text=text)
    return EpisodeOutcome(
        text=text,
        scenario_id=match["scenario_id"],
        alignment_target_id=match["target"],
        session_alignment_score=_to_float(match["score"]),
    )


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pair_outcomes(
    episodes: Sequence[Episode], outcomes: Sequence[EpisodeOutcome]
) -> Tuple[Episode, ...]:
    """Attach the k-th logged completion to the k-th episode.

    Raises ValueError when there are more completions than episodes, or when a
    completion names a different scenario or target than its episode: either
    means the episode split does not match the sessions the driver ran.
    """
    if len(outcomes) > len(episodes):
        raise ValueError(
            f"{len(outcomes)} completed episodes logged but only "
            f"{len(episodes)} found in {INPUT_OUTPUT_FILE}"
        )
    for episode, outcome in zip(episodes, outcomes):
        if not _outcome_matches(episode, outcome):
            raise ValueError(
                f"episode {episode.index} ({episode.scenario_id}, "
                f"{episode.alignment_target_id}) does not match its logged "
                f"completion: {outcome.text}"
            )
    paired = tuple(
        episode.model_copy(update={"outcome": outcome})
        for episode, outcome in zip(episodes, outcomes)
    )
    return paired + tuple(episodes[len(outcomes) :])


def _outcome_matches(episode: Episode, outcome: EpisodeOutcome) -> bool:
    if outcome.scenario_id is None:
        return True
    return outcome.scenario_id == episode.scenario_id and (
        episode.alignment_target_id is None
        or outcome.alignment_target_id == episode.alignment_target_id
    )


def is_open_world_run(run_dir: Path) -> bool:
    """Whether run_dir has records and a Hydra config using an open-world driver."""
    config_path = run_dir / CONFIG_FILE
    if not (run_dir / INPUT_OUTPUT_FILE).is_file() or not config_path.is_file():
        return False
    try:
        config = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError:
        return False
    driver = config.get("driver") if isinstance(config, dict) else None
    target = driver.get("_target_") if isinstance(driver, dict) else None
    return isinstance(target, str) and target.startswith(OPEN_WORLD_DRIVER_PREFIX)


def find_open_world_runs(root: Path) -> Tuple[Path, ...]:
    """Open-world run directories at or below root, sorted by path."""
    return tuple(
        sorted(
            path.parent
            for path in root.rglob(INPUT_OUTPUT_FILE)
            if is_open_world_run(path.parent)
        )
    )


def load_run(run_dir: Path) -> OpenWorldRun:
    """Load an open-world run directory.

    Raises OSError when input_output.json is missing and ValueError when it is
    not complete JSON (it may be mid-write) or when logged completions cannot
    be paired with episodes. Config, meta and timing files are optional.
    """
    log_path = run_dir / RAW_LOG_FILE
    # Completions are logged after their episode's records are saved, so
    # reading the log first never finds a completion without its episode.
    outcomes = (
        parse_episode_outcomes(log_path.read_text(encoding="utf-8", errors="replace"))
        if log_path.is_file()
        else ()
    )
    records = parse_records(json.loads((run_dir / INPUT_OUTPUT_FILE).read_text()))
    meta = _as_dict(_read_optional(run_dir / META_FILE, json.loads))
    raw_config = _as_dict(_read_optional(run_dir / CONFIG_FILE, yaml.safe_load))
    return OpenWorldRun(
        path=run_dir,
        episodes=pair_outcomes(split_episodes(records), outcomes),
        config=_experiment_config(raw_config),
        adm_profile=_optional_str(
            _as_dict(raw_config.get("interface")).get("adm_profile")
        ),
        version=_optional_str(meta.get("version")),
        username=_optional_str(meta.get("username")),
        timing=_read_optional(
            run_dir / TIMING_FILE,
            lambda text: TimingData.model_validate(json.loads(text)),
        ),
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _experiment_config(raw_config: Dict[str, Any]) -> Optional[ExperimentConfig]:
    if not raw_config:
        return None
    try:
        return ExperimentConfig.model_validate(raw_config)
    except ValueError:
        return None


def _read_optional(path: Path, parse: Callable[[str], T]) -> Optional[T]:
    """Parse an auxiliary file, or None when it is missing or unreadable.

    timing.json is written at the end of a run and may be caught mid-write.
    """
    if not path.is_file():
        return None
    try:
        return parse(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return None
