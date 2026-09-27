"""Write open-world run directories shaped like align-system's driver output."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

OPEN_WORLD_DRIVER = "align_system.drivers.itm_open_world.ITMOpenWorldDriver"
SCENARIO = "Test2026-OW_desert"
PATIENTS = ("Patient 1", "Patient 2")
COMPONENTS = (
    "align_system.algorithms.world_state_adm_component.WorldStateTrackerADMComponent",
    "align_system.algorithms.direct_regression_adm_component.OWDirectRegressionADMComponent",
    "align_system.algorithms.alignment_adm_component.MultinomialWeightedMidpointAlignmentADMComponent",
)
GENERIC_CHOICES = (
    ("check_vitals", "CHECK_VITALS", "Check a patient's vital signs"),
    ("end_scene_action", "END_SCENE", "End the scene"),
    ("move_to_patient", "MOVE_TO", "Move to a patient"),
    ("tag_patient", "TAG_CHARACTER", "Place the specified triage tag on a patient"),
    ("treat_patient", "TREAT_PATIENT", "Treat a patient with the specified supply"),
    ("evac_patient", "MOVE_TO_EVAC", "Move a patient to medevac"),
)
DRIVER_END_SCENE = "All patients have been tagged and treated"

Step = Tuple[str, Optional[str], Optional[Dict[str, Any]]]

EPISODE: Tuple[Step, ...] = (
    ("CHECK_VITALS", "Patient 1", None),
    ("TAG_CHARACTER", "Patient 1", {"category": "IMMEDIATE"}),
    ("TREAT_PATIENT", "Patient 1", {"treatment": "Tourniquet"}),
    ("MOVE_TO", "Patient 2", None),
    ("TAG_CHARACTER", "Patient 2", {"category": "DELAYED"}),
    ("END_SCENE", None, None),
    ("MOVE_TO_EVAC", "Patient 1", None),
)


def episode_records(
    target: Optional[str], scenario: str = SCENARIO, steps: Sequence[Step] = EPISODE
) -> List[Dict[str, Any]]:
    """Records for one scenario session.

    TA3's clock often stays put between actions, so elapsed_time repeats. The
    driver ends the scene itself and leaves the previous step's choice_info on
    that record, as align-system does.
    """
    records: List[Dict[str, Any]] = []
    for number, (action_type, character, parameters) in enumerate(steps):
        driver_chosen = action_type == "END_SCENE"
        records.append(
            _record(
                scenario,
                target,
                elapsed=(number // 2) * 10,
                action_type=action_type,
                character=character,
                parameters=parameters,
                justification=DRIVER_END_SCENE
                if driver_chosen
                else f"Why {action_type}",
                choice_info=records[-1]["choice_info"]
                if driver_chosen
                else _choice_info(),
            )
        )
    return records


def _choice_info() -> Dict[str, Any]:
    return {
        "predicted_kdma_values": {
            f"{patient}: injured": {"medical": [0.9], "affiliation": [1.0]}
            for patient in PATIENTS
        },
        "alignment_info": {
            "source": "MultinomialWeightedMidpointAlignmentADMComponent",
            "per_kdma": {},
            "votes": {"0": 1.0, "1": 0.0},
        },
        "per_step_timing_stats": [
            {"step": step, "step_num": index, "elapsed_s": 0.25 * (index + 1)}
            for index, step in enumerate(COMPONENTS)
        ],
    }


def _record(
    scenario: str,
    target: Optional[str],
    elapsed: int,
    action_type: str,
    character: Optional[str],
    parameters: Optional[Dict[str, Any]],
    justification: str,
    choice_info: Dict[str, Any],
) -> Dict[str, Any]:
    choices = [
        {
            "action_id": action_id,
            "action_type": kind,
            "intent_action": False,
            "unstructured": text,
        }
        for action_id, kind, text in GENERIC_CHOICES
    ]
    action: Dict[str, Any] = {
        "action_id": next(a for a, kind, _ in GENERIC_CHOICES if kind == action_type),
        "action_type": action_type,
        "intent_action": False,
        "unstructured": " ".join(filter(None, [action_type, character])),
        "justification": justification,
    }
    if character:
        action["character_id"] = character
    if parameters:
        action["parameters"] = parameters
    state = f"{scenario} at {elapsed}s"
    return {
        "input": {
            "scenario_id": scenario,
            "alignment_target_id": target,
            "full_state": {
                "unstructured": state,
                "elapsed_time": elapsed,
                "scenario_complete": False,
                "meta_info": {
                    "scene_id": "evac_decision"
                    if action_type == "MOVE_TO_EVAC"
                    else "treat_and_tag"
                },
                "characters": [
                    {
                        "id": patient,
                        "name": patient,
                        "unstructured": f"{patient} is injured",
                        "unseen": False,
                        "nearby": True,
                        "vitals": {},
                    }
                    for patient in PATIENTS
                ],
                "supplies": [{"type": "Tourniquet", "quantity": 999}],
                "events": [],
            },
            "state": state,
            "choices": choices,
        },
        "label": [{} for _ in choices],
        "choice_info": choice_info,
        "output": {
            "choice": next(
                i for i, c in enumerate(choices) if c["action_type"] == action_type
            ),
            "action": action,
        },
    }


def completion_line(target: str, score: float, scenario: str = SCENARIO) -> str:
    return (
        f"*Final state unstructured*: Scenario {scenario} complete for target "
        f"{target}. Session alignment score = {score}\n"
    )


def write_run(
    run_dir: Path,
    records: Sequence[Dict[str, Any]],
    log_lines: Sequence[str] = (),
    driver: str = OPEN_WORLD_DRIVER,
) -> Path:
    config = {
        "name": "action_based",
        "driver": {"_target_": driver},
        "adm": {
            "name": "phase2_pipeline_direct_regression",
            "structured_inference_engine": {"model_name": "test/model"},
        },
        "interface": {"adm_profile": "TEST_OPENWORLD"},
    }
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (run_dir / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "meta.json").write_text(
        json.dumps({"version": "0.5.11", "username": "test-session"})
    )
    (run_dir / "input_output.json").write_text(json.dumps(list(records), indent=2))
    (run_dir / "raw_align_system.log").write_text("Starting run\n" + "".join(log_lines))
    return run_dir
