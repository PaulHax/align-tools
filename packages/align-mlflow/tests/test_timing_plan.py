"""Trace durations from a captured timing basis preserve stable sessions."""

import json

import pytest
import yaml
from open_world_fixtures import completion_line, episode_records, write_run

from align_mlflow.timing import TimingPlan, resolve_timing_plan
from align_mlflow.traces import (
    run_key,
    run_label,
    session_scores,
    step_traces,
)
from align_utils.open_world import load_run


def _generic_source(tmp_path, records=None, raw_times=None):
    source = records if records is not None else episode_records(None)[:2]
    for record in source:
        record["choice_info"].pop("per_step_timing_stats", None)
    path = write_run(tmp_path / "run", source)
    if raw_times is not None:
        (path / "timing.json").write_text(
            json.dumps({"scenarios": [], "raw_times_s": raw_times})
        )
    return path


def test_recorded_action_durations_include_driver_actions_and_keep_original_json(
    tmp_path,
):
    records = episode_records(None)[4:6]
    run = load_run(_generic_source(tmp_path, records, [1.25, 0.375]))
    plan = resolve_timing_plan(run, 100_000)

    traces = step_traces(run, 9_000_000, timing_plan=plan)

    assert [trace.root.end_ns - trace.root.start_ns for trace in traces] == [
        1_250_000_000,
        375_000_000,
    ]
    assert [trace.root.start_ns for trace in traces] == [100_000, 1_251_100_000]
    assert all(not trace.children for trace in traces)
    assert all(
        trace.root.attributes["align.timing_source"] == "timing.json.raw_times_s"
        and trace.tags["align.timeline_type"] == "synthetic"
        for trace in traces
    )
    assert traces[1].root.attributes["align.choice_info_applies_to_action"] is False
    assert [
        {"input": trace.root.inputs["source"], **trace.root.outputs["source"]}
        for trace in traces
    ] == records


def test_component_timeline_keeps_pipeline_and_driver_durations(tmp_path):
    path = write_run(tmp_path / "run", episode_records(None))
    (path / "timing.json").write_text(
        json.dumps({"scenarios": [], "raw_times_s": [99.0] * 7})
    )
    run = load_run(path)

    plan = resolve_timing_plan(run, 100_000)
    traces = step_traces(run, 200_000, timing_plan=plan)
    legacy = step_traces(run, 100_000)

    assert traces == legacy
    assert [trace.root.end_ns - trace.root.start_ns for trace in traces] == [
        1_500_000_000,
    ] * 5 + [1_000, 1_500_000_000]
    assert [len(trace.children) for trace in traces] == [3] * 5 + [0, 3]


_CORRUPTED_PLAN_FIELDS: list[dict[str, object]] = [
    {"basis": "other"},
    {"origin_ns": "100"},
    {"origin_ns": True},
    {"origin_ns": -1},
    {"raw_times_s": ["0.5"]},
    {"raw_times_s": [True]},
    {"raw_times_s": [-0.5]},
    {"raw_times_s": [float("nan")]},
    {"raw_times_s": [float("inf")]},
    {"basis": "component_durations"},
]


@pytest.mark.parametrize(
    "raw_times, expected",
    [
        (None, [1_000, 1_000]),
        ([0.5], [1_000, 1_000]),
        ([0.5, 0.5, 0.5], [1_000, 1_000]),
        ([0.5, float("nan")], [500_000_000, 1_000]),
        ([float("inf"), 0.5], [1_000, 500_000_000]),
        ([-0.5, 0.5], [1_000, 500_000_000]),
        ([1e300, 0.5], [1_000, 500_000_000]),
    ],
)
def test_unresolved_action_timing_uses_explicit_unknown_spans(
    tmp_path, raw_times, expected
):
    run = load_run(_generic_source(tmp_path, raw_times=raw_times))

    traces = step_traces(run, 500, timing_plan=resolve_timing_plan(run, 500))

    assert [trace.root.start_ns for trace in traces] == [
        500,
        500 + expected[0] + 1_000_000,
    ]
    assert [trace.root.end_ns - trace.root.start_ns for trace in traces] == expected
    assert [trace.tags["align.timing_source"] for trace in traces] == [
        "unknown" if duration == 1_000 else "timing.json.raw_times_s"
        for duration in expected
    ]


def test_captured_timeline_ignores_later_timing_and_copy_origin(tmp_path):
    path = _generic_source(tmp_path, raw_times=[1.25, 0.375])
    run = load_run(path)
    plan = TimingPlan.from_json(resolve_timing_plan(run, 100_000).to_json())
    original = step_traces(run, 100_000, timing_plan=plan)
    (path / "timing.json").write_text("partial")

    copied = load_run(path)
    captured = resolve_timing_plan(copied, 9_000_000, existing=plan)

    assert step_traces(copied, 9_000_000, timing_plan=captured) == original


def test_legacy_import_does_not_select_later_action_timing(tmp_path):
    path = _generic_source(tmp_path, raw_times=[1.25, 0.375])
    run = load_run(path)
    plan = resolve_timing_plan(run, 100_000, legacy=True)

    assert step_traces(run, 100_000, timing_plan=plan) == step_traces(run, 100_000)


def test_uncaptured_action_has_unknown_duration(tmp_path):
    path = _generic_source(tmp_path, raw_times=[1.25, 0.375])
    plan = resolve_timing_plan(load_run(path), 100_000)
    records = episode_records(None)[:3]
    for record in records:
        record["choice_info"].pop("per_step_timing_stats", None)
    (path / "input_output.json").write_text(json.dumps(records))
    (path / "timing.json").write_text(
        json.dumps({"scenarios": [], "raw_times_s": [1.25, 0.375, 0.5]})
    )

    traces = step_traces(load_run(path), 100_000, timing_plan=plan)

    assert traces[2].root.end_ns - traces[2].root.start_ns == 1_000
    assert traces[2].tags["align.timing_source"] == "unknown"


@pytest.mark.parametrize(
    "stored",
    [
        "partial",
        "null",
        "[]",
        "{}",
        *[
            json.dumps(
                {
                    "basis": "raw_times_s",
                    "origin_ns": 100,
                    "raw_times_s": [0.5],
                    **changed,
                }
            )
            for changed in _CORRUPTED_PLAN_FIELDS
        ],
    ],
)
def test_corrupted_captured_timing_plan_reports_value_error(stored):
    with pytest.raises(ValueError, match="Invalid captured timing plan"):
        TimingPlan.from_json(stored)


def test_resolved_metadata_identifies_runs_and_score_sessions(tmp_path):
    path = _generic_source(tmp_path, episode_records(None))
    config_path = path / ".hydra/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    del config["adm"]
    config["driver"].update(
        {"architecture": "recorded_architecture", "model": "provider:model"}
    )
    config_path.write_text(yaml.safe_dump(config))
    (path / "raw_align_system.log").write_text(completion_line("scored-target", 0.5))
    run = load_run(path)
    traces = step_traces(run, 100_000)

    assert run_label(run) == "recorded_architecture TEST_OPENWORLD run"
    assert run_key(run) == "935f0bbf15a1030a"
    assert traces[0].tags["adm"] == "recorded_architecture"
    assert traces[0].tags["llm"] == "provider:model"
    assert "recorded_architecture TEST_OPENWORLD run" in traces[0].session_id
    assert session_scores(run)[0].session_id == traces[0].session_id
