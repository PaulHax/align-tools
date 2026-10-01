"""Loading open-world run directories written by align-system."""

import pytest
from open_world_runs import completion_line, episode_records, write_run

from align_utils.open_world import find_open_world_runs, load_run

MERIT_LOW = "ADEPT-June2025-merit-0.0"
MERIT_HIGH = "ADEPT-June2025-merit-1.0"


def test_load_run_splits_episodes_and_pairs_logged_scores(tmp_path):
    run_dir = write_run(
        tmp_path / "2026-08-31__09-17-11",
        episode_records(MERIT_LOW) + episode_records(MERIT_HIGH),
        [completion_line(MERIT_LOW, 0.83), completion_line(MERIT_HIGH, 0.31)],
    )

    run = load_run(run_dir)

    assert [
        (episode.first_record_index, len(episode.records), episode.alignment_target_id)
        for episode in run.episodes
    ] == [(0, 7, MERIT_LOW), (7, 7, MERIT_HIGH)]
    assert [episode.outcome.session_alignment_score for episode in run.episodes] == [
        0.83,
        0.31,
    ]
    assert (run.config.adm.name, run.adm_profile, run.username) == (
        "phase2_pipeline_direct_regression",
        "TEST_OPENWORLD",
        "test-session",
    )
    tag = run.episodes[0].records[1]
    assert tag.output.action.parameters == {"category": "IMMEDIATE"}
    assert tag.choice_info.alignment_info.votes == {"0": 1.0, "1": 0.0}
    assert [timing.step_num for timing in tag.choice_info.per_step_timing_stats] == [
        0,
        1,
        2,
    ]
    assert [record.chosen_by_driver for record in run.episodes[0].records] == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_unaligned_sessions_split_where_the_clock_restarts(tmp_path):
    run_dir = write_run(
        tmp_path / "run",
        episode_records(None) + episode_records(None),
        [completion_line(MERIT_LOW, 0.4), completion_line(MERIT_HIGH, 0.6)],
    )

    run = load_run(run_dir)

    assert [len(episode.records) for episode in run.episodes] == [7, 7]
    assert [episode.alignment_target_id for episode in run.episodes] == [None, None]
    assert [episode.outcome.alignment_target_id for episode in run.episodes] == [
        MERIT_LOW,
        MERIT_HIGH,
    ]


def test_scene_clock_resets_do_not_split_unaligned_sessions(tmp_path):
    records = []
    for _ in range(2):
        for scene in ["treat_and_tag", "building_explosion", "evac_decision"]:
            steps = episode_records(None)[:2]
            for index, record in enumerate(steps):
                state = record["input"]["full_state"]
                state["elapsed_time"] = index * 10
                state["meta_info"]["scene_id"] = scene
            records.extend(steps)
    run_dir = write_run(
        tmp_path / "run",
        records,
        [completion_line(MERIT_LOW, 0.4), completion_line(MERIT_HIGH, 0.6)],
    )

    run = load_run(run_dir)

    assert [episode.first_record_index for episode in run.episodes] == [0, 6]
    assert [len(episode.records) for episode in run.episodes] == [6, 6]
    assert [episode.outcome.session_alignment_score for episode in run.episodes] == [
        0.4,
        0.6,
    ]


def test_sessions_without_scene_ids_use_clock_restarts(tmp_path):
    records = episode_records(None) + episode_records(None)
    for record in records:
        record["input"]["full_state"]["meta_info"] = {}
    run_dir = write_run(
        tmp_path / "run",
        records,
        [completion_line(MERIT_LOW, 0.4), completion_line(MERIT_HIGH, 0.6)],
    )

    assert [len(episode.records) for episode in load_run(run_dir).episodes] == [7, 7]


def test_episode_in_progress_has_no_outcome(tmp_path):
    records = episode_records(MERIT_LOW) + episode_records(MERIT_HIGH)[:3]
    run = load_run(
        write_run(tmp_path / "run", records, [completion_line(MERIT_LOW, 0.83)])
    )

    assert [episode.outcome is not None for episode in run.episodes] == [True, False]
    assert len(run.episodes[1].records) == 3


def test_completion_line_still_being_written_is_ignored(tmp_path):
    partial_line = completion_line(MERIT_LOW, 0.83).rstrip("\n")[:-2]
    run_dir = write_run(tmp_path / "run", episode_records(MERIT_LOW), [partial_line])

    assert load_run(run_dir).episodes[0].outcome is None


def test_mismatched_optional_scores_keep_records_without_attaching_scores(tmp_path):
    run_dir = write_run(
        tmp_path / "run",
        episode_records(MERIT_LOW) + episode_records(MERIT_HIGH),
        [completion_line(MERIT_LOW, 0.83), completion_line(MERIT_LOW, 0.31)],
    )

    run = load_run(run_dir)

    assert len(run.records) == 14
    assert all(episode.outcome is None for episode in run.episodes)
    assert "Session scores skipped" in run.score_warning


def test_extra_logged_scores_do_not_block_loading_actions(tmp_path):
    run_dir = write_run(
        tmp_path / "run",
        episode_records(MERIT_LOW),
        [completion_line(MERIT_LOW, 0.83), completion_line(MERIT_HIGH, 0.31)],
    )

    run = load_run(run_dir)

    assert len(run.records) == 7
    assert run.episodes[0].outcome is None
    assert "Session scores skipped" in run.score_warning


def test_input_output_caught_mid_write_raises_value_error(tmp_path):
    run_dir = write_run(tmp_path / "run", episode_records(MERIT_LOW))
    text = (run_dir / "input_output.json").read_text()
    (run_dir / "input_output.json").write_text(text[: len(text) // 2])

    with pytest.raises(ValueError):
        load_run(run_dir)


def test_find_open_world_runs_includes_every_open_world_driver_only(tmp_path):
    adm_driven = write_run(
        tmp_path / "adm" / "2026-08-31__09-17-11", episode_records(MERIT_LOW)
    )
    agent_driven = write_run(
        tmp_path / "agent" / "2026-09-11__10-00-00",
        episode_records(MERIT_LOW),
        driver="align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver",
    )
    write_run(
        tmp_path / "phase1" / "2025-01-01__00-00-00",
        episode_records(MERIT_LOW),
        driver="align_system.drivers.itm_phase1.ITMPhase1Driver",
    )

    assert find_open_world_runs(tmp_path) == (adm_driven, agent_driven)
