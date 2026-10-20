"""Recorded metadata and captured completion logs for open-world runs."""

import json

import pytest
import yaml
from open_world_runs import completion_line, episode_records, write_run

from align_utils.open_world import load_run

DRIVER = "align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver"


def _source(tmp_path, driver, *, adm=None, meta=None):
    path = write_run(tmp_path / "run", episode_records(None)[:1])
    config = {
        "driver": {"_target_": DRIVER, **driver},
        "interface": {"adm_profile": "TEST_OPENWORLD"},
        **({"adm": adm} if adm is not None else {}),
    }
    (path / ".hydra/config.yaml").write_text(yaml.safe_dump(config))
    (path / "meta.json").write_text(json.dumps(meta or {}))
    return path


@pytest.mark.parametrize(
    "driver, expected",
    [
        (
            {"model": "provider:override", "chat_model": {"model": "default"}},
            "provider:override",
        ),
        ({"model": None, "chat_model": {"model": "recorded/model"}}, "recorded/model"),
        ({"chat_model": {"model": "recorded/model"}}, "recorded/model"),
        ({}, None),
    ],
)
def test_driver_model_metadata_follows_recorded_runtime_precedence(
    tmp_path, driver, expected
):
    path = _source(tmp_path, driver, meta={"driver": "langchain_agent"})

    run = load_run(path)

    assert run.metadata.llm_name == expected
    assert run.metadata.adm_name == "langchain_agent"
    assert run.metadata.driver_class == DRIVER
    assert run.config.adm.name == "unknown_adm"


def test_recorded_adm_metadata_stays_available_for_pipeline_runs(tmp_path):
    path = _source(
        tmp_path,
        {"architecture": "driver_architecture"},
        adm={
            "name": "recorded_adm",
            "structured_inference_engine": {"model_name": "recorded/backbone"},
        },
        meta={"architecture": "meta_architecture", "driver": "recorded_driver"},
    )

    run = load_run(path)

    assert run.metadata.adm_name == "recorded_adm"
    assert run.metadata.llm_name == "recorded/backbone"
    assert run.config.adm.name == "recorded_adm"


@pytest.mark.parametrize(
    "driver, meta, expected",
    [
        (
            {"architecture": "driver_architecture"},
            {"architecture": "meta_architecture"},
            "meta_architecture",
        ),
        ({"architecture": "driver_architecture"}, {}, "driver_architecture"),
        (
            {"extra_tools": {"zeta": {}, "alpha": {}}},
            {"driver": "recorded_driver"},
            "recorded_driver + alpha + zeta",
        ),
        ({}, {}, "ITMOpenWorldLangChainDriver"),
    ],
)
def test_driver_metadata_labels_use_recorded_names(tmp_path, driver, meta, expected):
    run = load_run(_source(tmp_path, driver, meta=meta))

    assert run.metadata.adm_name == expected
    assert run.config.adm.name == "unknown_adm"


def test_captured_log_controls_scores_without_reading_later_file_contents(tmp_path):
    target = "test-target"
    path = write_run(tmp_path / "run", episode_records(target))
    captured = completion_line(target, 0.25)
    (path / "raw_align_system.log").write_text(completion_line(target, 0.75))

    assert (
        load_run(path, log_text=captured).episodes[0].outcome.session_alignment_score
        == 0.25
    )
    assert load_run(path, log_text=None).episodes[0].outcome is None
    assert load_run(path).episodes[0].outcome.session_alignment_score == 0.75
