from __future__ import annotations

import copy

import pytest
import yaml

from src.config import ConfigError, parse_config


@pytest.fixture
def raw(repo):
    return yaml.safe_load((repo / "config.yaml").read_text(encoding="utf-8"))


def test_repo_config_loads(config):
    assert config.k == 3
    assert config.models["persona"] == config.models["naive"]


def test_judge_temperature_is_null_for_sonnet_5(config):
    """DEVIATION from spec 5: claude-sonnet-5 rejects any non-default sampling
    parameter with a 400. None means 'send no temperature field'."""
    assert config.models["judge"] == "claude-sonnet-5"
    assert config.temperature_for("judge") is None


def test_judge_thinking_disabled(config):
    assert config.thinking_for("judge") == "disabled"


def test_persona_temperature_survives(config):
    assert config.temperature_for("persona") == 0.7


def test_subsets_are_nested(config):
    b3, b8, b15 = (set(config.subsets[a]) for a in ("B3", "B8", "B15"))
    assert b3 < b8 < b15


def test_mismatched_persona_and_naive_models_raise(raw):
    data = copy.deepcopy(raw)
    data["models"]["naive"] = "claude-sonnet-5"
    with pytest.raises(ConfigError, match="confounds architecture with model capability"):
        parse_config(data)


def test_judge_sharing_generator_model_raises(raw):
    data = copy.deepcopy(raw)
    data["models"]["judge"] = data["models"]["persona"]
    with pytest.raises(ConfigError, match="must differ from the generator"):
        parse_config(data)


def test_k_other_than_three_raises(raw):
    data = copy.deepcopy(raw)
    data["k"] = 5
    with pytest.raises(ConfigError, match="k must be 3"):
        parse_config(data)


def test_subset_size_must_match_arm_name(raw):
    data = copy.deepcopy(raw)
    data["subsets"]["B3"] = ["001", "002"]
    with pytest.raises(ConfigError, match="names 2 personas but the arm is called B3"):
        parse_config(data)


def test_duplicate_persona_in_subset_raises(raw):
    data = copy.deepcopy(raw)
    data["subsets"]["B3"] = ["001", "001", "014"]
    with pytest.raises(ConfigError, match="duplicate persona ids"):
        parse_config(data)


def test_out_of_range_threshold_raises(raw):
    data = copy.deepcopy(raw)
    data["backlash_threshold"] = 1.5
    with pytest.raises(ConfigError, match="backlash_threshold"):
        parse_config(data)
