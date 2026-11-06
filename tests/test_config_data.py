import numpy as np
import pytest
from pydantic import ValidationError

from rl_pairtrading.config import AgentConfig, EnvironmentConfig, ProjectConfig, load_config
from rl_pairtrading.data.repository import DataValidationError, align_pair, validate_market_data


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 100, "replay_capacity": 5},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"gamma": 2},
        {"learning_rate": float("nan")},
        {"typo": 3},
        {"device": "tpu"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        AgentConfig(**kwargs)


def test_windows_and_immutability():
    with pytest.raises(ValidationError):
        EnvironmentConfig(short_window=30, long_window=20)
    with pytest.raises(ValidationError):
        EnvironmentConfig().lookback = 1


def test_symbol_validation(config):
    raw = config.model_dump()
    raw["research"]["primary_pair"] = ("UNKNOWN", "C")
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_config_paths(tmp_path, monkeypatch):
    from conftest import ROOT

    monkeypatch.chdir(tmp_path)
    assert load_config(ROOT / "configs/default.toml").paths.data_dir == ROOT / "datas"


def test_alignment(market):
    original = market["CS"].copy(deep=True)
    a, b = align_pair(market["CS"].iloc[::-1], market["C"].iloc[5:])
    assert a.index.equals(b.index) and a.index.is_monotonic_increasing
    assert market["CS"].equals(original)


@pytest.mark.parametrize("invalid", ["nan", "duplicate", "negative", "missing"])
def test_bad_data(market, invalid):
    data = market["CS"].copy()
    if invalid == "nan":
        data.iloc[0, data.columns.get_loc("open")] = np.nan
    elif invalid == "duplicate":
        data.index = [data.index[0]] * len(data)
    elif invalid == "negative":
        data.iloc[0, data.columns.get_loc("volume")] = -1
    else:
        data = data.drop(columns="dominant_id")
    with pytest.raises(DataValidationError):
        validate_market_data(data)
