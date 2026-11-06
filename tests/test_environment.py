import numpy as np
import pandas as pd
import pytest

from rl_pairtrading.data.repository import DataValidationError, LocalContractPrices
from rl_pairtrading.domain import Action, Position
from rl_pairtrading.environment.pair_trading import PairTradingEnv


def test_reset_and_length(config, market, prices):
    env = PairTradingEnv(market["CS"], market["C"], config.environment, prices)
    initial = env.reset()
    assert initial.shape == (config.environment.state_size,)
    done, count = False, 0
    while not done:
        state, _, done = env.step(Action.LONG)
        assert np.isfinite(state).all()
        count += 1
    assert count == len(market["CS"]) - config.environment.lookback - 1
    with pytest.raises(RuntimeError):
        env.step(Action.CLOSE)
    np.testing.assert_array_equal(env.reset(), initial)
    assert env.position == Position.FLAT


def test_no_future_leak(config, market, prices):
    a, b = market["CS"].copy(), market["C"].copy()
    env = PairTradingEnv(a, b, config.environment, prices)
    state = env.reset()
    t = config.environment.lookback
    a.iloc[t + 1 :, a.columns.get_loc("open")] *= 100
    a.iloc[t:, a.columns.get_loc("volume")] *= 100
    a.iloc[t:, a.columns.get_loc("open_interest")] *= 100
    changed = PairTradingEnv(a, b, config.environment, prices)
    np.testing.assert_array_equal(state, changed.reset())
    assert env.beta == changed.beta


def test_rewards_and_close(config, market, prices):
    env = PairTradingEnv(market["CS"], market["C"], config.environment, prices)
    env.reset()
    t, beta = config.environment.lookback, env.beta
    change = market["CS"].open.diff().iloc[t + 1] - beta * market["C"].open.diff().iloc[t + 1]
    assert env.step(Action.LONG)[1] == pytest.approx(
        change - 2 * config.environment.commission_per_leg
    )
    assert env.step(Action.CLOSE)[1] == pytest.approx(-2 * config.environment.commission_per_leg)
    assert env.step(Action.CLOSE)[1] == 0


def test_rollover(config, market, prices):
    a, b = market["CS"].copy(), market["C"].copy()
    t = config.environment.lookback
    date = a.index[t + 1]
    a.loc[date:, "dominant_id"] = "CS2005"
    quotes = LocalContractPrices(
        pd.DataFrame(
            {"open": [a.open.iloc[t] + 1]}, index=pd.MultiIndex.from_tuples([("CS2001", date)])
        )
    )
    env = PairTradingEnv(a, b, config.environment, quotes)
    env.reset()
    expected = 1 - env.beta * b.open.diff().iloc[t + 1] - 4 * config.environment.commission_per_leg
    assert env.step(Action.LONG)[1] == pytest.approx(expected)
    missing = PairTradingEnv(a, b, config.environment, prices)
    missing.reset()
    with pytest.raises(DataValidationError, match="Missing rollover"):
        missing.step(Action.LONG)


def test_zero_volume(config, market, prices):
    for frame in market.values():
        frame.loc[:, ["volume", "open_interest"]] = 0
    env = PairTradingEnv(market["CS"], market["C"], config.environment, prices)
    assert np.isfinite(env.reset()).all()
