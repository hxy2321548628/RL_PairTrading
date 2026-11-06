import numpy as np
import pandas as pd
import pytest

from rl_pairtrading.backtesting.engine import BacktestEngine, InsufficientMarginError
from rl_pairtrading.backtesting.metrics import calculate_metrics
from rl_pairtrading.config import BacktestConfig, ProjectConfig
from rl_pairtrading.data.repository import LocalContractPrices
from rl_pairtrading.domain import Action


def signals(market, actions, dates=None):
    return pd.DataFrame(
        {
            "date": market["CS"].index[: len(actions)] if dates is None else dates,
            "first": "CS",
            "second": "C",
            "action": actions,
            "hedge_ratio": 1.0,
        }
    )


def test_round_trip(config, market, prices):
    result = BacktestEngine(config, market, prices).run(
        signals(market, [Action.LONG, Action.CLOSE])
    )
    pnl = market["CS"].open.diff().iloc[1] - market["C"].open.diff().iloc[1]
    assert result.equity.iloc[-1] == pytest.approx(
        config.backtest.initial_cash + pnl - 4 * config.backtest.commission_per_contract
    )
    assert result.margin.iloc[-1] == 0
    assert result.returns.iloc[0] == pytest.approx(
        -2 * config.backtest.commission_per_contract / config.backtest.initial_cash
    )
    np.testing.assert_allclose(result.cash + result.margin, result.equity)


def test_reversal(config, market, prices):
    result = BacktestEngine(config, market, prices).run(signals(market, [0, 1, 2]))
    trades = result.trades[result.trades.date == market["CS"].index[1]]
    assert len(trades) == 4
    assert trades[trades.symbol == "CS"].quantity.tolist() == [-1, -1]


def test_rollover_accounting(config, market, prices):
    for data in market.values():
        data.loc[:, ["open", "settlement"]] = 100.0
    date = market["CS"].index[1]
    market["CS"].loc[date:, "dominant_id"] = "CS2005"
    market["CS"].loc[date:, ["open", "settlement"]] = 200.0
    quotes = LocalContractPrices(
        pd.DataFrame({"open": [103.0]}, index=pd.MultiIndex.from_tuples([("CS2001", date)]))
    )
    result = BacktestEngine(config, market, quotes).run(signals(market, [0, 0, 2]))
    assert result.equity.iloc[-1] == pytest.approx(
        config.backtest.initial_cash + 3 - 6 * config.backtest.commission_per_contract
    )
    assert result.trades.reason.tolist().count("roll_open") == 1


def test_sparse_signals(config, market, prices):
    result = BacktestEngine(config, market, prices).run(
        signals(market, [0, 2], dates=market["CS"].index[[0, 2]])
    )
    assert len(result.equity) == 3 and result.margin.iloc[1] == 0


def test_margin_failure(config, market, prices):
    raw = config.model_dump()
    raw["backtest"]["initial_cash"] = 1
    with pytest.raises(InsufficientMarginError):
        BacktestEngine(ProjectConfig.model_validate(raw), market, prices).run(signals(market, [0]))


def test_invalid_signals(config, market, prices):
    table = signals(market, [0])
    with pytest.raises(ValueError, match="Duplicate"):
        BacktestEngine(config, market, prices).run(pd.concat([table, table]))
    table.loc[0, "action"] = 9
    with pytest.raises(ValueError):
        BacktestEngine(config, market, prices).run(table)


def test_metrics():
    dates = pd.date_range("2020-01-01", periods=3)
    metrics = calculate_metrics(pd.Series([-0.1, 0.0, 0.0], index=dates), BacktestConfig())
    assert metrics.max_drawdown == pytest.approx(-0.1)
    assert np.isnan(metrics.recovery_days)
    flat = calculate_metrics(pd.Series([0.0, 0.0, 0.0], index=dates), BacktestConfig())
    assert flat.max_drawdown == 0 and np.isnan(flat.sharpe_ratio)


def test_terminal_liquidation_and_multiplier_hedge(config, market, prices):
    raw = config.model_dump()
    for instrument in raw["instruments"]:
        instrument["multiplier"] = 10 if instrument["symbol"] == "CS" else 5
    config = ProjectConfig.model_validate(raw)
    result = BacktestEngine(config, market, prices).run(signals(market, [Action.LONG]))
    opened = result.trades[result.trades.reason == "open"]
    assert opened.quantity.tolist() == [1, -2]
    assert result.trades.reason.tolist().count("terminal_close") == 2
    assert result.margin.iloc[-1] == 0
    assert result.equity.iloc[-1] == pytest.approx(
        config.backtest.initial_cash - 6 * config.backtest.commission_per_contract
    )


def test_shared_contracts_have_independent_pair_accounts(config, market, prices):
    raw = config.model_dump()
    third = dict(raw["instruments"][0], symbol="A", name="Third", data_file="unused.pkl")
    raw["instruments"] = (*raw["instruments"], third)
    config = ProjectConfig.model_validate(raw)
    market["A"] = market["CS"].copy()
    market["A"]["dominant_id"] = "A2001"
    first_pair = signals(market, [Action.LONG, Action.CLOSE])
    second_pair = first_pair.assign(first="A")
    engine = BacktestEngine(config, market, prices)
    single = engine.run(first_pair)
    combined = engine.run(pd.concat([first_pair, second_pair]))
    assert combined.equity.iloc[-1] - config.backtest.initial_cash == pytest.approx(
        2 * (single.equity.iloc[-1] - config.backtest.initial_cash)
    )
    assert len(combined.trades) == 8
    assert combined.margin.iloc[-1] == 0
