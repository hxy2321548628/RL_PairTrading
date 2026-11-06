"""Daily futures accounting with explicit targets, reversal and contract rollover."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from rl_pairtrading.config import ConfigModel, ProjectConfig
from rl_pairtrading.data.repository import ContractPriceProvider, validate_market_data
from rl_pairtrading.domain import Action


class InsufficientMarginError(RuntimeError):
    """The research account cannot fund this target; the run stops explicitly."""


class Signal(ConfigModel):
    date: datetime
    first: str
    second: str
    action: Action
    hedge_ratio: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def different_legs(self):
        if self.first == self.second:
            raise ValueError("A pair must contain different instruments")
        return self

    @property
    def pair(self) -> tuple[str, str]:
        return self.first, self.second


@dataclass
class Holding:
    symbol: str
    contract: str
    quantity: float
    mark_price: float
    multiplier: float
    margin_rate: float
    margin: float


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    margin: pd.Series
    returns: pd.Series
    trades: pd.DataFrame


class BacktestEngine:
    """Each pair owns its legs, including when multiple pairs share a contract.

    Missing daily signals mean CLOSE, never forward-fill a trading decision.
    Quantities are fractional research units. A production lot allocator is separate.
    """

    def __init__(
        self, config: ProjectConfig, data: dict[str, pd.DataFrame], prices: ContractPriceProvider
    ):
        self.config, self.prices = config, prices
        self.data = {symbol: validate_market_data(frame, symbol) for symbol, frame in data.items()}

    def run(self, signals: pd.DataFrame) -> BacktestResult:
        if signals.empty:
            raise ValueError("Cannot backtest an empty signal table")
        required = list(Signal.model_fields)
        missing = set(required) - set(signals.columns)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        records = [Signal.model_validate(row) for row in signals[required].to_dict("records")]
        targets: dict[pd.Timestamp, dict[tuple[str, str], Signal]] = {}
        for signal in records:
            date = pd.Timestamp(signal.date)
            daily = targets.setdefault(date, {})
            if signal.pair in daily:
                raise ValueError(f"Duplicate pair signal: {date} {signal.pair}")
            for symbol in signal.pair:
                self.config.instrument(symbol)
                if symbol not in self.data or date not in self.data[symbol].index:
                    raise ValueError(f"No market data for {symbol} at {date}")
            daily[signal.pair] = signal
        dates = sorted(targets)
        symbols = sorted({symbol for signal in records for symbol in signal.pair})
        calendar = self.data[symbols[0]].loc[dates[0] : dates[-1]].index
        # Explicit alignment policy: shared calendar, no silent forward-fill of settlements.
        for symbol in symbols[1:]:
            other = self.data[symbol].loc[dates[0] : dates[-1]].index
            if not calendar.equals(other):
                raise ValueError(
                    "Backtest instruments must share a trading calendar in the run interval"
                )
        cash = self.config.backtest.initial_cash
        holdings: dict[tuple[str, str], list[Holding]] = {}
        trades, history = [], []
        fee_rate = self.config.backtest.commission_per_contract

        def record(date, pair, leg, quantity, price, fee, reason):
            trades.append(
                {
                    "date": date,
                    "pair": "-".join(pair),
                    "symbol": leg.symbol,
                    "contract": leg.contract,
                    "quantity": quantity,
                    "price": price,
                    "commission": fee,
                    "reason": reason,
                }
            )

        def close(date, pair, leg, *, settlement=False, reason="close"):
            nonlocal cash
            row = self.data[leg.symbol].loc[date]
            price = (
                float(row.settlement)
                if settlement
                else (
                    float(row.open)
                    if row.dominant_id == leg.contract
                    else self.prices.open_price(leg.contract, date)
                )
            )
            fee = abs(leg.quantity) * fee_rate
            cash += leg.margin + (price - leg.mark_price) * leg.quantity * leg.multiplier - fee
            record(date, pair, leg, -leg.quantity, price, fee, reason)

        def make_leg(date, symbol, quantity):
            instrument = self.config.instrument(symbol)
            row = self.data[symbol].loc[date]
            margin = abs(quantity) * instrument.multiplier * row.open * instrument.margin_rate
            return Holding(
                symbol,
                row.dominant_id,
                quantity,
                float(row.open),
                instrument.multiplier,
                instrument.margin_rate,
                float(margin),
            )

        def open_legs(date, pair, legs, reason="open"):
            nonlocal cash
            required_cash = sum(leg.margin + abs(leg.quantity) * fee_rate for leg in legs)
            if required_cash > cash:
                raise InsufficientMarginError(
                    f"{date}: pair {pair} requires {required_cash:.2f}, available cash {cash:.2f}"
                )
            cash -= required_cash
            for leg in legs:
                record(
                    date,
                    pair,
                    leg,
                    leg.quantity,
                    leg.mark_price,
                    abs(leg.quantity) * fee_rate,
                    reason,
                )

        for date in calendar:
            daily = targets.get(date, {})
            desired = {}
            for pair, signal in daily.items():
                sign = int(signal.action.position)
                if sign:
                    # Match price-spread beta after accounting for instrument multipliers.
                    quantity = sign * self.config.backtest.base_quantity
                    ratio = (
                        self.config.instrument(pair[0]).multiplier
                        / self.config.instrument(pair[1]).multiplier
                    )
                    desired[pair] = (quantity, -quantity * signal.hedge_ratio * ratio)
            # Release all obsolete targets before allocating new positions.
            for pair in list(holdings):
                legs = holdings[pair]
                if pair not in desired or not np.allclose(
                    [leg.quantity for leg in legs], desired[pair], rtol=0, atol=0
                ):
                    for leg in legs:
                        close(date, pair, leg, reason="rebalance" if pair in desired else "close")
                    del holdings[pair]
            for pair, quantities in sorted(desired.items()):
                if pair not in holdings:
                    legs = [
                        make_leg(date, symbol, quantity)
                        for symbol, quantity in zip(pair, quantities, strict=True)
                    ]
                    open_legs(date, pair, legs)
                    holdings[pair] = legs
                else:
                    for index, leg in enumerate(holdings[pair]):
                        if self.data[leg.symbol].loc[date].dominant_id != leg.contract:
                            close(date, pair, leg, reason="roll_close")
                            replacement = make_leg(date, leg.symbol, leg.quantity)
                            open_legs(date, pair, [replacement], reason="roll_open")
                            holdings[pair][index] = replacement
            # Mark all legs before checking cash, avoiding intermediate leg-order failures.
            for legs in holdings.values():
                for leg in legs:
                    settlement = float(self.data[leg.symbol].loc[date].settlement)
                    cash += (settlement - leg.mark_price) * leg.quantity * leg.multiplier
                    margin = abs(leg.quantity) * leg.multiplier * settlement * leg.margin_rate
                    cash += leg.margin - margin
                    leg.mark_price, leg.margin = settlement, margin
            if date == calendar[-1] and self.config.backtest.liquidate_at_end:
                for pair, legs in holdings.items():
                    for leg in legs:
                        close(date, pair, leg, settlement=True, reason="terminal_close")
                holdings.clear()
            margin = sum(leg.margin for legs in holdings.values() for leg in legs)
            if cash < 0:
                raise InsufficientMarginError(f"{date}: negative cash after settlement: {cash:.2f}")
            history.append({"date": date, "cash": cash, "margin": margin, "equity": cash + margin})
        frame = pd.DataFrame(history).set_index("date")
        returns = frame.equity.pct_change()
        returns.iloc[0] = frame.equity.iloc[0] / self.config.backtest.initial_cash - 1
        return BacktestResult(
            frame.equity,
            frame.cash,
            frame.margin,
            returns.rename("return"),
            pd.DataFrame(
                trades,
                columns=[
                    "date",
                    "pair",
                    "symbol",
                    "contract",
                    "quantity",
                    "price",
                    "commission",
                    "reason",
                ],
            ),
        )
