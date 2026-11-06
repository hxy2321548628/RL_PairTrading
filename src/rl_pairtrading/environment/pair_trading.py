"""Open-to-open spread environment with causal observations and explicit rollover quotes."""

import numpy as np
import pandas as pd

from rl_pairtrading.config import EnvironmentConfig
from rl_pairtrading.data.repository import ContractPriceProvider, align_pair
from rl_pairtrading.data.statistics import hedge_ratio
from rl_pairtrading.domain import Action, Position


class PairTradingEnv:
    """At date t: observe open(t), execute action, earn PnL through open(t+1).

    Volume and open interest are lagged by one full session. Hedge estimation uses
    [t-lookback, t), and refreshes only while flat so an open spread retains its hedge.
    This is a research reward in price units, not a futures cash account.
    """

    def __init__(
        self,
        first: pd.DataFrame,
        second: pd.DataFrame,
        config: EnvironmentConfig,
        prices: ContractPriceProvider,
    ):
        self.first, self.second = align_pair(first, second)
        self.config, self.prices = config, prices
        if len(self.first) < config.lookback + 2:
            raise ValueError("Need lookback + at least two common trading dates")
        self.dates = self.first.index
        self._open = np.column_stack([self.first.open, self.second.open]).astype(float)
        self._volume = np.column_stack([self.first.volume, self.second.volume]).astype(float)
        self._oi = np.column_stack([self.first.open_interest, self.second.open_interest]).astype(
            float
        )
        self._contracts = np.column_stack([self.first.dominant_id, self.second.dominant_id])
        self.state_size = config.state_size
        self.action_size = len(Action)
        self._ready = False

    @property
    def date(self) -> pd.Timestamp:
        return self.dates[self._cursor]

    def reset(self) -> np.ndarray:
        self._cursor = self.config.lookback
        self.position = Position.FLAT
        self.beta = None
        self._beta_date = None
        self._done = False
        self._ready = True
        return self._state()

    def _refresh_beta(self) -> None:
        stale = (
            self._beta_date is None
            or (self.date - self._beta_date).days >= self.config.hedge_refresh_days
        )
        if self.beta is None or (self.position == Position.FLAT and stale):
            history = self._open[self._cursor - self.config.lookback : self._cursor]
            self.beta = hedge_ratio(history[:, 0], history[:, 1])
            self._beta_date = self.date

    def _state(self) -> np.ndarray:
        self._refresh_beta()
        t, cfg = self._cursor, self.config
        spread = self._open[t - cfg.lookback : t + 1, 0] - (
            self.beta * self._open[t - cfg.lookback : t + 1, 1]
        )
        long = spread[-cfg.long_window :]
        std = max(float(long.std(ddof=1)), cfg.numerical_epsilon)
        lagged_volume = np.maximum(self._volume[t - 1], cfg.numerical_epsilon)
        oi_change = (self._oi[t - 1] - self._oi[t - 2]) / np.maximum(
            self._oi[t - 2], cfg.numerical_epsilon
        )
        month_angle = 2 * np.pi * (self.date.month - 1) / 12
        day_angle = 2 * np.pi * (self.date.day - 1) / 31
        numeric = [
            np.sin(month_angle),
            np.cos(month_angle),
            np.cos(day_angle),
            np.sin(day_angle),
            np.log(lagged_volume[0] / lagged_volume[1]),
            oi_change[1],
            oi_change[0],
            (spread[-1] - long.mean()) / (cfg.bollinger_width * std),
            spread[-cfg.short_window :].mean() - long.mean(),
            spread[:-1].mean(),
            spread[:-1].std(ddof=1),
        ]
        position = [self.position == p for p in (Position.FLAT, Position.LONG, Position.SHORT)]
        state = np.asarray([*numeric, *position, *spread[-cfg.spread_lags :]], dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError(f"Non-finite observation at {self.date}")
        return state

    def step(self, action: Action | int) -> tuple[np.ndarray, float, bool]:
        if not self._ready or self._done:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        action = Action(action)
        target = action.position
        t = self._cursor
        next_open = self._open[t + 1].copy()
        rolls = self._contracts[t + 1] != self._contracts[t]
        # Flat actions need no old-contract prices. No fabricated crossover PnL.
        if target != Position.FLAT:
            for leg in np.flatnonzero(rolls):
                next_open[leg] = self.prices.open_price(self._contracts[t, leg], self.dates[t + 1])
        change = next_open - self._open[t]
        transaction_cost = 2 * abs(target - self.position) * self.config.commission_per_leg
        if target != Position.FLAT:
            transaction_cost += 2 * int(rolls.sum()) * self.config.commission_per_leg
        reward = target * (change[0] - self.beta * change[1]) - transaction_cost
        self.position = target
        self._cursor += 1
        self._done = self._cursor == len(self.dates) - 1
        return self._state(), float(reward), self._done
