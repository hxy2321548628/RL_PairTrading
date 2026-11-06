"""Return metrics including initial capital in drawdown calculations."""

import numpy as np
import pandas as pd

from rl_pairtrading.config import BacktestConfig


def calculate_metrics(returns: pd.Series, config: BacktestConfig) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Returns must be nonempty, finite and greater than -100%")
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.has_duplicates
        or (not returns.index.is_monotonic_increasing)
    ):
        raise ValueError("Metrics require unique increasing dates")
    wealth = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdowns = wealth / peaks - 1
    annual_return = wealth[-1] ** (config.annualization / len(values)) - 1
    volatility = (
        float(values.std(ddof=1) * np.sqrt(config.annualization)) if len(values) > 1 else 0.0
    )
    daily_risk_free = (1 + config.risk_free_rate) ** (1 / config.annualization) - 1
    sharpe = (
        (values.mean() - daily_risk_free) * config.annualization / volatility
        if volatility > 0
        else np.nan
    )
    maximum_drawdown = float(drawdowns.min())
    trough = int(drawdowns.argmin())
    if maximum_drawdown < 0:
        peak_candidates = np.flatnonzero(wealth[: trough + 1] == peaks[trough])
        # Initial capital has no prior timestamp; use first observation for duration only.
        peak_date = returns.index[peak_candidates[-1]] if len(peak_candidates) else returns.index[0]
        duration = (returns.index[trough] - peak_date).days
        recovery = np.flatnonzero(wealth[trough:] >= peaks[trough])
        recovery_days = (
            (returns.index[trough + recovery[0]] - returns.index[trough]).days
            if len(recovery)
            else np.nan
        )
    else:
        duration, recovery_days = 0, 0
    return pd.Series(
        {
            "total_return": wealth[-1] - 1,
            "annualized_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": maximum_drawdown,
            "calmar_ratio": annual_return / abs(maximum_drawdown)
            if maximum_drawdown < 0
            else np.nan,
            "drawdown_duration_days": duration,
            "recovery_days": recovery_days,
        }
    )
