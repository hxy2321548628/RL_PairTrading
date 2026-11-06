"""Statistical primitives, independent of notebook globals."""

import numpy as np
from statsmodels.tsa.stattools import coint


def hedge_ratio(first: np.ndarray, second: np.ndarray) -> float:
    """No-intercept OLS, preserving the original spread definition y - beta*x."""
    denominator = float(second @ second)
    if denominator <= 0:
        raise ValueError("Cannot estimate a hedge ratio from zero prices")
    beta = float(second @ first / denominator)
    if not np.isfinite(beta):
        raise ValueError("Non-finite hedge ratio")
    return beta


def cointegration_pvalue(first: np.ndarray, second: np.ndarray) -> float:
    """Engle–Granger, no deterministic term, matching the no-intercept hedge."""
    if len(first) != len(second) or len(first) < 20:
        raise ValueError("Cointegration requires at least 20 aligned observations")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Cointegration inputs must be finite")
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return 1.0
    return float(coint(first, second, trend="n", autolag="aic")[1])
