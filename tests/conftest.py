from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_pairtrading.config import ProjectConfig, load_config
from rl_pairtrading.data.repository import LocalContractPrices

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    base = load_config(ROOT / "configs/smoke.toml").model_dump()
    base["environment"].update(lookback=20, short_window=3, long_window=5)
    base["agent"].update(batch_size=4, replay_capacity=32)
    base["walk_forward"].update(train_size=30, validation_size=10, test_size=10)
    base["instruments"] = [i for i in base["instruments"] if i["symbol"] in ("CS", "C")]
    return ProjectConfig.model_validate(base)


@pytest.fixture
def market():
    dates = pd.bdate_range("2020-01-01", periods=90)
    rng = np.random.default_rng(1)
    second = 100 + np.cumsum(rng.normal(0, 0.3, len(dates)))
    first = 1.2 * second + rng.normal(0, 0.2, len(dates))

    def frame(prices, contract):
        return pd.DataFrame(
            {
                "open": prices,
                "settlement": prices,
                "volume": 100.0,
                "open_interest": 200.0,
                "dominant_id": contract,
            },
            index=dates,
        )

    return {"CS": frame(first, "CS2001"), "C": frame(second, "C2001")}


@pytest.fixture
def prices():
    return LocalContractPrices(
        pd.DataFrame(
            {"open": []}, index=pd.MultiIndex.from_arrays([[], []], names=["contract", "date"])
        )
    )
