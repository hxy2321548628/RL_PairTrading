"""Reusable walk-forward execution. Experiment selection remains in the notebook."""

import logging
from collections.abc import Mapping
from itertools import combinations

import pandas as pd

from rl_pairtrading.agents.dqn import DQNAgent
from rl_pairtrading.config import ProjectConfig
from rl_pairtrading.data.repository import ContractPriceProvider, align_pair
from rl_pairtrading.data.statistics import cointegration_pvalue
from rl_pairtrading.domain import Action
from rl_pairtrading.environment.pair_trading import PairTradingEnv
from rl_pairtrading.training.trainer import Trainer, evaluate
from rl_pairtrading.training.windows import walk_forward_windows

logger = logging.getLogger(__name__)
SIGNAL_COLUMNS = ["date", "first", "second", "action", "hedge_ratio"]


def run_walk_forward(
    data: Mapping[str, pd.DataFrame], prices: ContractPriceProvider, config: ProjectConfig
) -> pd.DataFrame:
    """Fresh agent per pair/window: no pretrained future knowledge is carried backwards."""
    frames = []
    for first_symbol, second_symbol in combinations(data, 2):
        first, second = align_pair(data[first_symbol], data[second_symbol])
        for window in walk_forward_windows(
            len(first), config.environment.lookback, config.walk_forward
        ):
            train_first, train_second = first.iloc[window.train], second.iloc[window.train]
            pvalue = cointegration_pvalue(train_first.open.to_numpy(), train_second.open.to_numpy())
            logger.info(
                "pair=%s/%s test=%s p=%.5f",
                first_symbol,
                second_symbol,
                first.index[window.test_start],
                pvalue,
            )
            if pvalue < config.walk_forward.significance:

                def environment(part: slice, first=first, second=second) -> PairTradingEnv:
                    return PairTradingEnv(
                        first.iloc[part], second.iloc[part], config.environment, prices
                    )

                agent = DQNAgent(
                    config.environment.state_size,
                    config.agent,
                    config.network,
                    config.training.seed,
                )
                Trainer(config.training).fit(
                    agent, environment(window.train), environment(window.validation)
                )
                result = evaluate(environment(window.test), agent).signals.reset_index()
            else:
                # Explicit flat targets prevent stale positions surviving rejected windows.
                result = pd.DataFrame(
                    {
                        "date": first.index[window.test_start : window.test_stop],
                        "action": int(Action.CLOSE),
                        "hedge_ratio": 1.0,
                    }
                )
            result["first"], result["second"] = first_symbol, second_symbol
            frames.append(result[SIGNAL_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "first", "second"])
