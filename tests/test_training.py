import inspect

import numpy as np
import pandas as pd
import torch

from rl_pairtrading.agents.dqn import DQNAgent, ReplayBuffer, Transition
from rl_pairtrading.environment.pair_trading import PairTradingEnv
from rl_pairtrading.training.trainer import Trainer, evaluate
from rl_pairtrading.training.walk_forward import run_walk_forward
from rl_pairtrading.training.windows import walk_forward_windows


def test_target_independence(config):
    agent = DQNAgent(
        config.environment.state_size, config.agent, config.network, config.training.seed
    )
    original = [parameter.clone() for parameter in agent.target_net.parameters()]
    for _ in range(config.agent.batch_size):
        agent.memory.push(
            Transition(
                np.ones(config.environment.state_size),
                0,
                2.0,
                np.zeros(config.environment.state_size),
                True,
            )
        )
    assert np.isfinite(agent.update())
    for before, target in zip(original, agent.target_net.parameters(), strict=True):
        assert torch.equal(before, target)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(agent.policy_net.parameters(), agent.target_net.parameters(), strict=True)
    )
    agent.sync_target()
    for policy, target in zip(
        agent.policy_net.parameters(), agent.target_net.parameters(), strict=True
    ):
        assert torch.equal(policy, target) and policy.data_ptr() != target.data_ptr()


def test_buffer_copies():
    memory, state = ReplayBuffer(2, 1), np.zeros(3)
    memory.push(Transition(state, 0, 1, state, False))
    state[:] = 9
    assert not memory.sample(1)[0].any()
    for _ in range(3):
        memory.push(Transition(state, 0, 1, state, True))
    assert len(memory) == 2


def test_training_checkpoint(config, market, prices, tmp_path):
    window = next(walk_forward_windows(90, config.environment.lookback, config.walk_forward))

    def env(part):
        return PairTradingEnv(
            market["CS"].iloc[part], market["C"].iloc[part], config.environment, prices
        )

    agent = DQNAgent(
        config.environment.state_size, config.agent, config.network, config.training.seed
    )
    result = Trainer(config.training).fit(
        agent, env(window.train), env(window.validation), checkpoint=tmp_path / "weights.pt"
    )
    assert result.best_episode >= 1
    assert "test_env" not in inspect.signature(Trainer.fit).parameters
    evaluation = evaluate(env(window.test), agent)
    assert evaluation.signals.index.equals(market["CS"].index[window.test_start : window.test_stop])
    reloaded = DQNAgent(config.environment.state_size, config.agent, config.network, 99)
    reloaded.load_weights(tmp_path / "weights.pt")
    pd.testing.assert_frame_equal(evaluate(env(window.test), reloaded).signals, evaluation.signals)


def test_window_boundaries(config):
    cfg = config.walk_forward.model_copy(update={"max_windows": None})
    windows = list(walk_forward_windows(87, config.environment.lookback, cfg))
    dates = []
    for window in windows:
        assert window.train.stop <= window.validation.start + config.environment.lookback
        assert window.validation.stop == window.test_start
        assert window.test.stop == window.test_stop + 1
        dates.extend(range(window.test_start, window.test_stop))
    assert dates == list(range(windows[0].test_start, 86))


def test_walk_forward_branches(config, market, prices, monkeypatch):
    monkeypatch.setattr(
        "rl_pairtrading.training.walk_forward.cointegration_pvalue", lambda *args: 1.0
    )
    signals = run_walk_forward(market, prices, config)
    assert len(signals) == config.walk_forward.test_size and (signals.action == 2).all()
    monkeypatch.setattr(
        "rl_pairtrading.training.walk_forward.cointegration_pvalue", lambda *args: 0.0
    )
    signals = run_walk_forward(market, prices, config)
    assert len(signals) == config.walk_forward.test_size
    assert not signals.duplicated(["date", "first", "second"]).any()
    assert np.isfinite(signals.hedge_ratio).all()


def test_early_stopping_restores_best_snapshot(config, market, prices, monkeypatch):
    from rl_pairtrading.training.trainer import EvaluationResult

    raw = config.model_dump()
    raw["training"].update(episodes=5, patience=1)
    from rl_pairtrading.config import ProjectConfig

    config = ProjectConfig.model_validate(raw)
    env = PairTradingEnv(market["CS"], market["C"], config.environment, prices)
    agent = DQNAgent(
        config.environment.state_size, config.agent, config.network, config.training.seed
    )
    snapshots = []

    def validation(_env, current):
        snapshots.append(
            {key: value.clone() for key, value in current.policy_net.state_dict().items()}
        )
        return EvaluationResult(10.0 if len(snapshots) == 1 else 0.0, pd.DataFrame())

    monkeypatch.setattr("rl_pairtrading.training.trainer.evaluate", validation)
    result = Trainer(config.training).fit(agent, env, env)
    assert result.best_episode == 1 and len(result.history) == 2
    for key, value in agent.policy_net.state_dict().items():
        assert torch.equal(value, snapshots[0][key])
