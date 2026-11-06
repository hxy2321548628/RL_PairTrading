"""Validation-only model selection; the test set is absent from this API."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from rl_pairtrading.agents.dqn import DQNAgent, Transition
from rl_pairtrading.config import TrainingConfig
from rl_pairtrading.domain import Action
from rl_pairtrading.environment.pair_trading import PairTradingEnv

logger = logging.getLogger(__name__)


class Policy(Protocol):
    def predict_action(self, state: np.ndarray) -> Action: ...


@dataclass(frozen=True)
class EvaluationResult:
    reward: float
    signals: pd.DataFrame


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_episode: int
    best_validation_reward: float


def evaluate(env: PairTradingEnv, policy: Policy) -> EvaluationResult:
    state = env.reset()
    total_reward = 0.0
    records = []
    done = False
    while not done:
        action = policy.predict_action(state)
        date, beta = env.date, env.beta
        state, reward, done = env.step(action)
        records.append({"date": date, "action": int(action), "hedge_ratio": beta, "reward": reward})
        total_reward += reward
    return EvaluationResult(total_reward, pd.DataFrame(records).set_index("date"))


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def fit(
        self,
        agent: DQNAgent,
        train_env: PairTradingEnv,
        validation_env: PairTradingEnv,
        *,
        checkpoint: Path | None = None,
    ) -> TrainingResult:
        best_score, best_episode, stale = -np.inf, 0, 0
        best_weights = None
        records = []
        for episode in range(1, self.config.episodes + 1):
            state, done, reward_sum = train_env.reset(), False, 0.0
            losses = []
            while not done:
                action = agent.sample_action(state)
                next_state, reward, done = train_env.step(action)
                agent.memory.push(Transition(state, int(action), reward, next_state, done))
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                reward_sum += reward
                state = next_state
            if episode % self.config.target_update_episodes == 0:
                agent.sync_target()
            score = evaluate(validation_env, agent).reward
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation reward")
            records.append(
                {
                    "episode": episode,
                    "train_reward": reward_sum,
                    "validation_reward": score,
                    "loss": float(np.mean(losses)) if losses else np.nan,
                    "epsilon": agent.epsilon,
                }
            )
            logger.info("episode=%s train=%.4f validation=%.4f", episode, reward_sum, score)
            if score > best_score + self.config.min_improvement:
                best_score, best_episode, stale = score, episode, 0
                best_weights = deepcopy(agent.policy_net.state_dict())
            else:
                stale += 1
            if stale >= self.config.patience:
                break
        agent.policy_net.load_state_dict(best_weights)
        agent.sync_target()
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: value.cpu() for key, value in best_weights.items()}, checkpoint)
        return TrainingResult(pd.DataFrame(records).set_index("episode"), best_episode, best_score)
