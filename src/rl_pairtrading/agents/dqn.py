"""DQN components with independent target parameters and isolated replay RNG."""

import math
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from rl_pairtrading.config import AgentConfig, NetworkConfig
from rl_pairtrading.domain import Action, Activation, Device


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self._buffer.append(
            Transition(
                transition.state.copy(),
                int(transition.action),
                transition.reward,
                transition.next_state.copy(),
                transition.done,
            )
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if not 0 < batch_size <= len(self):
            raise ValueError("Batch size must be positive and no greater than replay size")
        batch = self._random.sample(list(self._buffer), batch_size)
        return tuple(np.stack(values) for values in zip(*batch, strict=True))

    def __len__(self) -> int:
        return len(self._buffer)


def resolve_device(device: Device) -> torch.device:
    if device == Device.AUTO:
        if torch.cuda.is_available():
            return torch.device(Device.CUDA.value)
        if torch.backends.mps.is_available():
            return torch.device(Device.MPS.value)
        return torch.device(Device.CPU.value)
    if device == Device.CUDA and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use auto or cpu")
    if device == Device.MPS and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable; use auto or cpu")
    return torch.device(device.value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, config: NetworkConfig):
        super().__init__()
        activations = {
            Activation.RELU: nn.ReLU,
            Activation.LEAKY_RELU: nn.LeakyReLU,
            Activation.TANH: nn.Tanh,
        }
        sizes = (state_size, *config.hidden_sizes, len(Action))
        layers: list[nn.Module] = []
        for index, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            layers.append(nn.Linear(inputs, outputs))
            if index < len(sizes) - 2:
                layers.append(activations[config.activation]())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    def __init__(self, state_size: int, config: AgentConfig, network: NetworkConfig, seed: int):
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(seed)
        self.policy_net = PolicyNetwork(state_size, network).to(self.device)
        self.target_net = deepcopy(self.policy_net).to(self.device)
        self.target_net.requires_grad_(False)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.replay_capacity, seed)
        self._random = random.Random(seed)
        self.sample_count = 0
        self.loss_function = nn.SmoothL1Loss()

    @property
    def epsilon(self) -> float:
        cfg = self.config
        return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -self.sample_count / cfg.epsilon_decay
        )

    def sample_action(self, state: np.ndarray) -> Action:
        self.sample_count += 1
        if self._random.random() < self.epsilon:
            return Action(self._random.randrange(len(Action)))
        return self.predict_action(state)

    @torch.no_grad()
    def predict_action(self, state: np.ndarray) -> Action:
        values = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return Action(int(self.policy_net(values).argmax(dim=1).item()))

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update(self) -> float | None:
        if len(self.memory) < self.config.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        predicted = self.policy_net(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            expected = (
                rewards + self.config.gamma * self.target_net(next_states).max(1).values * ~dones
            )
        loss = self.loss_function(predicted, expected)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite DQN loss; inspect input features and learning rate"
            )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        return float(loss.item())

    def load_weights(self, path: Path) -> None:
        """Inference/transfer weights, not a full optimizer/replay resume checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state)
        self.sync_target()
