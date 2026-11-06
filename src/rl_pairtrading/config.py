"""Validated, immutable experiment configuration. No import-time I/O."""

import tomllib
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from rl_pairtrading.domain import Activation, Device, Exchange

Probability = Annotated[float, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PathsConfig(ConfigModel):
    data_dir: Path = Path("datas")
    output_dir: Path = Path("artifacts")
    contracts_file: str = "future_contract.pkl"


class InstrumentConfig(ConfigModel):
    symbol: str = Field(pattern=r"^[A-Z]+$")
    name: str
    exchange: Exchange = Exchange.DCE
    data_file: str
    multiplier: PositiveFloat = 1.0
    margin_rate: Annotated[float, Field(gt=0, le=1)] = 0.1


class EnvironmentConfig(ConfigModel):
    lookback: PositiveInt = 750
    short_window: PositiveInt = 5
    long_window: PositiveInt = 20
    spread_lags: PositiveInt = 5
    hedge_refresh_days: PositiveInt = 180
    commission_per_leg: NonNegativeFloat = 0.25
    bollinger_width: PositiveFloat = 2.0
    numerical_epsilon: PositiveFloat = 1e-8

    @model_validator(mode="after")
    def valid_windows(self) -> Self:
        if not self.short_window < self.long_window <= self.lookback:
            raise ValueError("Require short_window < long_window <= lookback")
        if self.spread_lags > self.lookback or self.lookback < 4:
            raise ValueError("lookback must cover spread_lags and at least four observations")
        return self

    @property
    def state_size(self) -> int:
        # 11 numeric features, 3 position indicators, configurable spread history.
        return 11 + 3 + self.spread_lags


class NetworkConfig(ConfigModel):
    hidden_sizes: tuple[PositiveInt, ...] = (60, 120, 60, 30, 10)
    activation: Activation = Activation.RELU


class AgentConfig(ConfigModel):
    device: Device = Device.AUTO
    gamma: Probability = 0.9
    learning_rate: PositiveFloat = 1e-5
    epsilon_start: Probability = 1.0
    epsilon_end: Probability = 1e-9
    epsilon_decay: PositiveFloat = 4000.0
    batch_size: PositiveInt = 64
    replay_capacity: PositiveInt = 10000
    gradient_clip: PositiveFloat = 10.0

    @model_validator(mode="after")
    def valid_replay(self) -> Self:
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        return self


class TrainingConfig(ConfigModel):
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 41
    episodes: PositiveInt = 100
    target_update_episodes: PositiveInt = 10
    patience: PositiveInt = 20
    min_improvement: NonNegativeFloat = 0.0


class WalkForwardConfig(ConfigModel):
    train_size: PositiveInt = 940
    validation_size: Annotated[int, Field(ge=2)] = 60
    test_size: PositiveInt = 120
    significance: Annotated[float, Field(gt=0, lt=1)] = 0.05
    max_windows: PositiveInt | None = None

    @model_validator(mode="after")
    def enough_training(self) -> Self:
        if self.train_size < 2:
            raise ValueError("train_size must contain at least two observations")
        return self


class BacktestConfig(ConfigModel):
    initial_cash: PositiveFloat = 20000.0
    commission_per_contract: NonNegativeFloat = 1.2
    base_quantity: PositiveFloat = 1.0
    liquidate_at_end: bool = True
    annualization: PositiveInt = 252
    risk_free_rate: Annotated[float, Field(gt=-1)] = 0.0


class ResearchConfig(ConfigModel):
    primary_pair: tuple[str, str] = ("CS", "C")
    run_walk_forward: bool = False


class ProjectConfig(ConfigModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    instruments: tuple[InstrumentConfig, ...]
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)

    @model_validator(mode="after")
    def validate_symbols(self) -> Self:
        symbols = [instrument.symbol for instrument in self.instruments]
        if len(symbols) < 2 or len(symbols) != len(set(symbols)):
            raise ValueError("Provide at least two instruments with unique symbols")
        a, b = self.research.primary_pair
        if a == b or a not in symbols or b not in symbols:
            raise ValueError("primary_pair must reference two different configured instruments")
        return self

    def instrument(self, symbol: str) -> InstrumentConfig:
        for instrument in self.instruments:
            if instrument.symbol == symbol:
                return instrument
        raise KeyError(f"Unknown instrument: {symbol}")


class Credentials(BaseSettings):
    """Explicitly instantiate only for online access; excluded from experiment snapshots."""

    model_config = SettingsConfigDict(extra="ignore", env_file=None)
    ts_token: SecretStr | None = None


def load_config(path: str | Path, *, project_root: Path | None = None) -> ProjectConfig:
    """Read TOML. Relative paths resolve against project_root, defaulting to configs/.. ."""
    path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root else path.parent.parent
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    paths = raw.setdefault("paths", {})
    for key, default in (("data_dir", "datas"), ("output_dir", "artifacts")):
        value = Path(paths.get(key, default)).expanduser()
        paths[key] = value if value.is_absolute() else root / value
    return ProjectConfig.model_validate(raw)
