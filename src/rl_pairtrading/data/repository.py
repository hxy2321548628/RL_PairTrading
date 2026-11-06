"""Validated local market data and an injectable contract price interface."""

from typing import Protocol

import numpy as np
import pandas as pd

from rl_pairtrading.config import ProjectConfig

REQUIRED_COLUMNS = ("open", "settlement", "volume", "open_interest", "dominant_id")


class DataValidationError(ValueError):
    """Market data cannot satisfy the execution contract."""


class ContractPriceProvider(Protocol):
    def open_price(self, contract: str, date: pd.Timestamp) -> float: ...


def validate_market_data(frame: pd.DataFrame, name: str = "market") -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataValidationError(f"{name}: missing columns {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise DataValidationError(f"{name}: require a non-null DatetimeIndex")
    if frame.empty or frame.index.has_duplicates:
        raise DataValidationError(f"{name}: empty data or duplicate dates")
    result = frame.sort_index().copy()
    numeric = result[list(REQUIRED_COLUMNS[:-1])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataValidationError(f"{name}: non-finite market values")
    if (numeric[:, :2] <= 0).any() or (numeric[:, 2:] < 0).any():
        raise DataValidationError(f"{name}: prices must be positive, volume and OI nonnegative")
    if result.dominant_id.isna().any() or result.dominant_id.astype(str).str.len().eq(0).any():
        raise DataValidationError(f"{name}: missing dominant contract")
    result["dominant_id"] = result.dominant_id.astype(str)
    return result


def align_pair(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, second = validate_market_data(first), validate_market_data(second)
    dates = first.index.intersection(second.index).sort_values()
    if dates.empty:
        raise DataValidationError("Pair has no common trading dates")
    return first.loc[dates].copy(), second.loc[dates].copy()


class LocalDataRepository:
    """Only load trusted local pickle files. Never modifies source data."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def load(self, symbol: str) -> pd.DataFrame:
        instrument = self.config.instrument(symbol)
        path = self.config.paths.data_dir / instrument.data_file
        return validate_market_data(pd.read_pickle(path), symbol)

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            instrument.symbol: self.load(instrument.symbol)
            for instrument in self.config.instruments
        }

    def contract_prices(self) -> "LocalContractPrices":
        path = self.config.paths.data_dir / self.config.paths.contracts_file
        return LocalContractPrices(pd.read_pickle(path))


class LocalContractPrices:
    """Lookup old-contract opens at rollover; missing quotes fail explicitly."""

    def __init__(self, frame: pd.DataFrame):
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
            raise DataValidationError("Contract quotes require index (contract, date)")
        if "open" not in frame or frame.index.has_duplicates:
            raise DataValidationError("Contract quotes need unique keys and an open column")
        self._quotes = frame.copy()
        self._quotes.index = pd.MultiIndex.from_arrays(
            [
                frame.index.get_level_values(0).astype(str),
                pd.to_datetime(frame.index.get_level_values(1)),
            ]
        )
        if self._quotes.index.has_duplicates:
            raise DataValidationError("Duplicate normalized contract quote keys")

    def open_price(self, contract: str, date: pd.Timestamp) -> float:
        try:
            price = float(self._quotes.loc[(contract, pd.Timestamp(date)), "open"])
        except KeyError as exc:
            raise DataValidationError(f"Missing rollover open: {contract} at {date}") from exc
        if not np.isfinite(price) or price <= 0:
            raise DataValidationError(f"Invalid rollover open: {contract} at {date}")
        return price
