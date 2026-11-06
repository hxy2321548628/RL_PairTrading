"""Calendar-aligned train/validation/test windows with historical warmup."""

from collections.abc import Iterator
from dataclasses import dataclass

from rl_pairtrading.config import WalkForwardConfig


@dataclass(frozen=True)
class Window:
    train: slice
    validation: slice
    test: slice
    test_start: int
    test_stop: int


def walk_forward_windows(length: int, lookback: int, config: WalkForwardConfig) -> Iterator[Window]:
    """test_stop is exclusive for action dates; its price is the last outcome observation.

    Training and validation each stop before the next partition begins; their final
    observation has no action. This deliberately avoids crossing partition boundaries.
    """
    first = lookback + config.train_size + config.validation_size
    for number, start in enumerate(range(first, length - 1, config.test_size)):
        if config.max_windows is not None and number >= config.max_windows:
            break
        stop = min(start + config.test_size, length - 1)
        yield Window(
            train=slice(
                start - lookback - config.train_size - config.validation_size,
                start - config.validation_size,
            ),
            validation=slice(start - config.validation_size - lookback, start),
            test=slice(start - lookback, stop + 1),
            test_start=start,
            test_stop=stop,
        )
