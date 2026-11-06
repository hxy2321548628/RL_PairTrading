"""Domain vocabulary shared by training, signals and accounting."""

from enum import IntEnum, StrEnum


class Position(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class Action(IntEnum):
    LONG = 0
    SHORT = 1
    CLOSE = 2

    @property
    def position(self) -> Position:
        return {
            Action.LONG: Position.LONG,
            Action.SHORT: Position.SHORT,
            Action.CLOSE: Position.FLAT,
        }[self]


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Activation(StrEnum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"


class Exchange(StrEnum):
    DCE = "DCE"
    ZCE = "ZCE"
    SHFE = "SHFE"
    CFFEX = "CFFEX"
    INE = "INE"
    GFEX = "GFEX"
