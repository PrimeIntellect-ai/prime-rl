from typing import TYPE_CHECKING, Any

from prime_rl.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from prime_rl.utils.envs import *

    # PyTorch
    RANK: int
    WORLD_SIZE: int
    LOCAL_RANK: int
    LOCAL_WORLD_SIZE: int
    MASTER_ADDR: str
    MASTER_PORT: int


_TRAINING_ENV_PARSERS = {
    "RANK": int,
    "WORLD_SIZE": int,
    "LOCAL_RANK": int,
    "LOCAL_WORLD_SIZE": int,
    "MASTER_ADDR": str,
    "MASTER_PORT": int,
    **_BASE_ENV_PARSERS,
}

_TRAINING_ENV_DEFAULTS = {
    "RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_RANK": "0",
    "LOCAL_WORLD_SIZE": "1",
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "29500",
}

set_defaults(_TRAINING_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_TRAINING_ENV_PARSERS, name)


def __dir__() -> list[str]:
    return get_dir(_TRAINING_ENV_PARSERS)
