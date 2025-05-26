from typing import TYPE_CHECKING, Any
import os

from zeroband.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import PRIME_LOG_LEVEL, CUDA_VISIBLE_DEVICES

    # Prime
    TRAINING_ENABLE_ACCEPTED_CHECK: bool = False
    PRIME_DASHBOARD_AUTH_TOKEN: str | None = None
    PRIME_API_BASE_URL: str | None = None
    PRIME_RUN_ID: str | None = None
    PRIME_DASHBOARD_METRIC_INTERVAL: int = 1
    SHARDCAST_OUTPUT_DIR: str | None = None

    # PyTorch
    RANK: int
    WORLD_SIZE: int
    LOCAL_RANK: int
    LOCAL_WORLD_SIZE: int


_TRAINING_ENV_PARSERS = {
    "RANK": int,
    "WORLD_SIZE": int,
    "LOCAL_RANK": int,
    "LOCAL_WORLD_SIZE": int,
    "TRAINING_ENABLE_ACCEPTED_CHECK": lambda x: x.lower() in ["true", "1", "yes", "y"],
    "PRIME_API_BASE_URL": str,
    "PRIME_DASHBOARD_AUTH_TOKEN": str,
    "PRIME_RUN_ID": lambda: str,
    "PRIME_DASHBOARD_METRIC_INTERVAL": int,
    "SHARDCAST_OUTPUT_DIR": str,
    **_BASE_ENV_PARSERS,
}

_TRAINING_ENV_DEFAULTS = {
    "PRIME_DASHBOARD_METRIC_INTERVAL": "1",
    "RANK": "0",
    "WORLD_SIZE": "1",
    "LOCAL_RANK": "0",
    "LOCAL_WORLD_SIZE": "1",
}

set_defaults(_TRAINING_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_TRAINING_ENV_PARSERS, name)


def __dir__() -> list[str]:
    return get_dir(_TRAINING_ENV_PARSERS)
