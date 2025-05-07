from typing import TYPE_CHECKING, Any, List
import os

from zeroband.utils.envs import _BASE_ENV, get_env_value, get_dir

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import PRIME_LOG_LEVEL, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, CUDA_VISIBLE_DEVICES

    TRAINING_ENABLE_ACCEPTED_CHECK: bool = False
    PRIME_DASHBOARD_AUTH_TOKEN: str | None = None
    PRIME_API_BASE_URL: str | None = None
    PRIME_RUN_ID: str | None = None
    PRIME_DASHBOARD_METRIC_INTERVAL: int = 1
    SHARDCAST_OUTPUT_DIR: str | None = None

_TRAINING_ENV = {
    "TRAINING_ENABLE_ACCEPTED_CHECK": lambda: os.getenv("TRAINING_ENABLE_ACCEPTED_CHECK", "false").lower() in ["true", "1", "yes", "y"],
    "PRIME_API_BASE_URL": lambda: os.getenv("PRIME_API_BASE_URL"),
    "PRIME_DASHBOARD_AUTH_TOKEN": lambda: os.getenv("PRIME_DASHBOARD_AUTH_TOKEN"),
    "PRIME_RUN_ID": lambda: os.getenv("PRIME_RUN_ID"),
    "PRIME_DASHBOARD_METRIC_INTERVAL": lambda: int(os.getenv("PRIME_DASHBOARD_METRIC_INTERVAL", "1")),
    "SHARDCAST_OUTPUT_DIR": lambda: os.getenv("SHARDCAST_OUTPUT_DIR", None),
    **_BASE_ENV,
}


def __getattr__(name: str) -> Any:
    return get_env_value(_TRAINING_ENV, name)


def __dir__() -> List[str]:
    return get_dir(_TRAINING_ENV)
