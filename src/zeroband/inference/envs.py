import os
from typing import TYPE_CHECKING, Any, List

from zeroband.utils.envs import _BASE_ENV, get_env_value, get_dir

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import PRIME_LOG_LEVEL, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, CUDA_VISIBLE_DEVICES

    SHARDCAST_SERVERS: List[str] | None = None
    SHARDCAST_BACKLOG_VERSION: int = -1
    NODE_ADDRESS: str | None = None

_INFERENCE_ENV = {
    "SHARDCAST_SERVERS": lambda: os.getenv("SHARDCAST_SERVERS", None).split(",")
    if os.getenv("SHARDCAST_SERVERS", None) is not None
    else None,
    "SHARDCAST_BACKLOG_VERSION": lambda: int(os.getenv("SHARDCAST_BACKLOG_VERSION", "-1")),
    "NODE_ADDRESS": lambda: os.getenv("NODE_ADDRESS", None),
    **_BASE_ENV,
}

def __getattr__(name: str) -> Any:
    return get_env_value(_INFERENCE_ENV, name)


def __dir__() -> List[str]:
    return get_dir(_INFERENCE_ENV)
