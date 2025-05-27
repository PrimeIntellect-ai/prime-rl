import os
from typing import TYPE_CHECKING, Any, List

from zeroband.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import PRIME_LOG_LEVEL, CUDA_VISIBLE_DEVICES

    # vLLM
    VLLM_USE_V1: str
    VLLM_CONFIGURE_LOGGING: str

    # Rust
    RUST_LOG: str

    # Shardcast
    SHARDCAST_SERVERS: List[str] | None = None
    SHARDCAST_BACKLOG_VERSION: int = -1

    # Protocol
    GROUP_ID: str | None = None

_INFERENCE_ENV_PARSERS = {
    "VLLM_USE_V1": str,
    "VLLM_CONFIGURE_LOGGING": str,
    "SHARDCAST_SERVERS": lambda x: x.split(","),
    "SHARDCAST_BACKLOG_VERSION": int,
    "GROUP_ID": str,
    "RUST_LOG": str,
    **_BASE_ENV_PARSERS,
}

_INFERENCE_ENV_DEFAULTS = {
    "SHARDCAST_BACKLOG_VERSION": "-1",
    "VLLM_CONFIGURE_LOGGING": "0",  # Disable vLLM logging unless explicitly enabled
    "VLLM_USE_V1": "0",  # Use v0 engine (TOPLOC and PP do not support v1 yet)
    "RUST_LOG": "off",  # Disable Rust logs (from prime-iroh)
}

set_defaults(_INFERENCE_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_INFERENCE_ENV_PARSERS, name)


def __dir__() -> List[str]:
    return get_dir(_INFERENCE_ENV_PARSERS)
