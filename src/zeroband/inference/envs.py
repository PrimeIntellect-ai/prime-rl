import os
from typing import TYPE_CHECKING, Any, List

from zeroband.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import *

    # vLLM
    VLLM_USE_V1: str
    VLLM_CONFIGURE_LOGGING: str

    # HF
    HF_HUB_CACHE: str
    HF_HUB_DISABLE_PROGRESS_BARS: str
    HF_HUB_ETAG_TIMEOUT: int

    # Rust
    RUST_LOG: str

    # Shardcast
    SHARDCAST_SERVERS: List[str] | None = None
    SHARDCAST_BACKLOG_VERSION: int = -1

_INFERENCE_ENV_PARSERS = {
    "VLLM_USE_V1": str,
    "VLLM_CONFIGURE_LOGGING": str,
    "HF_HUB_CACHE": str,
    "HF_HUB_DISABLE_PROGRESS_BARS": str,
    "HF_HUB_ETAG_TIMEOUT": int,
    "RUST_LOG": str,
    "SHARDCAST_SERVERS": lambda x: x.split(","),
    "SHARDCAST_BACKLOG_VERSION": int,
    **_BASE_ENV_PARSERS,
}

_INFERENCE_ENV_DEFAULTS = {
    "SHARDCAST_BACKLOG_VERSION": "-1",
    "VLLM_CONFIGURE_LOGGING": "0",  # Disable vLLM logging unless explicitly enabled
    "VLLM_USE_V1": "0",  # Use v0 engine (TOPLOC and PP do not support v1 yet)
    "RUST_LOG": "off",  # Disable Rust logs (from prime-iroh)
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Disable HF progress bars
    "HF_HUB_ETAG_TIMEOUT": "500",  # Set request timeout to 500s to avoid model download issues
}

set_defaults(_INFERENCE_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_INFERENCE_ENV_PARSERS, name)


def __dir__() -> List[str]:
    return get_dir(_INFERENCE_ENV_PARSERS)
