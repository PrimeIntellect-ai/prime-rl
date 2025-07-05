from typing import TYPE_CHECKING, Any, List

from zeroband.utils.envs import (
    _ENV_PARSERS as _BASE_ENV_PARSERS,
    get_env_value,
    get_dir,
    set_defaults,
)

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from zeroband.utils.envs import *

    # HF
    HF_HUB_CACHE: str
    HF_HUB_DISABLE_PROGRESS_BARS: str
    HF_HUB_ETAG_TIMEOUT: int

_EVAL_ENV_PARSERS = {
    "HF_HUB_CACHE": str,
    "HF_HUB_DISABLE_PROGRESS_BARS": str,
    "HF_HUB_ETAG_TIMEOUT": int,
    **_BASE_ENV_PARSERS,
}

_EVAL_ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",  # Disable HF progress bars
    "HF_HUB_ETAG_TIMEOUT": "500",  # Set request timeout to 500s to avoid model download issues
}

set_defaults(_EVAL_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_EVAL_ENV_PARSERS, name)


def __dir__() -> List[str]:
    return get_dir(_EVAL_ENV_PARSERS)
