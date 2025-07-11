from typing import TYPE_CHECKING, Any

from prime_rl.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from prime_rl.utils.envs import *

    # vLLM
    VLLM_CONFIGURE_LOGGING: int

    # tqdm
    TQDM_DISABLE: int


_ORCHESTRATOR_ENV_PARSERS = {
    "VLLM_CONFIGURE_LOGGING": int,
    "TQDM_DISABLE": int,
    **_BASE_ENV_PARSERS,
}

_ORCHESTRATOR_ENV_DEFAULTS = {
    "VLLM_CONFIGURE_LOGGING": "0",
}

set_defaults(_ORCHESTRATOR_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_ORCHESTRATOR_ENV_PARSERS, name)


def __dir__() -> list[str]:
    return get_dir(_ORCHESTRATOR_ENV_PARSERS)
