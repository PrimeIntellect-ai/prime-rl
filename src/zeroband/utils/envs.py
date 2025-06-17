import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    # PyTorch
    CUDA_VISIBLE_DEVICES: list[int]


# All environment variables with their type parsers
_ENV_PARSERS: dict[str, Callable[[str], Any]] = {
    "CUDA_VISIBLE_DEVICES": lambda x: list(map(int, x.split(","))),
}

# Subset of environment variables with default values
_ENV_DEFAULTS: dict[str, str] = {}


# Initialize environment variables that have defaults
def set_defaults(env_defaults: dict[str, str]):
    for key, default_value in env_defaults.items():
        if os.getenv(key) is None:
            os.environ[key] = default_value


set_defaults(_ENV_DEFAULTS)


def get_env_value(env_parser: dict[str, Callable[[str], Any]], key: str) -> Any:
    """Get an environment variable value with type safety and parsing."""
    if key not in env_parser:
        raise AttributeError(f"Invalid environment variable: {key}")

    raw_value = os.getenv(key)
    if raw_value is None:
        return None
    parser = env_parser[key]

    return parser(raw_value)


def get_dir(env_parser: dict[str, Callable[[str], Any]]) -> list[str]:
    """Get list of available environment variable names."""
    return list(env_parser.keys())


def __getattr__(name: str) -> Any:
    return get_env_value(_ENV_PARSERS, name)


def __dir__() -> list[str]:
    return get_dir(_ENV_PARSERS)
