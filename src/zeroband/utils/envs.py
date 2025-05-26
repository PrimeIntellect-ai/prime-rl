import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    # Prime
    PRIME_LOG_LEVEL: str

    # vLLM
    VLLM_USE_V1: str

    # PyTorch
    RANK: int
    WORLD_SIZE: int
    LOCAL_RANK: int
    LOCAL_WORLD_SIZE: int
    CUDA_VISIBLE_DEVICES: List[int]

# Shared environment variables between training and inference
_BASE_ENV: Dict[str, Any] = {
    "PRIME_LOG_LEVEL": lambda: os.getenv("PRIME_LOG_LEVEL", "INFO"),
    "VLLM_USE_V1": lambda: os.getenv("VLLM_USE_V1", "0"),
    "RANK": lambda: int(os.getenv("RANK")) if os.getenv("RANK") is not None else None,
    "WORLD_SIZE": lambda: int(os.getenv("WORLD_SIZE")) if os.getenv("WORLD_SIZE") is not None else None,
    "LOCAL_RANK": lambda: int(os.getenv("LOCAL_RANK")) if os.getenv("LOCAL_RANK") is not None else None,
    "LOCAL_WORLD_SIZE": lambda: int(os.getenv("LOCAL_WORLD_SIZE")) if os.getenv("LOCAL_WORLD_SIZE") is not None else None,
    "CUDA_VISIBLE_DEVICES": lambda: list(map(int, os.getenv("CUDA_VISIBLE_DEVICES").split(",")))
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None
    else None,
}

# Set all environment variables (if default is used)
for key, value in _BASE_ENV.items():
    if value() is not None:
        os.environ[key] = str(value())

# Dynamically set external logging based on PRIME_LOG_LEVEL
log_level = getattr(logging, os.environ["PRIME_LOG_LEVEL"], logging.INFO)
if log_level >= logging.INFO:
    # Disable vLLM and Rust logs if log level is INFO or higher
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["RUST_LOG"] = "off"
else:
    # Enable vLLm and specific Rust logs if log level is DEBUG or higher
    os.environ["RUST_LOG=prime-iroh"] = "debug"


def get_env_value(envs: Dict[str, Any], key: str) -> Any:
    if key not in envs:
        raise AttributeError(f"Invalid environment variable: {key}")
    return envs[key]()


def get_dir(envs: Dict[str, Any]) -> List[str]:
    return list(envs.keys())


def __getattr__(name: str) -> Any:
    return get_env_value(_BASE_ENV, name)


def __dir__() -> List[str]:
    return get_dir(_BASE_ENV)
