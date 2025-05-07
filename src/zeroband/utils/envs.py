from typing import TYPE_CHECKING, Any, Dict, List
import os

if TYPE_CHECKING:
    # Prime
    PRIME_LOG_LEVEL: str = "INFO"

    # PyTorch
    RANK: int = 0
    WORLD_SIZE: int = 1
    LOCAL_RANK: int = 0
    LOCAL_WORLD_SIZE: int = 1
    CUDA_VISIBLE_DEVICES: List[int] = [0]

# Shared environment variables between training and inference
_BASE_ENV: Dict[str, Any] = {
    "PRIME_LOG_LEVEL": lambda: os.getenv("PRIME_LOG_LEVEL", "INFO"),
    "RANK": lambda: int(os.getenv("RANK", "0")),
    "WORLD_SIZE": lambda: int(os.getenv("WORLD_SIZE", "1")),
    "LOCAL_RANK": lambda: int(os.getenv("LOCAL_RANK", "0")),
    "LOCAL_WORLD_SIZE": lambda: int(os.getenv("LOCAL_WORLD_SIZE", "1")),
    "CUDA_VISIBLE_DEVICES": lambda: list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))),
}


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
