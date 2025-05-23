import os
from typing import TYPE_CHECKING, Any, Dict, List

# Force using vLLM v0
os.environ["VLLM_USE_V1"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid HF warning

# Disable rust logs if not manually set
if os.getenv("RUST_LOG") is None:
    os.environ["RUST_LOG"] = "off"

if os.getenv("LOG_LEVEL") is None:
    os.environ["LOG_LEVEL"] = "ERROR"

if TYPE_CHECKING:
    # Prime
    PRIME_LOG_LEVEL: str

    # Rust
    RUST_LOG: str

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
    "RUST_LOG": lambda: os.getenv("RUST_LOG", "off"),
    "VLLM_USE_V1": lambda: os.getenv("VLLM_USE_V1", "0"),
    "RANK": lambda: int(os.getenv("RANK", "0")),
    "WORLD_SIZE": lambda: int(os.getenv("WORLD_SIZE", "1")),
    "LOCAL_RANK": lambda: int(os.getenv("LOCAL_RANK", "0")),
    "LOCAL_WORLD_SIZE": lambda: int(os.getenv("LOCAL_WORLD_SIZE", "1")),
    "CUDA_VISIBLE_DEVICES": lambda: list(map(int, os.getenv("CUDA_VISIBLE_DEVICES", "0").split(","))),
    "VLLM_USE_V1": lambda: os.getenv("VLLM_USE_V1", "0"),
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
