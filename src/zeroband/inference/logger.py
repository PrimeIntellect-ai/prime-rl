import logging
import os
import sys

from loguru import logger

import zeroband.inference.envs as env

_LOGGER = None


def setup_logger(tag: str) -> None:
    """Setup multi-process loguru.logger"""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    # Format logger
    format = "<level>{level: <5}</level> | <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{extra[tag]}</cyan> - {message}"
    logger.configure(extra={"tag": tag})
    logger.remove()
    logger.add(sys.stderr, format=format, enqueue=True, level=env.PRIME_LOG_LEVEL)

    # Set global logger
    _LOGGER = logger

    # Turn off vLLM and safetensor logging if log severity is at least INFO
    log_level = getattr(logging, env.PRIME_LOG_LEVEL, logging.INFO)
    if log_level >= logging.INFO:
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"
        os.environ["SAFETENSORS_FAST_GPU"] = "0"
        os.environ["TQDM_DISABLE"] = "1"

    logging.disable(logging.CRITICAL)

    # Disable rust logs if not manually set
    if os.environ.get("RUST_LOG") is None:
        os.environ["RUST_LOG"] = "off"


def get_logger():
    """Get global logger instance."""
    assert _LOGGER is not None, "Logger not initialized"
    return _LOGGER
