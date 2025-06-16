import logging
import sys

from loguru import logger
from loguru._logger import Logger

from zeroband.inference.config import ParallelConfig, PipelineParallelConfig

ALLOWED_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}

_LOGGER: Logger | None = None


def setup_logger(level: str, parallel_config: ParallelConfig) -> Logger:
    global _LOGGER
    if _LOGGER is not None:
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Define the base format for the logger
    time = "<fg 0>{time:HH:mm:ss}</fg 0>"
    message = "<fg 0>[</fg 0> <level>{level: >8}</level> <fg 0>]</fg 0> <level>{message}</level>"
    debug = "PID={process.id} | TID={thread.id} | {file}::{line}" if level.upper() == "DEBUG" else ""

    # Add parallel information to the format
    parallel = []
    if parallel_config.dp.is_enabled:
        parallel.append(f"DP={parallel_config.dp.rank}")
    if parallel_config.pp.is_enabled:
        parallel.append(f"PP={parallel_config.pp.rank}")
    if parallel:
        if debug:
            debug += " | "
        debug += f"{' | '.join(parallel)}"

    format = f"{time} {debug} {message}"

    logger.remove()
    logger.add(sys.stdout, format=format, level=level.upper(), enqueue=True, backtrace=True, diagnose=True)

    _LOGGER = logger.bind(dp_rank=parallel_config.dp.rank, pp_rank=parallel_config.pp.rank)
    return _LOGGER


def get_logger() -> Logger:
    global _LOGGER
    if _LOGGER is None:
        raise RuntimeError("Logger not setup. Call setup_logger first.")

    return _LOGGER


def reset_logger() -> None:
    global _LOGGER
    _LOGGER = None


if __name__ == "__main__":
    logger = setup_logger(level="debug", parallel_config=ParallelConfig(pp=PipelineParallelConfig(world_size=2)))
    logger.debug("Debug message")
    logger.info("Info message")
    logger.success("Success message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    try:
        a = 1 / 0
    except Exception as e:
        logger.exception(e)
