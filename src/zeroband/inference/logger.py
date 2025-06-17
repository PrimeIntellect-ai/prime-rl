import sys

from loguru import logger
from loguru._logger import Logger

from zeroband.inference.config import LogConfig, ParallelConfig

_LOGGER: Logger | None = None
NO_BOLD = "\033[22m"
RESET = "\033[0m"


def setup_logger(log_config: LogConfig, parallel_config: ParallelConfig) -> Logger:
    global _LOGGER
    if _LOGGER is not None:
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Define the time format for the logger.
    time = "<fg 0>{time:zz HH:mm:ss}</fg 0>"
    if log_config.utc:
        time = "<fg 0>{time:zz HH:mm:ss!UTC}</fg 0>"

    # Define the colorized log level and message
    message = "".join(
        [
            "<level>{level: >8}</level>",
            f" <level>{NO_BOLD}",
            "{message}",
            f"{RESET}</level>",
        ]
    )

    # Define the debug information in debug mode
    debug = "PID={process.id} | TID={thread.id} | {file}::{line}" if log_config.level.upper() == "DEBUG" else ""

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
    if debug:
        debug = f"[{debug}]"

    # Assemble the final format
    format = f"{time} {debug} {message}"

    # Remove all default handlers
    logger.remove()

    # Install new handler on all ranks, if specified. Otherwise, only install on the main rank
    if log_config.all_ranks or parallel_config.dp.rank == 0:
        logger.add(sys.stdout, format=format, level=log_config.level.upper(), enqueue=True, backtrace=True, diagnose=True)

    # Bind the logger to access the DP and PP rank
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
    logger = setup_logger(log_config=LogConfig(utc=True), parallel_config=ParallelConfig())
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
