import sys

from loguru import logger
from loguru._logger import Logger

from zeroband.training.config import LogConfig
from zeroband.training.world_info import WorldInfo
from zeroband.utils.logger import get_logger, set_logger

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def setup_logger(log_config: LogConfig, world_info: WorldInfo) -> Logger:
    if get_logger() is not None:
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
    if world_info.num_gpus > 1:
        parallel = f"Rank {world_info.rank}"
        if debug:
            debug += " | "
        debug += parallel
    if debug:
        debug = f"[{debug}]"

    # Assemble the final format
    format = f"{time} {debug} {message}"

    # Remove all default handlers
    logger.remove()

    # Install new handler on all ranks, if specified. Otherwise, only install on the main rank
    if log_config.all_ranks or world_info.rank == 0:
        logger.add(sys.stdout, format=format, level=log_config.level.upper(), enqueue=True, backtrace=True, diagnose=True)

    # Bind the logger to access the rank
    set_logger(logger)

    return logger


if __name__ == "__main__":
    from zeroband.training.world_info import get_world_info

    world_info = get_world_info()
    logger = setup_logger(log_config=LogConfig(utc=True, all_ranks=True), world_info=world_info)
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
