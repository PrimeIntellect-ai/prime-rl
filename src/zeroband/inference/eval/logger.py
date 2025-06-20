import sys

from loguru import logger
from loguru._logger import Logger

from zeroband.inference.config import LogConfig
from zeroband.utils.logger import get_logger, set_logger

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def setup_logger(log_config: LogConfig) -> Logger:
    if get_logger() is not None:
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    # Define the time format for the logger.
    time = "<dim>{time:HH:mm:ss}</dim>"
    if log_config.utc:
        time = "<dim>{time:zz HH:mm:ss!UTC}</dim>"

    # Define the colorized log level and message
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            "{message}",
            f"{RESET}</level>",
        ]
    )

    # Define the debug information in debug mode
    debug = "PID={process.id} | {file}::{line}" if log_config.level.upper() == "DEBUG" else ""

    # Assemble the final format
    format = time + debug + message

    # Install new handler
    logger.remove()
    logger.add(sys.stdout, format=format, level=log_config.level.upper(), enqueue=True, backtrace=True, diagnose=True)

    # Disable critical logging
    logger.critical = lambda _: None

    # Bind the logger to access the DP and PP rank
    set_logger(logger)

    return logger
