import sys

from loguru._logger import Logger

from prime_rl.utils.config import LogConfig

# Global loguru logger instance
_LOGGER: Logger | None = None

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def format_time(config: LogConfig) -> str:
    time = "<dim>{time:HH:mm:ss}</dim>"
    if config.utc:
        time = "<dim>{time:zz HH:mm:ss!UTC}</dim>"
    return time


def format_message() -> str:
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            "{message}",
            f"{RESET}</level>",
        ]
    )
    return message


def format_debug(config: LogConfig) -> str:
    if config.level.upper() != "DEBUG":
        return ""
    return "".join([f"<level>{NO_BOLD}", " [{file}::{line}]", f"{RESET}</level>"])


def setup_handlers(logger: Logger, format: str, config: LogConfig, rank: int) -> Logger:
    # Remove all default handlers
    logger.remove()

    # Install new handler on the main rank
    if rank == 0:
        logger.add(sys.stdout, format=format, level=config.level.upper(), colorize=True)

    # Disable critical logging
    logger.critical = lambda _: None

    return logger


def set_logger(logger: Logger) -> None:
    """
    Set the global logger. This function is shared across submodules such as
    training and inference, and should be called *exactly once* from a
    module-specific `setup_logger` function with the logger instance.
    """
    global _LOGGER
    _LOGGER = logger


def get_logger() -> Logger | None:
    """
    Get the global logger. This function is shared across submodules such as
    training and inference to accesst the global logger instance. Raises if the
    logger has not been set.

    Returns:
        The global logger.
    """
    global _LOGGER
    return _LOGGER


def reset_logger() -> None:
    """Reset the global logger. Useful mainly in test to clear loggers between tests."""
    global _LOGGER
    _LOGGER = None
