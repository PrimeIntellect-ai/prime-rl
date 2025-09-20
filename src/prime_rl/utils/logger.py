import sys
from pathlib import Path

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


def setup_handlers(logger: Logger, format: str, config: LogConfig, rank: int, output_dir: Path) -> Logger:
    # Remove all default handlers
    logger.remove()

    # Install console on the master rank
    if rank == 0:
        logger.add(sys.stdout, format=format, level=config.level.upper(), colorize=True)

    # If specified, install file handlers on all ranks
    if config.file:
        log_file = output_dir / "logs" / f"rank_{rank}.log"
        logger.add(log_file, format=format, level=config.level.upper())

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


def get_logger() -> Logger:
    """
    Get the global logger. This function is shared across submodules such as
    training and inference to accesst the global logger instance. Raises if the
    logger has not been set.

    Returns:
        The global logger.
    """
    global _LOGGER
    if _LOGGER is None:
        raise RuntimeError("Logger not set. Please call `set_logger` first.")
    return _LOGGER


def reset_logger() -> None:
    """Reset the global logger. Useful mainly in test to clear loggers between tests."""
    global _LOGGER
    _LOGGER = None
