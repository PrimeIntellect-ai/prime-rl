import logging
from logging import Logger, Formatter

from zeroband.utils import envs


class PrimeFormatter(Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record):
        log_format = "{asctime} [{name}] [{levelname}] [{filename}:{lineno}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%m-%d %H:%M:%S")
        return formatter.format(record)


ALLOWED_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "CRITICAL": logging.CRITICAL}


def get_logger(name: str | None = None) -> Logger:
    # Get logger from Python's built-in registry
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        # Set log level
        level = envs.PRIME_LOG_LEVEL
        logger.setLevel(ALLOWED_LEVELS.get(level.upper(), logging.INFO))

        # Add handler with custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(PrimeFormatter())
        logger.addHandler(handler)

        # Prevent the log messages from being propagated to the root logger
        logger.propagate = False

    return logger


def reset_logger(name: str | None = None) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
