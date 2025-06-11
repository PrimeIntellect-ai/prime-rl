import logging
from logging import Formatter, Logger
from typing import Literal

from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils import envs


class PrimeFormatter(Formatter):
    def __init__(self, world_info: WorldInfo | None = None):
        super().__init__()
        self.world_info = world_info

    def format(self, record):
        if self.world_info is not None:
            record.rank = self.world_info.rank
            log_format = "{asctime} [{name}] [Rank {rank}] [{levelname}] [{filename}:{lineno}] {message}"
        else:
            log_format = "{asctime} [{name}] [{levelname}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%m-%d %H:%M:%S")
        return formatter.format(record)


ALLOWED_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}
LoggerName = Literal["INFER", "TRAIN"]


def get_logger(name: LoggerName) -> Logger:
    # Get logger from Python's built-in registry
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        level = envs.PRIME_LOG_LEVEL
        world_info = None
        if name == "TRAIN":
            world_info = get_world_info()
            if world_info.local_rank == 0:
                # On first rank, set log level from env var
                logger.setLevel(ALLOWED_LEVELS.get(level, logging.INFO))
            else:
                # Else, only log critical messages
                logger.setLevel(logging.CRITICAL)
        else:
            # Set log level
            logger.setLevel(ALLOWED_LEVELS.get(level, logging.INFO))

        # Add handler with custom formatter
        handler = logging.StreamHandler()
        handler.setFormatter(PrimeFormatter(world_info=world_info))
        logger.addHandler(handler)

        # Prevent the log messages from being propagated to the root logger
        logger.propagate = False

    return logger


def reset_logger(name: str | None = None) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()
