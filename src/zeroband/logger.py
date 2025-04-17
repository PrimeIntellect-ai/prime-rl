import logging
import os

from zeroband.training.world_info import get_world_info

logger = None


class CustomFormatter(logging.Formatter):
    def __init__(self, local_rank: int):
        super().__init__()
        self.local_rank = local_rank

    def format(self, record):
        log_format = "{asctime} [{levelname}] [{name}] [Rank {local_rank}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%H:%M:%S")
        record.local_rank = self.local_rank
        return formatter.format(record)


def get_logger(name: str = "TRAIN") -> logging.Logger:
    global logger
    if logger is not None:
        return logger
    world_info = get_world_info()

    logger = logging.getLogger(name)

    if world_info.local_rank == 0:
        level = os.environ.get("ZERO_BAND_LOG_LEVEL", "INFO")

        if level == "DEBUG":
            logger.setLevel(level=logging.DEBUG)
        elif level == "INFO":
            logger.setLevel(level=logging.INFO)
        else:
            raise ValueError(f"Invalid log level: {level}")
    else:
        logger.setLevel(level=logging.CRITICAL)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter(world_info.local_rank))
    logger.addHandler(handler)
    logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    return logger
