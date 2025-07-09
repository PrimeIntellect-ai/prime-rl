from pathlib import Path

from loguru import logger as loguru_logger
from loguru._logger import Logger

from zeroband.trainer.config import LogConfig
from zeroband.trainer.world import World
from zeroband.utils.logger import format_debug, format_message, format_time, get_logger, set_logger, setup_handlers


def setup_logger(log_config: LogConfig, world: World) -> Logger:
    if get_logger():
        raise RuntimeError("Logger already setup. Call reset_logger first.")

    message = format_message()
    time = format_time(log_config)
    debug = format_debug(log_config)
    format = time + message + debug

    # Setup the logger handlers
    if log_config.path:
        if world.world_size > 1:
            log_config.path = Path(log_config.path.as_posix() + str(world.rank))
        log_config.path = Path(log_config.path.as_posix() + ".log")
    logger = setup_handlers(loguru_logger, format, log_config, rank=world.rank)
    set_logger(logger)

    return logger
