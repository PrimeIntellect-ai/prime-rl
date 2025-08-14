from loguru import logger as loguru_logger
from loguru._logger import Logger

from prime_rl.orchestrator.config import LogConfig
from prime_rl.utils.logger import format_debug, format_message, format_time, set_logger, setup_handlers


def setup_logger(log_config: LogConfig) -> Logger:
    message = format_message()
    time = format_time(log_config)
    debug = format_debug(log_config)
    format = time + message + debug

    # Setup the logger handlers
    logger = setup_handlers(loguru_logger, format, log_config, rank=0)
    set_logger(logger)

    return logger
