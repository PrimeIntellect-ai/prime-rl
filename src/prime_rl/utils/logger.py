import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Global logger instance
_LOGGER = None
_LOG_CABIN = None

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def setup_logger(log_level: str, log_file: Path | None = None):
    global _LOGGER
    if _LOGGER is not None:
        raise RuntimeError("Logger already set. Please call `setup_logger` only once.")

    # Format message
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            "{message}",
            f"{RESET}</level>",
        ]
    )
    time = "<dim>{time:HH:mm:ss}</dim>"
    if log_level.upper() != "DEBUG":
        debug = ""
    else:
        debug = "".join([f"<level>{NO_BOLD}", " [{file}::{line}]", f"{RESET}</level>"])
    format = time + message + debug

    # Remove all default handlers
    logger.remove()

    # Install console handler
    logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True)

    # If specified, install file handler
    if log_file is not None:
        if log_file.exists():
            log_file.unlink()
        logger.add(log_file, format=format, level=log_level.upper(), colorize=True)

    # Disable critical logging
    logger.critical = lambda _: None

    # Set the global logger instance
    _LOGGER = logger

    return logger


def get_logger():
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


def reset_logger():
    """Reset the global logger. Useful mainly in test to clear loggers between tests."""
    global _LOGGER
    _LOGGER = None


def setup_logcabin(config, world = None):
    from log_cabin_v2 import Dispatcher as LogCabinDispatcher
    import os
    import json
    if world is None or world.is_master:
        logcabin = LogCabinDispatcher(
            local_path=os.path.expanduser("~/.logs"),
            server_url="http://log-cabin-v2/ingestion",
            username=os.getenv("LOG_CABIN_USER", os.getenv("USER", None)),
        )
        config_dict = json.loads(config.model_dump_json())
        logcabin.init(run_name=config.run_name, info=config_dict)
    else:
        logcabin = None

    class LogCabinWrapper:
        def __init__(self, log_cabin_client):
            self.log_cabin_client = log_cabin_client

        def log(self, entry_name, data):
            """clean up slashes in data keys"""
            if data is None:
                return
            data = {k: v for k, v in data.items() if v is not None}
            data = {k.replace("/", "_"): np.array(v) if isinstance(v, list) else v for k, v in data.items()}
            self.log_cabin_client.log(entry_name, data)
        
    global _LOG_CABIN
    _LOG_CABIN = LogCabinWrapper(logcabin) if logcabin is not None else None
    return _LOG_CABIN


def get_logcabin():
    global _LOG_CABIN
    if _LOG_CABIN is None:
        raise RuntimeError("Log cabin not set. Please call `setup_logcabin` first.")
    return _LOG_CABIN
