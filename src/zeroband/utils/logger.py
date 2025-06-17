from loguru._logger import Logger

# Global loguru logger instance
_LOGGER: Logger | None = None


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
    training and inference to accesst the global logger instance.

    Returns:
        The global logger.
    """
    global _LOGGER
    return _LOGGER


def reset_logger() -> None:
    """Reset the global logger. Useful mainly in test to clear loggers between tests."""
    global _LOGGER
    _LOGGER = None
