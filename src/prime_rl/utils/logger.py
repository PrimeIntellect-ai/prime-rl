import os
import sys
from pathlib import Path

# Global logger instance
_LOGGER = None

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _should_use_json(json: bool | None) -> bool:
    if json is not None:
        return json

    # Prefer explicit format selector.
    fmt = os.getenv("PRIME_RL_LOG_FORMAT", "").strip().lower()
    if fmt == "json":
        return True
    if fmt in {"pretty", "text", "human"}:
        return False

    # Boolean toggle fallback.
    if _is_truthy_env(os.getenv("PRIME_RL_LOG_JSON")):
        return True

    # Loguru-compatible env var (nice when deploying via generic tooling).
    if _is_truthy_env(os.getenv("LOGURU_SERIALIZE")):
        return True

    return False


def setup_logger(log_level: str, log_file: Path | None = None, json: bool | None = None):
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

    # NOTE: We are creating a new "module-level" logger instance for prime-rl so that third-party code cannot "hijack" our logger
    # This is a bit hacky because loguru does not publicly expose the logger class, but oh well, it works
    from loguru._logger import Core as _Core
    from loguru._logger import Logger as _Logger

    logger = _Logger(
        core=_Core(),
        exception=None,
        depth=0,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )

    json_logging = _should_use_json(json)

    # Install console handler
    if json_logging:
        logger.add(sys.stdout, level=log_level.upper(), serialize=True)
    else:
        logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True)

    # If specified, install file handler
    if log_file is not None:
        if log_file.exists():
            log_file.unlink()
        if json_logging:
            logger.add(log_file, level=log_level.upper(), serialize=True)
        else:
            logger.add(log_file, format=format, level=log_level.upper(), colorize=True)

    # Disable critical logging
    logger.critical = lambda *args, **kwargs: None

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
    if _LOGGER is not None:
        # Remove all sinks/handlers to avoid duplicate logging across test runs.
        _LOGGER.remove()
    _LOGGER = None
