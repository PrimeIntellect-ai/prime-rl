import logging
import sys
from pathlib import Path

# Global logger instance
_LOGGER = None

NO_BOLD = "\033[22m"
RESET = "\033[0m"


class _VerifiersInterceptHandler(logging.Handler):
    """Intercept stdlib logging from verifiers and route to loguru with [verifiers] tag."""

    def emit(self, record: logging.LogRecord) -> None:
        logger = get_logger()
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, f"[verifiers] {record.getMessage()}")


def setup_logger(log_level: str, log_file: Path | None = None, append: bool = False, tag: str | None = None):
    global _LOGGER
    if _LOGGER is not None:
        raise RuntimeError("Logger already set. Please call `setup_logger` only once.")

    # Format message with optional tag prefix
    tag_prefix = f"[{tag}] " if tag else ""
    message = "".join(
        [
            " <level>{level: >7}</level>",
            f" <level>{NO_BOLD}",
            f"{tag_prefix}{{message}}",
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

    # Install console handler
    logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True)

    # If specified, install file handler
    if log_file is not None:
        if not append and log_file.exists():
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


def intercept_verifiers_logging(level: str = "DEBUG"):
    """Intercept all verifiers stdlib logging and route through prime-rl loguru with [verifiers] tag."""
    vf_logger = logging.getLogger("verifiers")
    vf_logger.handlers.clear()
    vf_logger.addHandler(_VerifiersInterceptHandler())
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
