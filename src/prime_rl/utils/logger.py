import json as json_module
import logging
import sys
import traceback
from pathlib import Path

# Global logger instance
_LOGGER = None

NO_BOLD = "\033[22m"
RESET = "\033[0m"


def _build_log_entry(record) -> dict:
    """Build a flat JSON log entry from a loguru record."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    if record["exception"] is not None:
        exc = record["exception"]
        log_entry["exception"] = "".join(traceback.format_exception(exc.type, exc.value, exc.traceback))
    # Extract tag from extra if present (used by workers to identify themselves)
    extra = record["extra"]
    if extra:
        if "tag" in extra:
            log_entry["tag"] = extra["tag"]
            extra = {k: v for k, v in extra.items() if k != "tag"}
        if extra:
            log_entry["extra"] = extra
    return log_entry


def _json_sink(message) -> None:
    """Sink that outputs flat JSON to stdout for log aggregation (Loki, Grafana, etc.)."""
    log_entry = _build_log_entry(message.record)
    sys.stdout.write(json_module.dumps(log_entry) + "\n")
    sys.stdout.flush()


class _JsonFileSink:
    """File sink that keeps the handle open and writes flat JSON lines."""

    def __init__(self, log_file: Path):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_file, "a")

    def write(self, message) -> None:
        log_entry = _build_log_entry(message.record)
        self._file.write(json_module.dumps(log_entry) + "\n")
        self._file.flush()


class _VerifiersInterceptHandler(logging.Handler):
    """Intercept stdlib logging from verifiers and route to loguru with [verifiers] prefix."""

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


def setup_logger(
    log_level: str,
    log_file: Path | None = None,
    append: bool = False,
    tag: str | None = None,
    json: bool = False,
):
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

    # Bind tag to logger context for JSON mode (in non-JSON mode, tag is in the format string)
    if json and tag:
        logger = logger.bind(tag=tag)

    # Install console handler (enqueue=True for non-blocking in async contexts)
    if json:
        logger.add(_json_sink, level=log_level.upper(), enqueue=True)
    else:
        logger.add(sys.stdout, format=format, level=log_level.upper(), colorize=True, enqueue=True)

    # If specified, install file handler
    if log_file is not None:
        if not append and log_file.exists():
            log_file.unlink()
        if json:
            file_sink = _JsonFileSink(log_file)
            logger.add(file_sink.write, level=log_level.upper(), enqueue=True)
        else:
            logger.add(log_file, format=format, level=log_level.upper(), colorize=True, enqueue=True)

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
    """Intercept all verifiers stdlib logging and route through prime-rl loguru with [verifiers] prefix."""
    vf_logger = logging.getLogger("verifiers")
    vf_logger.handlers.clear()
    vf_logger.addHandler(_VerifiersInterceptHandler())
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
