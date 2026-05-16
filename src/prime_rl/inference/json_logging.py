"""Stdlib `logging`-based JSON formatter and dictConfig for vLLM.

Trainer and orchestrator emit JSON via loguru
(`prime_rl.utils.logger.json_sink`). vLLM uses Python's stdlib
`logging` and spawns workers via `multiprocessing.spawn`, so we can't
share the loguru sink directly. We emit the same flat shape from a
stdlib `Formatter` and drive it via a dictConfig file pointed to by
`VLLM_LOGGING_CONFIG_PATH` — env vars cross the spawn boundary, so
worker processes inherit the same logging config.
"""

import datetime
import json
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record matching `build_log_entry` in
    `prime_rl.utils.logger` so the platform's JsonLogsViewer renders
    inference rows identically to trainer / orchestrator rows."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            entry["exception"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(entry, default=str)


# Loguru-only level names that stdlib `logging` doesn't recognize.
# Mapped to the nearest stdlib level so `dictConfig` accepts them.
_LOGURU_TO_STDLIB = {"TRACE": "DEBUG", "SUCCESS": "INFO"}


def build_dict_config(level: str = "info") -> dict[str, Any]:
    """`logging.config.dictConfig` payload that routes vLLM, uvicorn,
    fastapi, and the root logger through `JsonFormatter` to stdout.

    Used inline for the parent process and serialized to disk for vLLM
    workers (via `VLLM_LOGGING_CONFIG_PATH`).
    """
    upper = _LOGURU_TO_STDLIB.get(level.upper(), level.upper())
    handler = {
        "class": "logging.StreamHandler",
        "formatter": "prime_rl_json",
        "level": upper,
        "stream": "ext://sys.stdout",
    }
    # Each named logger is configured explicitly: vLLM, uvicorn, and
    # transformers all set propagate=False on their own loggers, so a
    # root-only setup wouldn't reach them.
    named = ("vllm", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "transformers")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "prime_rl_json": {
                "()": "prime_rl.inference.json_logging.JsonFormatter",
            },
        },
        "handlers": {"stdout_json": handler},
        "loggers": {
            "": {"handlers": ["stdout_json"], "level": upper},
            **{name: {"handlers": ["stdout_json"], "level": upper, "propagate": False} for name in named},
        },
    }


def write_logging_config(level: str = "info") -> Path:
    """Persist a dictConfig JSON to a temp path. Caller sets the path on
    `VLLM_LOGGING_CONFIG_PATH` so vLLM (and its spawned workers) load
    the same config at import time."""
    config = build_dict_config(level)
    fd, path_str = tempfile.mkstemp(prefix="prime_rl_logging_", suffix=".json")
    path = Path(path_str)
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return path
