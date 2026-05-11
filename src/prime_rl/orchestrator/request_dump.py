from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from verifiers.types import SamplingArgs
from verifiers.utils.save_utils import make_serializable

logger = logging.getLogger(__name__)

_PATCHED = False


def _dump_dir() -> Path | None:
    raw_dir = os.environ.get("PRIME_RL_CHAT_REQUEST_DUMP_DIR")
    if not raw_dir:
        return None
    return Path(raw_dir)


def _normalize_sampling_args(sampling_args: SamplingArgs) -> dict[str, Any]:
    args = dict(sampling_args)
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    return {key: value for key, value in args.items() if value is not None}


def _request_payload(
    *,
    prompt: Any,
    model: str,
    sampling_args: SamplingArgs,
    tools: Any,
) -> dict[str, Any]:
    request = {
        "model": model,
        "messages": prompt,
        **_normalize_sampling_args(sampling_args),
    }
    if tools:
        request["tools"] = tools
    return request


def _state_summary(state: Any) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    keys = (
        "trajectory_id",
        "example_id",
        "env_name",
        "task",
        "problem",
        "question",
        "answer",
    )
    return {key: state[key] for key in keys if key in state}


def _exception_summary(exc: BaseException) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(exc).__name__,
        "repr": repr(exc),
        "message": str(exc),
    }
    response = getattr(exc, "response", None)
    if response is not None:
        summary["status_code"] = getattr(response, "status_code", None)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            summary["response_text"] = text[:8000]
        request = getattr(response, "request", None)
        if request is not None:
            summary["url"] = str(getattr(request, "url", ""))
            summary["method"] = getattr(request, "method", None)
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_dump(record: dict[str, Any]) -> None:
    root = _dump_dir()
    if root is None:
        return

    failed_dir = root / "failed_chat_requests"
    failed_dir.mkdir(parents=True, exist_ok=True)

    request_id = record["request_id"]
    path = failed_dir / f"{request_id}.json"
    with path.open("w") as f:
        json.dump(
            _json_safe(record),
            f,
            default=make_serializable,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")

    index_path = root / "failed_chat_requests.jsonl"
    with index_path.open("a") as f:
        json.dump(
            _json_safe(
                {
                    "request_id": request_id,
                    "path": path.as_posix(),
                    "ts": record["ts"],
                    "exception": record["exception"],
                    "state": record.get("state"),
                }
            ),
            f,
            default=make_serializable,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")

    logger.warning("Dumped failed chat completions request to %s", path)


def install_chat_request_dump_patch() -> None:
    """Dump exact OpenAI chat-completions requests when env rollouts fail.

    The env server is spawned with multiprocessing ``spawn``, so this patch must
    be installed from inside the child process before the env worker resolves the
    verifiers client.
    """
    global _PATCHED
    if _PATCHED:
        return

    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )

    original = OpenAIChatCompletionsClient.get_native_response

    async def get_native_response_with_dump(
        self: Any,
        prompt: Any,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return await original(self, prompt, model, sampling_args, tools, **kwargs)
        except Exception as exc:
            extra_headers = kwargs.get("extra_headers")
            state = kwargs.get("state")
            record = {
                "request_id": (
                    f"{int(time.time_ns())}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
                ),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pid": os.getpid(),
                "client_base_url": str(
                    getattr(getattr(self, "client", None), "base_url", "")
                ),
                "headers": extra_headers or {},
                "state": _state_summary(state),
                "request": _request_payload(
                    prompt=prompt,
                    model=model,
                    sampling_args=sampling_args,
                    tools=tools,
                ),
                "exception": _exception_summary(exc),
            }
            _write_dump(record)
            raise

    OpenAIChatCompletionsClient.get_native_response = get_native_response_with_dump
    _PATCHED = True
