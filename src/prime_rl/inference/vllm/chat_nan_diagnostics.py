from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Request
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_REQUEST_RING: deque[dict[str, Any]] = deque(maxlen=256)
_REQUEST_RING_SIZE = 256


def enabled() -> bool:
    return os.getenv("PRIME_RL_CHAT_NAN_DIAG", "").lower() in _TRUTHY


def dump_dir() -> Path:
    return Path(os.getenv("PRIME_RL_CHAT_NAN_DIAG_DIR", "/tmp/prime-rl-chat-nan-diag"))


def dump_limit() -> int:
    try:
        return int(os.getenv("PRIME_RL_CHAT_NAN_DIAG_DUMP_LIMIT", "5"))
    except ValueError:
        return 5


def ring_size() -> int:
    try:
        return max(0, int(os.getenv("PRIME_RL_CHAT_NAN_DIAG_RING_SIZE", "256")))
    except ValueError:
        return 256


def _request_ring() -> deque[dict[str, Any]]:
    global _REQUEST_RING, _REQUEST_RING_SIZE
    size = ring_size()
    if size != _REQUEST_RING_SIZE:
        _REQUEST_RING = deque(list(_REQUEST_RING)[-size:], maxlen=size)
        _REQUEST_RING_SIZE = size
    return _REQUEST_RING


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    return value


def nonfinite_paths(value: Any, prefix: str = "$", limit: int = 256) -> list[str]:
    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(paths) >= limit:
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                paths.append(path)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for idx, child in enumerate(item):
                visit(child, f"{path}[{idx}]")

    visit(value, prefix)
    return paths


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    return value


def model_dump(value: Any) -> Any:
    return _model_dump(value)


def _request_id(parsed_request: Any, raw_request: Request | None) -> str:
    request_id = getattr(parsed_request, "request_id", None)
    if request_id:
        return str(request_id)
    if raw_request is not None:
        request_metadata = getattr(raw_request.state, "request_metadata", None)
        request_id = getattr(request_metadata, "request_id", None)
        if request_id:
            return str(request_id)
    return "unknown"


def _safe_headers(raw_request: Request | None) -> dict[str, str]:
    if raw_request is None:
        return {}
    headers = dict(raw_request.headers)
    for key in list(headers):
        if key.lower() in {"authorization", "x-api-key"}:
            headers[key] = "<redacted>"
    return headers


async def _request_body(raw_request: Request | None) -> Any:
    if raw_request is None:
        return None
    try:
        return await raw_request.json()
    except Exception as exc:
        return {"body_error": repr(exc)}


async def snapshot_chat_request(*, endpoint: str, parsed_request: Any, raw_request: Request | None) -> dict[str, Any]:
    snapshot = {
        "created_at": time.time(),
        "endpoint": endpoint,
        "request_id": _request_id(parsed_request, raw_request),
        "model": getattr(parsed_request, "model", None),
        "url": str(raw_request.url) if raw_request is not None else None,
        "method": raw_request.method if raw_request is not None else None,
        "headers": _safe_headers(raw_request),
        "body": await _request_body(raw_request),
        "parsed_request": _safe_json_value(_model_dump(parsed_request)),
    }
    _request_ring().append(snapshot)
    return snapshot


def _next_dump_path(prefix: str, request_id: str) -> Path | None:
    root = dump_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create chat NaN diagnostics directory: %s", root)
        return None

    if len(list(root.glob("*.json"))) >= dump_limit():
        return None

    request_part = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in request_id)
    return root / f"{int(time.time() * 1000)}_{prefix}_{request_part}.json"


async def dump_chat_nonfinite_response(
    *,
    endpoint: str,
    request_snapshot: dict[str, Any],
    response_payload: Any,
    exception: BaseException | None = None,
) -> None:
    if not enabled():
        return

    paths = nonfinite_paths(response_payload)
    if not paths and exception is None:
        return

    request_id = str(request_snapshot.get("request_id") or "unknown")
    path = _next_dump_path("chat_nonfinite_response", request_id)
    if path is None:
        return

    payload = {
        "kind": "chat_nonfinite_response",
        "created_at": time.time(),
        "host": socket.gethostname(),
        "endpoint": endpoint,
        "request_id": request_id,
        "model": request_snapshot.get("model"),
        "exception": repr(exception) if exception is not None else None,
        "traceback": traceback.format_exc() if exception is not None else None,
        "response_nonfinite_paths": paths,
        "response_payload": _safe_json_value(response_payload),
        "request": request_snapshot,
        "recent_requests": list(_request_ring()),
    }
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.exception("Failed to write chat NaN diagnostics dump: %s", path)
    else:
        logger.error("Wrote chat NaN diagnostics dump: %s", path)
