from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


def enabled() -> bool:
    return os.getenv("PRIME_RL_LORA_NAN_DIAG", "").lower() in {"1", "true", "yes", "on"}


def probe_enabled() -> bool:
    return os.getenv("PRIME_RL_LORA_NAN_DIAG_PROBE", "").lower() in {"1", "true", "yes", "on"}


def generate_check_enabled() -> bool:
    return os.getenv("PRIME_RL_LORA_NAN_DIAG_CHECK_GENERATE", "").lower() in {"1", "true", "yes", "on"}


def dump_dir() -> Path:
    return Path(os.getenv("PRIME_RL_LORA_NAN_DIAG_DIR", "/tmp/prime-rl-lora-nan-diag"))


def dump_limit() -> int:
    try:
        return int(os.getenv("PRIME_RL_LORA_NAN_DIAG_DUMP_LIMIT", "5"))
    except ValueError:
        return 5


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(v) for v in value)
    return False


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    return value


def _next_dump_path(prefix: str, request_id: str | None = None) -> Path | None:
    root = dump_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create LoRA NaN diagnostics directory: %s", root)
        return None

    if len(list(root.glob("*.json"))) >= dump_limit():
        return None

    request_part = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (request_id or "unknown"))
    return root / f"{int(time.time() * 1000)}_{prefix}_{request_part}.json"


async def _request_snapshot(raw_request: Request | None) -> dict[str, Any]:
    if raw_request is None:
        return {}
    headers = dict(raw_request.headers)
    for key in list(headers):
        if key.lower() in {"authorization", "x-api-key"}:
            headers[key] = "<redacted>"
    snapshot: dict[str, Any] = {
        "url": str(raw_request.url),
        "method": raw_request.method,
        "headers": headers,
    }
    try:
        body = await raw_request.json()
    except Exception as exc:
        snapshot["body_error"] = repr(exc)
    else:
        snapshot["body"] = body
    return snapshot


async def dump_nonfinite_response(
    *,
    endpoint: str,
    request_id: str,
    model_name: str | None,
    raw_request: Request | None,
    response_payload: Any | None,
    exception: BaseException | None = None,
    lora_name: str | None = None,
    lora_int_id: int | None = None,
) -> None:
    if not enabled():
        return
    path = _next_dump_path("nonfinite_response", request_id)
    if path is None:
        return

    payload = {
        "kind": "nonfinite_response",
        "created_at": time.time(),
        "endpoint": endpoint,
        "request_id": request_id,
        "model_name": model_name,
        "lora_name": lora_name,
        "lora_int_id": lora_int_id,
        "exception": repr(exception) if exception is not None else None,
        "traceback": traceback.format_exc() if exception is not None else None,
        "response_contains_nonfinite": contains_nonfinite(response_payload),
        "response_payload": _safe_json_value(response_payload),
        "request": await _request_snapshot(raw_request),
    }
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.exception("Failed to write LoRA NaN diagnostics dump: %s", path)
    else:
        logger.error("Wrote LoRA NaN diagnostics dump: %s", path)


async def json_response_or_dump(
    *,
    endpoint: str,
    request_id: str,
    model_name: str | None,
    raw_request: Request | None,
    response_payload: Any,
    lora_name: str | None = None,
    lora_int_id: int | None = None,
) -> JSONResponse:
    try:
        return JSONResponse(content=response_payload)
    except ValueError as exc:
        if "Out of range float values" in str(exc) or contains_nonfinite(response_payload):
            await dump_nonfinite_response(
                endpoint=endpoint,
                request_id=request_id,
                model_name=model_name,
                raw_request=raw_request,
                response_payload=response_payload,
                exception=exc,
                lora_name=lora_name,
                lora_int_id=lora_int_id,
            )
        raise


def adapter_path_summary(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    summary: dict[str, Any] = {"path": root.as_posix(), "exists": root.exists()}
    if not root.exists():
        return summary
    files: dict[str, dict[str, Any]] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt"):
        file_path = root / name
        if file_path.exists():
            stat = file_path.stat()
            files[name] = {"size": stat.st_size, "mtime": stat.st_mtime}
    summary["files"] = files
    return summary


def log_adapter_file_finiteness(lora_path: str | Path, *, lora_name: str, lora_int_id: int | None) -> None:
    if not enabled():
        return
    root = Path(lora_path)
    stats: dict[str, Any] = {
        "lora_name": lora_name,
        "lora_int_id": lora_int_id,
        "path": root.as_posix(),
        "exists": root.exists(),
        "tensors": {},
    }
    if not root.exists():
        logger.error("LoRA diagnostic path does not exist: %s", stats)
        return

    st_path = root / "adapter_model.safetensors"
    if not st_path.exists():
        logger.info("LoRA adapter file diagnostic: %s", json.dumps(stats, sort_keys=True))
        return

    try:
        from safetensors.torch import safe_open

        with safe_open(st_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                tensor = tensors.get_tensor(key)
                finite = torch.isfinite(tensor)
                stats["tensors"][key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "all_finite": bool(finite.all().item()),
                    "nonfinite_count": int((~finite).sum().item()),
                }
    except Exception:
        logger.exception("Failed to inspect LoRA adapter at %s", root)
    else:
        logger.info("LoRA adapter tensor diagnostic: %s", json.dumps(stats, sort_keys=True))
