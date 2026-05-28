from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import socket
import time
import traceback
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from fastapi import Request
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_REQUEST_RING: deque[dict[str, Any]] = deque(maxlen=256)
_REQUEST_RING_SIZE = 256
_REQUEST_BY_OBJECT: OrderedDict[int, dict[str, Any]] = OrderedDict()
_ACTIVE_GENERATIONS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_GENERATION_RING: deque[dict[str, Any]] = deque(maxlen=128)
_DUMP_COUNTS_BY_PREFIX: dict[str, int] = {}
_NONFINITE_EXCEPTION_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])nan(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"non[- ]?finite", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])[+-]?inf(?:inity)?(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"float values are not json compliant", re.IGNORECASE),
)


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in _TRUTHY


def enabled() -> bool:
    return _truthy("PRIME_RL_CHAT_REPLAY_DIAG") or _truthy("PRIME_RL_CHAT_NAN_DIAG")


def fail_fast() -> bool:
    return _truthy("PRIME_RL_CHAT_REPLAY_FAIL_FAST") or _truthy("PRIME_RL_CHAT_NAN_DIAG_FAIL_FAST")


def exit_after_dump() -> bool:
    return _truthy("PRIME_RL_CHAT_REPLAY_EXIT_AFTER_DUMP") or _truthy("PRIME_RL_CHAT_NAN_DIAG_EXIT_AFTER_DUMP")


def dump_dir() -> Path:
    return Path(
        os.getenv(
            "PRIME_RL_CHAT_REPLAY_DIAG_DIR",
            os.getenv("PRIME_RL_CHAT_NAN_DIAG_DIR", "/tmp/prime-rl-chat-replay-diag"),
        )
    )


def dump_limit() -> int:
    return _env_int("PRIME_RL_CHAT_REPLAY_DUMP_LIMIT", _env_int("PRIME_RL_CHAT_NAN_DIAG_DUMP_LIMIT", 5))


def ring_size() -> int:
    return max(0, _env_int("PRIME_RL_CHAT_REPLAY_RING_SIZE", _env_int("PRIME_RL_CHAT_NAN_DIAG_RING_SIZE", 256)))


def generation_ring_size() -> int:
    return max(0, _env_int("PRIME_RL_CHAT_REPLAY_GENERATION_RING_SIZE", 128))


def capture_logprobs() -> bool:
    return not _truthy("PRIME_RL_CHAT_REPLAY_DISABLE_LOGPROBS")


def capture_prompt_text() -> bool:
    return not _truthy("PRIME_RL_CHAT_REPLAY_DISABLE_PROMPT_TEXT")


def capture_token_ids() -> bool:
    return not _truthy("PRIME_RL_CHAT_REPLAY_DISABLE_TOKEN_IDS")


def text_limit() -> int:
    return max(0, _env_int("PRIME_RL_CHAT_REPLAY_TEXT_LIMIT", 262144))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _request_ring() -> deque[dict[str, Any]]:
    global _REQUEST_RING, _REQUEST_RING_SIZE
    size = ring_size()
    if size != _REQUEST_RING_SIZE:
        _REQUEST_RING = deque(list(_REQUEST_RING)[-size:], maxlen=size)
        _REQUEST_RING_SIZE = size
    return _REQUEST_RING


def _generation_ring() -> deque[dict[str, Any]]:
    global _GENERATION_RING
    size = generation_ring_size()
    if _GENERATION_RING.maxlen != size:
        _GENERATION_RING = deque(list(_GENERATION_RING)[-size:], maxlen=size)
    return _GENERATION_RING


def _prune_request_object_map() -> None:
    while len(_REQUEST_BY_OBJECT) > ring_size() * 2 + 16:
        _REQUEST_BY_OBJECT.popitem(last=False)


def _truncate_text(value: str) -> str:
    limit = text_limit()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 20:
        return repr(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return _truncate_text(value) if isinstance(value, str) else value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and value.__class__.__module__ == "enum":
        return value.name
    if dataclasses.is_dataclass(value):
        return _safe_json_value(dataclasses.asdict(value), depth=depth + 1)
    if hasattr(value, "model_dump"):
        try:
            return _safe_json_value(value.model_dump(mode="json"), depth=depth + 1)
        except TypeError:
            try:
                return _safe_json_value(value.model_dump(), depth=depth + 1)
            except Exception:
                return repr(value)
        except Exception:
            try:
                return _safe_json_value(value.model_dump(mode="python"), depth=depth + 1)
            except TypeError:
                try:
                    return _safe_json_value(value.model_dump(), depth=depth + 1)
                except Exception:
                    return repr(value)
            except Exception:
                return repr(value)
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(v, depth=depth + 1) for v in value]
    return repr(value)


def model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
        except Exception as exc:
            try:
                return value.model_dump(mode="python")
            except TypeError:
                try:
                    return value.model_dump()
                except Exception:
                    raise exc
            except Exception:
                raise exc
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def nonfinite_paths(value: Any, prefix: str = "$", limit: int = 512) -> list[str]:
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


def _exception_suggests_nonfinite(exception: BaseException | None) -> bool:
    if exception is None:
        return False
    text = f"{type(exception).__name__}: {exception}"
    return any(pattern.search(text) for pattern in _NONFINITE_EXCEPTION_PATTERNS)


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


async def _raw_request_body(raw_request: Request | None) -> dict[str, Any]:
    if raw_request is None:
        return {"raw_text": None, "json": None, "sha256": None, "num_bytes": 0}
    try:
        body_bytes = await raw_request.body()
    except Exception as exc:
        return {"error": repr(exc)}

    raw_text = body_bytes.decode("utf-8", errors="replace")
    body_json: Any
    try:
        body_json = json.loads(raw_text)
    except Exception as exc:
        body_json = {"json_error": repr(exc)}
    return {
        "raw_text": raw_text,
        "json": body_json,
        "sha256": hashlib.sha256(body_bytes).hexdigest(),
        "num_bytes": len(body_bytes),
    }


async def snapshot_chat_request(*, endpoint: str, parsed_request: Any, raw_request: Request | None) -> dict[str, Any]:
    diag_id = uuid.uuid4().hex
    body = await _raw_request_body(raw_request)
    snapshot = {
        "kind": "chat_request_snapshot",
        "diag_id": diag_id,
        "created_at": time.time(),
        "endpoint": endpoint,
        "request_id": _request_id(parsed_request, raw_request),
        "model": getattr(parsed_request, "model", None),
        "url": str(raw_request.url) if raw_request is not None else None,
        "method": raw_request.method if raw_request is not None else None,
        "headers": _safe_headers(raw_request),
        "body": body,
        "parsed_request": _safe_json_value(parsed_request),
    }
    _request_ring().append(snapshot)
    _REQUEST_BY_OBJECT[id(parsed_request)] = snapshot
    _REQUEST_BY_OBJECT.move_to_end(id(parsed_request))
    _prune_request_object_map()
    if raw_request is not None:
        raw_request.state.prime_rl_chat_replay_diag_id = diag_id
    return snapshot


def request_snapshot_for_object(parsed_request: Any) -> dict[str, Any] | None:
    snapshot = _REQUEST_BY_OBJECT.get(id(parsed_request))
    if snapshot is not None:
        _REQUEST_BY_OBJECT.move_to_end(id(parsed_request))
    return snapshot


def _sampling_params_payload(params: Any) -> dict[str, Any] | None:
    if params is None:
        return None
    payload: dict[str, Any] = {}
    for key, value in vars(params).items():
        if key.startswith("_"):
            continue
        if key in {"skip_clone", "output_kind", "output_text_buffer_length"}:
            continue
        payload[key] = _safe_json_value(value)
    return payload


def _lora_payload(lora_request: Any) -> dict[str, Any] | None:
    if lora_request is None:
        return None
    return {
        "lora_name": getattr(lora_request, "lora_name", None),
        "lora_int_id": getattr(lora_request, "lora_int_id", None),
        "lora_path": getattr(lora_request, "lora_path", None),
        "long_lora_max_len": getattr(lora_request, "long_lora_max_len", None),
    }


def record_engine_request(
    serving: Any,
    *,
    request_id: str,
    inputs: Any,
    params: Any,
    lora_request: Any,
) -> None:
    if not enabled():
        return
    try:
        components = serving._extract_prompt_components(inputs)
        token_ids = list(components.token_ids or [])
        prompt_text = components.text if capture_prompt_text() else None
        sampling_params = _sampling_params_payload(params)
        model_name = getattr(getattr(serving, "model_config", None), "model", None)
        entry = {
            "kind": "engine_request",
            "request_id": request_id,
            "created_at": time.time(),
            "model": model_name,
            "prompt_num_tokens": len(token_ids),
            "prompt_text": _truncate_text(prompt_text) if isinstance(prompt_text, str) else None,
            "prompt_token_ids": token_ids if capture_token_ids() else None,
            "sampling_params": sampling_params,
            "lora_request": _lora_payload(lora_request),
            "generate_replay_body": {
                "request_id": f"replay-{request_id}",
                "model": model_name,
                "token_ids": token_ids if capture_token_ids() else [],
                "sampling_params": sampling_params,
                "stream": False,
            },
            "updates": [],
            "final_output": None,
            "finished_at": None,
            "error": None,
        }
    except Exception as exc:
        logger.exception("Failed to record chat replay engine request: %s", request_id)
        entry = {
            "kind": "engine_request",
            "request_id": request_id,
            "created_at": time.time(),
            "error": repr(exc),
            "updates": [],
        }

    _ACTIVE_GENERATIONS[request_id] = entry
    _ACTIVE_GENERATIONS.move_to_end(request_id)
    while len(_ACTIVE_GENERATIONS) > generation_ring_size() * 2 + 16:
        _, old = _ACTIVE_GENERATIONS.popitem(last=False)
        _generation_ring().append(old)


def _logprob_position_payload(position: Any) -> Any:
    if position is None:
        return None
    try:
        items = position.items()
    except AttributeError:
        return _safe_json_value(position)

    return [
        {
            "token_id": int(token_id),
            "logprob": _safe_json_value(getattr(logprob, "logprob", None)),
            "rank": getattr(logprob, "rank", None),
            "decoded_token": getattr(logprob, "decoded_token", None),
        }
        for token_id, logprob in items
    ]


def _sample_logprobs_payload(logprobs: Any) -> Any:
    if logprobs is None or not capture_logprobs():
        return None
    try:
        return [_logprob_position_payload(position) for position in logprobs]
    except Exception:
        return _safe_json_value(logprobs)


def _completion_output_payload(output: Any) -> dict[str, Any]:
    token_ids = list(getattr(output, "token_ids", []) or [])
    return {
        "index": getattr(output, "index", None),
        "text": _truncate_text(getattr(output, "text", "") or ""),
        "token_ids": token_ids if capture_token_ids() else None,
        "num_token_ids": len(token_ids),
        "cumulative_logprob": _safe_json_value(getattr(output, "cumulative_logprob", None)),
        "logprobs": _sample_logprobs_payload(getattr(output, "logprobs", None)),
        "finish_reason": getattr(output, "finish_reason", None),
        "stop_reason": _safe_json_value(getattr(output, "stop_reason", None)),
        "lora_request": _lora_payload(getattr(output, "lora_request", None)),
    }


def request_output_payload(output: Any) -> dict[str, Any]:
    prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
    encoder_prompt_token_ids = list(getattr(output, "encoder_prompt_token_ids", []) or [])
    return {
        "request_id": getattr(output, "request_id", None),
        "prompt": _truncate_text(getattr(output, "prompt", "") or "") if capture_prompt_text() else None,
        "prompt_token_ids": prompt_token_ids if capture_token_ids() else None,
        "prompt_num_tokens": len(prompt_token_ids),
        "encoder_prompt": _truncate_text(getattr(output, "encoder_prompt", "") or "")
        if capture_prompt_text()
        else None,
        "encoder_prompt_token_ids": encoder_prompt_token_ids if capture_token_ids() else None,
        "encoder_prompt_num_tokens": len(encoder_prompt_token_ids),
        "prompt_logprobs": _sample_logprobs_payload(getattr(output, "prompt_logprobs", None)),
        "outputs": [_completion_output_payload(completion) for completion in getattr(output, "outputs", [])],
        "finished": getattr(output, "finished", None),
        "num_cached_tokens": getattr(output, "num_cached_tokens", None),
        "kv_transfer_params": _safe_json_value(getattr(output, "kv_transfer_params", None)),
    }


def record_request_output(request_id: str, output: Any) -> None:
    if not enabled():
        return
    entry = _ACTIVE_GENERATIONS.get(request_id)
    if entry is None:
        entry = {
            "kind": "engine_request",
            "request_id": request_id,
            "created_at": time.time(),
            "updates": [],
        }
        _ACTIVE_GENERATIONS[request_id] = entry
    payload = request_output_payload(output)
    payload["captured_at"] = time.time()
    entry["final_output"] = payload
    updates = entry.setdefault("updates", [])
    updates.append(payload)
    max_updates = max(1, _env_int("PRIME_RL_CHAT_REPLAY_MAX_UPDATES_PER_REQUEST", 32))
    if len(updates) > max_updates:
        del updates[: len(updates) - max_updates]


def complete_generation(request_id: str, *, response_payload: Any | None = None, error: BaseException | None = None) -> None:
    if not enabled():
        return
    entry = _ACTIVE_GENERATIONS.pop(request_id, None)
    if entry is None:
        entry = {
            "kind": "engine_request",
            "request_id": request_id,
            "created_at": time.time(),
            "updates": [],
        }
    entry["finished_at"] = time.time()
    if response_payload is not None:
        entry["response_nonfinite_paths"] = nonfinite_paths(response_payload)
    if error is not None:
        entry["error"] = repr(error)
    _generation_ring().append(entry)


def generation_snapshot(request_id: str) -> dict[str, Any] | None:
    if request_id in _ACTIVE_GENERATIONS:
        return _ACTIVE_GENERATIONS[request_id]
    for entry in reversed(_GENERATION_RING):
        if entry.get("request_id") == request_id:
            return entry
    return None


def active_generation_snapshots() -> list[dict[str, Any]]:
    return list(_ACTIVE_GENERATIONS.values())


def recent_generation_snapshots() -> list[dict[str, Any]]:
    return list(_generation_ring())


def _next_dump_path(prefix: str, request_id: str) -> Path | None:
    root = dump_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create chat replay diagnostics directory: %s", root)
        return None

    if _DUMP_COUNTS_BY_PREFIX.get(prefix, 0) >= dump_limit():
        return None

    request_part = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in request_id)
    _DUMP_COUNTS_BY_PREFIX[prefix] = _DUMP_COUNTS_BY_PREFIX.get(prefix, 0) + 1
    return root / f"{int(time.time() * 1000)}_{prefix}_{request_part}.json"


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_procid": os.getenv("SLURM_PROCID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "prime_rl_git_sha": os.getenv("PRIME_RL_GIT_SHA"),
    }
    try:
        import transformers
        import vllm

        metadata["vllm_version"] = getattr(vllm, "__version__", None)
        metadata["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception as exc:
        metadata["version_error"] = repr(exc)
    return metadata


async def dump_chat_nonfinite_response(
    *,
    endpoint: str,
    request_snapshot: dict[str, Any],
    response_payload: Any,
    exception: BaseException | None = None,
) -> Path | None:
    if not enabled():
        return None

    paths = nonfinite_paths(response_payload)
    exception_is_nonfinite = _exception_suggests_nonfinite(exception)
    if not paths and not exception_is_nonfinite:
        return None

    payload_id = (
        response_payload.get("id")
        if isinstance(response_payload, dict)
        else getattr(response_payload, "id", None)
    )
    request_id = str(payload_id or request_snapshot.get("request_id") or "unknown")
    prefix = "chat_nonfinite_response" if paths else "chat_nonfinite_exception"
    path = _next_dump_path(prefix, request_id)
    if path is None:
        return None

    generation = generation_snapshot(request_id)
    payload = {
        "kind": prefix,
        "created_at": time.time(),
        "runtime": _runtime_metadata(),
        "endpoint": endpoint,
        "request_id": request_id,
        "model": request_snapshot.get("model"),
        "exception": repr(exception) if exception is not None else None,
        "traceback": traceback.format_exc() if exception is not None else None,
        "response_nonfinite_paths": paths,
        "response_payload": _safe_json_value(response_payload),
        "request": request_snapshot,
        "generation": generation,
        "active_generations_at_dump": active_generation_snapshots(),
        "recent_generations": recent_generation_snapshots(),
        "recent_requests": list(_request_ring()),
        "replay_notes": {
            "chat_request": "POST request.body.json to /v1/chat/completions.",
            "engine_request": "POST generation.generate_replay_body to /inference/v1/generate to bypass chat rendering.",
            "cohort": "Replay active_generations_at_dump or recent_requests concurrently to approximate decode batch state.",
        },
    }
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.exception("Failed to write chat replay diagnostics dump: %s", path)
        return None
    logger.error("Wrote chat replay diagnostics dump: %s", path)
    return path
