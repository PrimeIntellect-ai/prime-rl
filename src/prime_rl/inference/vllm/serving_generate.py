"""/v1/generate endpoint — accepts pre-tokenized inputs.

Text-only tokens in, tokens out. The Renderer does all tokenization client-side.
No Jinja rendering, no server-side chat template application.

VLMs do not use this endpoint. The orchestrator routes VLMs to MITO
(/v1/chat/completions) where vLLM handles image preprocessing and chat
templating server-side.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import Request
from pydantic import BaseModel
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

try:
    from vllm.inputs.engine import tokens_input
except ImportError:
    # vLLM 0.18.x used this name/path; keep the dirty repro branch able to
    # compare the current generate path against the previous vLLM stack.
    from vllm.inputs.data import token_inputs as tokens_input
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


_CAUSAL_LOCK = threading.Lock()
_CAUSAL_ACTIVE: dict[str, dict[str, Any]] = {}
_CAUSAL_COMPLETED: list[dict[str, Any]] = []


# ── Request / Response schemas ───────────────────────────────────────


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt_token_ids: list[int]

    # When unset, fill from max_model_len - prompt_len at request time so we
    # match /v1/chat/completions behavior. The previous 4096 hard default
    # silently truncated long completions on 8k+ context runs (e.g. hendrycks
    # reasoning rollouts capped at 4096 tokens, making rendered rollouts look
    # shorter than main's for the same model).
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    n: int = 1
    stop_token_ids: list[int] | None = None
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    prompt_logprobs: bool = False
    priority: int = 0
    # Prefix-cache invalidation salt. Must match main's
    # /v1/chat/completions/tokens path: the orchestrator sets
    # `extra_body["cache_salt"] = str(ckpt_step)` on every rollout
    # request. vLLM's KV cache hashes include this salt, so when the
    # step changes the cache misses and KV is recomputed with fresh
    # weights. Without this, renderers path silently reuses stale KV
    # from before the latest weight update and its logprobs drift from
    # the trainer's forward pass (mismatch_kl grows 3x over training).
    cache_salt: str | None = None


class GenerateChoiceResponse(BaseModel):
    index: int
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None
    routed_experts: dict | None = None


class GenerateResponse(BaseModel):
    id: str
    model: str
    prompt_token_ids: list[int]
    choices: list[GenerateChoiceResponse]
    usage: dict
    prompt_logprobs: list[float | None] | None = None


# ── Handler ──────────────────────────────────────────────────────────


class OpenAIServingGenerate:
    """Lightweight generate handler — tokens in, tokens out."""

    def __init__(self, engine_client: EngineClient, chat_handler: Any | None = None):
        self.engine_client = engine_client
        self.chat_handler = chat_handler

    async def generate(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse | ErrorResponse | dict:
        # Pre-rendered TokensInput shape (type="token") — avoids vLLM's
        # "raw prompt" deprecation that targets plain lists/strings.
        engine_prompt = tokens_input(request.prompt_token_ids, cache_salt=request.cache_salt)

        # Match /v1/chat/completions: if the client didn't ask for a specific
        # cap, let the model generate up to whatever room is left in context.
        # vLLM v1 AsyncLLM exposes model_config directly (no async getter).
        max_tokens = request.max_tokens
        if max_tokens is None:
            max_model_len = self.engine_client.model_config.max_model_len
            max_tokens = max(1, max_model_len - len(request.prompt_token_ids))

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            seed=request.seed,
            n=request.n,
            stop_token_ids=request.stop_token_ids or [],
            repetition_penalty=request.repetition_penalty,
            min_tokens=request.min_tokens,
            logprobs=1,
            prompt_logprobs=1 if request.prompt_logprobs else None,
            skip_special_tokens=False,
        )

        request_id = f"gen-{uuid4().hex[:16]}"
        routed_experts_map: dict[int, dict] = {}
        final_output: RequestOutput | None = None
        data_parallel_rank = None
        lora_request = None
        trace_headers = None
        if self.chat_handler is not None:
            data_parallel_rank = self.chat_handler._get_data_parallel_rank(raw_request)
            trace_headers = await self.chat_handler._get_trace_headers(raw_request.headers)
            lora_request = self.chat_handler._maybe_get_adapters(request)
        replay_request_body = _generate_replay_request_body(request, max_tokens=max_tokens)
        causal_record = _record_causal_request_start(
            request_id=request_id,
            request=request,
            max_tokens=max_tokens,
            raw_request=raw_request,
            data_parallel_rank=data_parallel_rank,
        )
        initial_replay_dump_path = await _dump_generate_replay(
            request_id=request_id,
            request_body=replay_request_body,
            raw_request=raw_request,
            data_parallel_rank=data_parallel_rank,
            status="running",
            response=None,
            error=None,
            nonfinite=None,
        )
        if initial_replay_dump_path is not None:
            causal_record["replay_dump_path"] = initial_replay_dump_path
            _record_causal_request_replay_dump_path(request_id, initial_replay_dump_path)

        generator = self.engine_client.generate(
            engine_prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=data_parallel_rank,
        )

        # Drain the generator without polling ``raw_request.is_disconnected``
        # per decode step. That poll is one ASGI ``receive()`` await per
        # yielded token — at ~2k concurrent rollouts each producing dozens of
        # tokens, it was the single largest per-step overhead on the /generate
        # path (vLLM's own chat completions handler uses the CancelledError
        # pattern for the same reason). Starlette cancels this coroutine on
        # client disconnect, so the except branch still catches it.
        try:
            async for output in generator:
                for comp_output in output.outputs:
                    if comp_output.routed_experts is not None:
                        routed_experts_map[comp_output.index] = _encode_routed_experts(comp_output.routed_experts)
                final_output = output
        except asyncio.CancelledError:
            _record_causal_request_end(
                request_id,
                status="cancelled",
                error={"type": "CancelledError", "message": "client disconnected"},
            )
            await self.engine_client.abort(request_id)
            raise
        except Exception as exc:
            _record_causal_request_end(
                request_id,
                status="engine_error",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            await _dump_generate_replay(
                request_id=request_id,
                request_body=replay_request_body,
                raw_request=raw_request,
                data_parallel_rank=data_parallel_rank,
                status="engine_error",
                response=None,
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                nonfinite=None,
            )
            raise

        if final_output is None:
            _record_causal_request_end(
                request_id,
                status="empty_output",
                error={"type": "EmptyOutput", "message": "No output generated"},
            )
            await _dump_generate_replay(
                request_id=request_id,
                request_body=replay_request_body,
                raw_request=raw_request,
                data_parallel_rank=data_parallel_rank,
                status="empty_output",
                response=None,
                error={"type": "EmptyOutput", "message": "No output generated"},
                nonfinite=None,
            )
            return {"error": "No output generated"}

        choices = []
        for output in final_output.outputs:
            token_ids = list(output.token_ids)
            logprobs_list: list[float] = []
            if output.logprobs:
                for i, lp_dict in enumerate(output.logprobs):
                    if i < len(token_ids) and token_ids[i] in lp_dict:
                        logprobs_list.append(lp_dict[token_ids[i]].logprob)
                    else:
                        logprobs_list.append(0.0)

            choices.append(
                GenerateChoiceResponse(
                    index=output.index,
                    token_ids=token_ids,
                    logprobs=logprobs_list,
                    finish_reason=output.finish_reason,
                    routed_experts=routed_experts_map.get(output.index),
                )
            )

        prompt_len = len(final_output.prompt_token_ids)
        completion_len = sum(len(c.token_ids) for c in choices)
        prompt_logprobs = _extract_prompt_logprobs(final_output.prompt_logprobs)

        response = GenerateResponse(
            id=request_id,
            model=request.model or "",
            prompt_token_ids=list(final_output.prompt_token_ids),
            choices=choices,
            usage={
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
            prompt_logprobs=prompt_logprobs,
        )
        nonfinite = _find_non_finite_generate_value(
            response,
            request=request,
            data_parallel_rank=data_parallel_rank,
            tokenizer_getter=lambda: _get_diagnostic_tokenizer(
                self.chat_handler,
                self.engine_client,
            ),
        )
        replay_dump_path = await _dump_generate_replay(
            request_id=request_id,
            request_body=replay_request_body,
            raw_request=raw_request,
            data_parallel_rank=data_parallel_rank,
            status="nonfinite" if nonfinite is not None else "ok",
            response=response,
            error=None,
            nonfinite=nonfinite,
        )
        _record_causal_request_end(
            request_id,
            status="nonfinite" if nonfinite is not None else "ok",
            response=response,
            nonfinite=nonfinite,
            replay_dump_path=replay_dump_path,
        )
        if nonfinite is not None:
            if replay_dump_path is not None:
                nonfinite["replay_dump_path"] = replay_dump_path
            await _dump_generate_causal_incident(
                dump_dir=os.environ.get("PRIME_RL_GENERATE_REPLAY_DIR"),
                request_id=request_id,
                causal_record=causal_record,
                nonfinite=nonfinite,
                replay_dump_path=replay_dump_path,
            )
            return _non_finite_generate_error(nonfinite)
        return response


def _encode_routed_experts(arr: np.ndarray) -> dict:
    return {
        "data": base64.b85encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


def _extract_prompt_logprobs(
    prompt_logprobs: list[dict[int, Any] | None] | Mapping[int, Any] | None,
) -> list[float | None] | None:
    if prompt_logprobs is None:
        return None
    if isinstance(prompt_logprobs, Mapping):
        prompt_logprobs = [prompt_logprobs]

    extracted: list[float | None] = []
    for token_logprobs in prompt_logprobs:
        if not token_logprobs:
            extracted.append(None)
            continue
        selected = next(iter(token_logprobs.values()))
        logprob = selected.logprob if hasattr(selected, "logprob") else selected.get("logprob")
        extracted.append(float(logprob) if logprob is not None else None)
    return extracted


def _record_causal_request_start(
    *,
    request_id: str,
    request: GenerateRequest,
    max_tokens: int,
    raw_request: Request,
    data_parallel_rank: int | None,
) -> dict[str, Any]:
    if not _causal_diagnostics_enabled():
        return {}

    created_unix = time.time()
    headers = _safe_replay_headers(raw_request)
    record = {
        "schema": "prime_rl.generate_causal_request.v1",
        "request_id": request_id,
        "created_unix": created_unix,
        "started_unix": created_unix,
        "status": "running",
        "model": request.model,
        "prompt_len": len(request.prompt_token_ids),
        "prompt_hash": _hash_prompt_token_ids(request.prompt_token_ids),
        "max_tokens": max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "seed": request.seed,
        "n": request.n,
        "stop_token_ids": request.stop_token_ids or [],
        "repetition_penalty": request.repetition_penalty,
        "min_tokens": request.min_tokens,
        "prompt_logprobs": request.prompt_logprobs,
        "priority": request.priority,
        "cache_salt": request.cache_salt,
        "data_parallel_rank": data_parallel_rank,
        "client": _safe_client(raw_request),
        "headers": headers,
        "session_id": headers.get("x-session-id"),
    }

    with _CAUSAL_LOCK:
        record["active_request_ids_at_start"] = sorted(_CAUSAL_ACTIVE)
        record["preceding_completed_request_ids"] = [
            item["request_id"] for item in _CAUSAL_COMPLETED[-_causal_window_size() :]
        ]
        _CAUSAL_ACTIVE[request_id] = record
    return record


def _record_causal_request_end(
    request_id: str,
    *,
    status: str,
    response: GenerateResponse | None = None,
    nonfinite: dict[str, Any] | None = None,
    replay_dump_path: str | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    if not _causal_diagnostics_enabled():
        return

    finished_unix = time.time()
    with _CAUSAL_LOCK:
        record = _CAUSAL_ACTIVE.pop(request_id, {"request_id": request_id})
        active_at_end = sorted(_CAUSAL_ACTIVE)

    record.update(
        {
            "status": status,
            "finished_unix": finished_unix,
            "duration_seconds": finished_unix - record.get("started_unix", finished_unix),
            "active_request_ids_at_end": active_at_end,
            "replay_dump_path": replay_dump_path,
            "error": error,
            "nonfinite": _causal_nonfinite_summary(nonfinite),
        }
    )
    if response is not None:
        record.update(_causal_response_summary(response))

    with _CAUSAL_LOCK:
        _CAUSAL_COMPLETED.append(record)
        max_items = _causal_window_size()
        if len(_CAUSAL_COMPLETED) > max_items:
            del _CAUSAL_COMPLETED[: len(_CAUSAL_COMPLETED) - max_items]


def _record_causal_request_replay_dump_path(request_id: str, replay_dump_path: str) -> None:
    if not _causal_diagnostics_enabled():
        return

    with _CAUSAL_LOCK:
        record = _CAUSAL_ACTIVE.get(request_id)
        if record is not None:
            record["replay_dump_path"] = replay_dump_path


async def _dump_generate_causal_incident(
    *,
    dump_dir: str | None,
    request_id: str,
    causal_record: dict[str, Any],
    nonfinite: dict[str, Any],
    replay_dump_path: str | None,
) -> str | None:
    if not dump_dir or not _causal_dump_on_nonfinite_enabled():
        return None

    try:
        incident = _build_causal_incident(
            request_id=request_id,
            causal_record=causal_record,
            nonfinite=nonfinite,
            replay_dump_path=replay_dump_path,
        )
        return await asyncio.to_thread(_write_generate_causal_incident, Path(dump_dir), request_id, incident)
    except Exception:
        logger.exception("Failed to dump /v1/generate causal incident")
        return None


def _build_causal_incident(
    *,
    request_id: str,
    causal_record: dict[str, Any],
    nonfinite: dict[str, Any],
    replay_dump_path: str | None,
) -> dict[str, Any]:
    with _CAUSAL_LOCK:
        active = [dict(item) for item in _CAUSAL_ACTIVE.values()]
        completed = [dict(item) for item in _CAUSAL_COMPLETED[-_causal_window_size() :]]

    referenced_ids = set(causal_record.get("active_request_ids_at_start") or [])
    referenced_ids.update(causal_record.get("active_request_ids_at_end") or [])
    referenced_ids.update(causal_record.get("preceding_completed_request_ids") or [])
    referenced_ids.add(request_id)

    referenced = [item for item in completed + active if item.get("request_id") in referenced_ids]
    replay_paths = {
        str(item["request_id"]): item.get("replay_dump_path")
        for item in referenced
        if item.get("request_id") and item.get("replay_dump_path")
    }
    if replay_dump_path is not None:
        replay_paths[request_id] = replay_dump_path

    return {
        "schema": "prime_rl.generate_causal_incident.v1",
        "created_unix": time.time(),
        "request_id": request_id,
        "failing_request": causal_record,
        "nonfinite": nonfinite,
        "replay_dump_path": replay_dump_path,
        "active_request_ids_at_start": causal_record.get("active_request_ids_at_start", []),
        "active_request_ids_at_end": causal_record.get("active_request_ids_at_end", []),
        "preceding_completed_request_ids": causal_record.get("preceding_completed_request_ids", []),
        "referenced_request_ids": sorted(referenced_ids),
        "referenced_requests": referenced,
        "replay_dump_paths": replay_paths,
        "active_requests_snapshot": active,
        "completed_requests_snapshot": completed,
    }


def _write_generate_causal_incident(dump_dir: Path, request_id: str, incident: dict[str, Any]) -> str:
    incident_dir = dump_dir.expanduser() / "incidents" / request_id
    incident_dir.mkdir(parents=True, exist_ok=True)
    path = incident_dir / "incident.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(incident), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return str(path)


def _causal_response_summary(response: GenerateResponse) -> dict[str, Any]:
    choices = []
    for choice in response.choices:
        zero_offsets = [idx for idx, token_id in enumerate(choice.token_ids) if token_id == 0]
        choices.append(
            {
                "choice_index": choice.index,
                "completion_len": len(choice.token_ids),
                "finish_reason": choice.finish_reason,
                "token_id_prefix": choice.token_ids[:64],
                "token_0_count": len(zero_offsets),
                "token_0_offsets": zero_offsets[:64],
                "nonfinite_logprob_offsets": [
                    idx for idx, logprob in enumerate(choice.logprobs) if not math.isfinite(logprob)
                ][:64],
            }
        )

    return {
        "usage": dict(response.usage),
        "completion_len": sum(choice["completion_len"] for choice in choices),
        "choices": choices,
    }


def _causal_nonfinite_summary(nonfinite: dict[str, Any] | None) -> dict[str, Any] | None:
    if nonfinite is None:
        return None
    keys = (
        "field",
        "choice_index",
        "token_offset",
        "token_id",
        "completion_len",
        "finish_reason",
        "prompt_len",
        "value",
    )
    return {key: nonfinite.get(key) for key in keys if key in nonfinite}


def _causal_diagnostics_enabled() -> bool:
    return bool(os.environ.get("PRIME_RL_GENERATE_REPLAY_DIR"))


def _causal_dump_on_nonfinite_enabled() -> bool:
    raw = os.environ.get("PRIME_RL_GENERATE_CAUSAL_DUMP_ON_NONFINITE")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _causal_window_size() -> int:
    raw = os.environ.get("PRIME_RL_GENERATE_CAUSAL_WINDOW_SIZE")
    if raw is None:
        return 256
    try:
        return max(1, int(raw))
    except ValueError:
        return 256


def _hash_prompt_token_ids(prompt_token_ids: list[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in prompt_token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _generate_replay_request_body(request: GenerateRequest, *, max_tokens: int) -> dict[str, Any]:
    body = request.model_dump(mode="json", exclude_none=True)
    body["max_tokens"] = max_tokens
    body["stop_token_ids"] = request.stop_token_ids or []
    return body


async def _dump_generate_replay(
    *,
    request_id: str,
    request_body: dict[str, Any],
    raw_request: Request,
    data_parallel_rank: int | None,
    status: str,
    response: GenerateResponse | None,
    error: dict[str, Any] | None,
    nonfinite: dict[str, Any] | None,
) -> str | None:
    dump_dir = os.environ.get("PRIME_RL_GENERATE_REPLAY_DIR")
    if not dump_dir:
        return None

    record = {
        "schema": "prime_rl.generate_replay.v1",
        "created_unix": time.time(),
        "request_id": request_id,
        "endpoint": "/v1/generate",
        "status": status,
        "client": _safe_client(raw_request),
        "headers": _safe_replay_headers(raw_request),
        "data_parallel_rank": data_parallel_rank,
        "request": request_body,
        "response": response.model_dump(mode="python") if response is not None else None,
        "error": error,
        "nonfinite": nonfinite,
    }

    try:
        return await asyncio.to_thread(_write_generate_replay, Path(dump_dir), request_id, record)
    except Exception:
        logger.exception("Failed to dump /v1/generate replay record")
        return None


def _write_generate_replay(dump_dir: Path, request_id: str, record: dict[str, Any]) -> str:
    request_dir = dump_dir.expanduser() / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    path = request_dir / f"{request_id}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return str(path)


def _safe_client(raw_request: Request) -> dict[str, Any] | None:
    if raw_request.client is None:
        return None
    return {
        "host": raw_request.client.host,
        "port": raw_request.client.port,
    }


def _safe_replay_headers(raw_request: Request) -> dict[str, str]:
    allowlist = {
        "traceparent",
        "x-request-id",
        "x-session-id",
        "x-trace-id",
        "x-b3-traceid",
    }
    return {key: value for key, value in raw_request.headers.items() if key.lower() in allowlist}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if math.isfinite(float_value):
            return float_value
        return {"__nonfinite_float__": repr(float_value)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _find_non_finite_generate_value(
    response: GenerateResponse,
    *,
    request: GenerateRequest,
    data_parallel_rank: int | None,
    tokenizer_getter: Callable[[], Any | None],
) -> dict[str, Any] | None:
    prompt_len = len(response.prompt_token_ids)
    common = {
        "request_id": response.id,
        "model": response.model,
        "prompt_len": prompt_len,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "n": request.n,
        "cache_salt": request.cache_salt,
        "data_parallel_rank": data_parallel_rank,
    }

    for choice in response.choices:
        completion_len = len(choice.token_ids)
        for token_offset, logprob in enumerate(choice.logprobs):
            if math.isfinite(logprob):
                continue
            token_id = choice.token_ids[token_offset] if token_offset < completion_len else None
            tokenizer = tokenizer_getter()
            return {
                **common,
                "field": "choices.logprobs",
                "choice_index": choice.index,
                "token_offset": token_offset,
                "token_id": token_id,
                "completion_len": completion_len,
                "finish_reason": choice.finish_reason,
                **_token_window_context(
                    "completion",
                    choice.token_ids,
                    token_offset,
                    tokenizer=tokenizer,
                ),
                **_short_sequence_context(
                    "completion",
                    choice.token_ids,
                    tokenizer=tokenizer,
                ),
                **_prompt_tail_context(response.prompt_token_ids, tokenizer=tokenizer),
                "value": repr(logprob),
            }

    if response.prompt_logprobs is not None:
        for token_offset, logprob in enumerate(response.prompt_logprobs):
            if logprob is None or math.isfinite(logprob):
                continue
            token_id = response.prompt_token_ids[token_offset] if token_offset < prompt_len else None
            tokenizer = tokenizer_getter()
            return {
                **common,
                "field": "prompt_logprobs",
                "token_offset": token_offset,
                "token_id": token_id,
                **_token_window_context(
                    "prompt",
                    response.prompt_token_ids,
                    token_offset,
                    tokenizer=tokenizer,
                ),
                **_prompt_tail_context(response.prompt_token_ids, tokenizer=tokenizer),
                "value": repr(logprob),
            }

    return None


def _get_diagnostic_tokenizer(*objects: Any | None) -> Any | None:
    for obj in objects:
        tokenizer = _find_tokenizer_on_object(obj)
        if tokenizer is not None:
            return tokenizer
    return None


def _find_tokenizer_on_object(obj: Any | None) -> Any | None:
    if obj is None:
        return None
    if _can_decode_tokens(obj):
        return obj

    for attr in ("tokenizer", "_tokenizer"):
        tokenizer = getattr(obj, attr, None)
        if _can_decode_tokens(tokenizer):
            return tokenizer

    renderer = getattr(obj, "renderer", None)
    if renderer is not None:
        tokenizer = _find_tokenizer_on_object(renderer)
        if tokenizer is not None:
            return tokenizer

    get_tokenizer = getattr(obj, "get_tokenizer", None)
    if callable(get_tokenizer):
        try:
            tokenizer = get_tokenizer()
        except Exception:
            return None
        if _can_decode_tokens(tokenizer):
            return tokenizer

    return None


def _can_decode_tokens(tokenizer: Any | None) -> bool:
    return tokenizer is not None and callable(getattr(tokenizer, "decode", None))


def _token_window_context(
    prefix: str,
    token_ids: list[int],
    token_offset: int,
    *,
    tokenizer: Any | None,
    radius: int = 16,
) -> dict[str, Any]:
    start = max(0, token_offset - radius)
    end = min(len(token_ids), token_offset + radius + 1)
    window_ids = token_ids[start:end]
    return {
        f"{prefix}_token_window_start": start,
        f"{prefix}_token_window_end": end,
        f"{prefix}_token_window_token_ids": window_ids,
        **_decode_token_context(
            f"{prefix}_token_window",
            window_ids,
            tokenizer=tokenizer,
        ),
    }


def _short_sequence_context(
    prefix: str,
    token_ids: list[int],
    *,
    tokenizer: Any | None,
    max_tokens: int = 128,
) -> dict[str, Any]:
    if len(token_ids) > max_tokens:
        return {}
    return {
        f"{prefix}_token_ids": token_ids,
        **_decode_token_context(prefix, token_ids, tokenizer=tokenizer),
    }


def _prompt_tail_context(
    prompt_token_ids: list[int],
    *,
    tokenizer: Any | None,
    max_tokens: int = 64,
) -> dict[str, Any]:
    tail_ids = prompt_token_ids[-max_tokens:]
    return {
        "prompt_tail_token_start": len(prompt_token_ids) - len(tail_ids),
        "prompt_tail_token_ids": tail_ids,
        **_decode_token_context("prompt_tail", tail_ids, tokenizer=tokenizer),
    }


def _decode_token_context(
    prefix: str,
    token_ids: list[int],
    *,
    tokenizer: Any | None,
) -> dict[str, Any]:
    if not _can_decode_tokens(tokenizer):
        return {}

    context: dict[str, Any] = {}
    try:
        context[f"{prefix}_text"] = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
    except Exception as exc:
        context[f"{prefix}_decode_error"] = repr(exc)

    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_ids_to_tokens):
        try:
            context[f"{prefix}_tokens"] = list(convert_ids_to_tokens(token_ids))
        except Exception as exc:
            context[f"{prefix}_tokens_error"] = repr(exc)
    return context


def _non_finite_generate_error(context: dict[str, Any]) -> ErrorResponse:
    message = "Non-finite /v1/generate response value before JSON serialization: " + json.dumps(context, sort_keys=True)
    logger.error(message)
    return ErrorResponse(
        error=ErrorInfo(
            message=message,
            type="BadRequestError",
            param=context["field"],
            code=400,
        )
    )
