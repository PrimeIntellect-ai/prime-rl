"""Dynamo vLLM worker entrypoint with prime-rl admin routes."""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from typing import Any, Callable
from uuid import uuid4

from prime_rl.utils.logger import get_logger

logger = get_logger()


async def _pause_generation(handler, body: dict[str, Any]) -> dict[str, str]:
    mode = body.get("mode", "keep")
    clear_cache = bool(body.get("clear_cache", False))
    await handler.engine_client.pause_generation(mode=mode, clear_cache=clear_cache)
    return {"status": "ok"}


async def _resume_generation(handler, _body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.resume_generation()
    return {"status": "ok"}


async def _update_weights(handler, body: dict[str, Any]) -> dict[str, str]:
    weight_dir = body.get("weight_dir")
    await handler.engine_client.pause_generation(mode="keep", clear_cache=False)
    try:
        await handler.engine_client.collective_rpc("update_weights_from_path", args=(weight_dir,))
        reset_prefix_cache = bool(body.get("reset_prefix_cache", True))
        if reset_prefix_cache and hasattr(handler.engine_client, "reset_prefix_cache"):
            await handler.engine_client.reset_prefix_cache()
    finally:
        await handler.engine_client.resume_generation()
    return {"status": "ok"}


async def _init_broadcaster(handler, body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.collective_rpc(
        "init_broadcaster",
        args=(
            body.get("host"),
            body.get("port"),
            body.get("rank_offset"),
            body.get("inference_world_size"),
            body.get("timeout"),
            body.get("quantize_in_weight_transfer", False),
        ),
    )
    return {"status": "ok"}


async def _liveness_probe(handler, _body: dict[str, Any]) -> dict[str, str]:
    await handler.engine_client.collective_rpc("liveness_probe")
    return {"status": "ok"}


def _sampling_params(handler, body: dict[str, Any]):
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(**handler.default_sampling_params)
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "min_tokens",
    ):
        if key in body and body[key] is not None and hasattr(params, key):
            setattr(params, key, body[key])

    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        params.max_tokens = max_tokens

    logprobs = body.get("logprobs")
    top_logprobs = body.get("top_logprobs")
    if logprobs is True:
        params.logprobs = top_logprobs or 1
    elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
        params.logprobs = logprobs
    elif top_logprobs not in (None, 0):
        params.logprobs = top_logprobs

    return params


def _decode_token(tokenizer, token_id: int, fallback: str | None = None) -> str:
    if fallback is not None:
        return fallback
    return tokenizer.decode([token_id])


def _token_logprobs(tokenizer, output) -> dict[str, Any] | None:
    if output.logprobs is None:
        return None

    content: list[dict[str, Any]] = []
    for index, token_id in enumerate(output.token_ids):
        if index >= len(output.logprobs):
            break
        logprobs = output.logprobs[index]
        if logprobs is None or token_id not in logprobs:
            continue

        selected = logprobs[token_id]
        token = _decode_token(tokenizer, token_id, getattr(selected, "decoded_token", None))
        top_entries = []
        for top_token_id, top in logprobs.items():
            top_token = _decode_token(tokenizer, top_token_id, getattr(top, "decoded_token", None))
            top_entries.append(
                {
                    "token": top_token,
                    "bytes": list(top_token.encode("utf-8")),
                    "logprob": float(top.logprob),
                }
            )

        content.append(
            {
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": float(selected.logprob),
                "top_logprobs": top_entries,
            }
        )

    if not content:
        return None
    return {"content": content, "refusal": None}


def _extract_selected_logprob(token_logprobs: Any, token_id: int | None = None) -> float | None:
    if not token_logprobs:
        return None

    if token_id is not None and hasattr(token_logprobs, "get"):
        selected = token_logprobs.get(token_id)
        if selected is None:
            return None
    else:
        selected = next(iter(token_logprobs.values()))

    if hasattr(selected, "logprob"):
        logprob = selected.logprob
    elif isinstance(selected, Mapping):
        logprob = selected.get("logprob")
    else:
        logprob = selected
    return float(logprob) if logprob is not None else None


def _extract_prompt_logprobs(prompt_logprobs: Any, prompt_token_ids: list[int]) -> list[float | None] | None:
    if prompt_logprobs is None:
        return None

    if hasattr(prompt_logprobs, "values"):
        prompt_logprobs = [prompt_logprobs]

    extracted: list[float | None] = []
    for index, token_logprobs in enumerate(prompt_logprobs):
        token_id = prompt_token_ids[index] if index < len(prompt_token_ids) else None
        extracted.append(_extract_selected_logprob(token_logprobs, token_id))
    return extracted


def _completion_logprobs(output) -> list[float]:
    if output.logprobs is None:
        return []

    logprobs: list[float] = []
    for index, token_id in enumerate(output.token_ids):
        if index >= len(output.logprobs):
            break
        logprob = _extract_selected_logprob(output.logprobs[index], token_id)
        logprobs.append(0.0 if logprob is None else logprob)
    return logprobs


def _encode_routed_experts(arr: Any) -> dict[str, Any]:
    return {
        "data": base64.b85encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


async def _generate(handler, body: dict[str, Any]) -> dict[str, Any]:
    from vllm.inputs.engine import tokens_input

    prompt_token_ids = body["prompt_token_ids"]
    request_id = body.get("request_id") or f"gen-{uuid4().hex[:16]}"
    model = body.get("model") or getattr(handler.config, "served_model_name", None) or handler.config.model
    sampling_params = _sampling_params(handler, body)
    sampling_params.logprobs = 1
    if body.get("prompt_logprobs"):
        sampling_params.prompt_logprobs = 1
    if hasattr(sampling_params, "skip_special_tokens"):
        sampling_params.skip_special_tokens = False

    routing = body.get("routing") or {}
    dp_rank = handler._to_local_dp_rank(routing.get("dp_rank"))
    priority = -int(routing.get("priority", body.get("priority", 0)))
    lora_request = handler._resolve_lora_request(model)

    final_output = None
    routed_experts_map: dict[int, dict[str, Any]] = {}
    try:
        async for output in handler.engine_client.generate(
            tokens_input(prompt_token_ids, cache_salt=body.get("cache_salt")),
            sampling_params,
            request_id,
            lora_request=lora_request,
            data_parallel_rank=dp_rank,
            priority=priority,
        ):
            for comp_output in output.outputs:
                if comp_output.routed_experts is not None:
                    routed_experts_map[comp_output.index] = _encode_routed_experts(comp_output.routed_experts)
            final_output = output
    except Exception:
        logger.exception("Dynamo generate request failed")
        raise

    if final_output is None:
        return {"error": "No output generated"}

    choices = []
    for output in final_output.outputs:
        token_ids = list(output.token_ids)
        choices.append(
            {
                "index": output.index,
                "token_ids": token_ids,
                "logprobs": _completion_logprobs(output),
                "finish_reason": output.finish_reason,
                "routed_experts": routed_experts_map.get(output.index),
            }
        )

    prompt_ids = list(getattr(final_output, "prompt_token_ids", prompt_token_ids))
    completion_len = sum(len(choice["token_ids"]) for choice in choices)
    return {
        "id": request_id,
        "model": model,
        "prompt_token_ids": prompt_ids,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_len,
            "total_tokens": len(prompt_ids) + completion_len,
        },
        "prompt_logprobs": _extract_prompt_logprobs(getattr(final_output, "prompt_logprobs", None), prompt_ids),
    }


async def _chat_completions(handler, body: dict[str, Any]) -> dict[str, Any]:
    from vllm.inputs import TokensPrompt

    prompt_token_ids = body["prompt_token_ids"]
    request_id = body.get("request_id") or f"chatcmpl-{uuid4().hex}"
    model = body.get("model") or getattr(handler.config, "served_model_name", None) or handler.config.model
    sampling_params = _sampling_params(handler, body)

    routing = body.get("routing") or {}
    dp_rank = handler._to_local_dp_rank(routing.get("dp_rank"))
    priority = -int(routing.get("priority", 0))
    lora_request = handler._resolve_lora_request(model)

    final_output = None
    try:
        async for output in handler.engine_client.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params,
            request_id,
            lora_request=lora_request,
            data_parallel_rank=dp_rank,
            priority=priority,
        ):
            final_output = output
    except Exception:
        logger.exception("Dynamo chat completion generation failed")
        raise

    if final_output is None or not final_output.outputs:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": 0,
                "total_tokens": len(prompt_token_ids),
            },
            "prompt_token_ids": prompt_token_ids,
        }

    choice_output = final_output.outputs[0]
    completion_ids = list(choice_output.token_ids)
    logprobs = _token_logprobs(handler.engine_client.tokenizer, choice_output)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": choice_output.text},
                "finish_reason": choice_output.finish_reason,
                "logprobs": logprobs,
                "token_ids": completion_ids,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_token_ids) + len(completion_ids),
        },
        "prompt_token_ids": prompt_token_ids,
    }


def _bind(handler, callback: Callable[[Any, dict[str, Any]], Any]):
    async def route(body: dict[str, Any] | None = None):
        return await callback(handler, body or {})

    return route


def patch_dynamo_vllm_worker() -> None:
    """Register prime-rl admin routes on Dynamo's worker system server."""
    from dynamo.vllm.handlers import BaseWorkerHandler

    if getattr(BaseWorkerHandler, "_prime_rl_admin_routes_patched", False):
        return

    original_init = BaseWorkerHandler.__init__

    def patched_init(self, runtime, *args, **kwargs) -> None:
        original_init(self, runtime, *args, **kwargs)
        if getattr(self, "_prime_rl_admin_routes_registered", False):
            return
        runtime.register_engine_route("pause", _bind(self, _pause_generation))
        runtime.register_engine_route("resume", _bind(self, _resume_generation))
        runtime.register_engine_route("update_weights", _bind(self, _update_weights))
        runtime.register_engine_route("init_broadcaster", _bind(self, _init_broadcaster))
        runtime.register_engine_route("liveness", _bind(self, _liveness_probe))
        runtime.register_engine_route("generate", _bind(self, _generate))
        runtime.register_engine_route("chat_completions", _bind(self, _chat_completions))
        self._prime_rl_admin_routes_registered = True
        logger.info(
            "Registered prime-rl Dynamo admin routes: "
            "/engine/pause, /engine/resume, /engine/update_weights, /engine/init_broadcaster, "
            "/engine/liveness, /engine/generate, /engine/chat_completions"
        )

    BaseWorkerHandler.__init__ = patched_init
    BaseWorkerHandler._prime_rl_admin_routes_patched = True


def main() -> None:
    patch_dynamo_vllm_worker()

    from dynamo.vllm.main import main as dynamo_vllm_main

    dynamo_vllm_main()


if __name__ == "__main__":
    main()
