import asyncio
import time
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.utils import TypeBasedDispatcher

from prime_rl_sglang.nccl import (
    PrimeRLInitBroadcasterReqInput,
    PrimeRLUpdateWeightsReqInput,
    init_broadcaster,
    update_weights,
)

_PATCHED = False


def _response(success: bool, message: str):
    content = {"success": success, "message": message}
    return JSONResponse(content, status_code=HTTPStatus.OK if success else HTTPStatus.BAD_REQUEST)


async def _send_prime_rl_control_request(tokenizer_manager: TokenizerManager, obj):
    tokenizer_manager.auto_create_handle_loop()
    tokenizer_manager.send_to_scheduler.send_pyobj(obj)
    tokenizer_manager.model_update_result = asyncio.Future()
    result = await tokenizer_manager.model_update_result
    return result.success, result.message


def _patch_tokenizer_manager() -> None:
    async def prime_rl_init_broadcaster(self: TokenizerManager, obj: PrimeRLInitBroadcasterReqInput):
        return await _send_prime_rl_control_request(self, obj)

    async def prime_rl_update_weights(self: TokenizerManager, obj: PrimeRLUpdateWeightsReqInput):
        return await _send_prime_rl_control_request(self, obj)

    TokenizerManager.prime_rl_init_broadcaster = prime_rl_init_broadcaster
    TokenizerManager.prime_rl_update_weights = prime_rl_update_weights


def _patch_scheduler() -> None:
    original_init_request_dispatcher = Scheduler.init_request_dispatcher

    def init_request_dispatcher(self: Scheduler):
        original_init_request_dispatcher(self)
        self._request_dispatcher += TypeBasedDispatcher(
            [
                (PrimeRLInitBroadcasterReqInput, lambda req: init_broadcaster(self, req)),
                (PrimeRLUpdateWeightsReqInput, lambda req: update_weights(self, req)),
            ]
        )

    Scheduler.init_request_dispatcher = init_request_dispatcher


def _patch_routes() -> None:
    @http_server.app.post("/init_broadcaster")
    async def prime_rl_init_broadcaster(request: Request):
        data = await request.json()
        obj = PrimeRLInitBroadcasterReqInput(
            host=data["host"],
            port=data["port"],
            rank_offset=data["rank_offset"],
            inference_world_size=data["inference_world_size"],
            timeout=data["timeout"],
        )
        success, message = await http_server._global_state.tokenizer_manager.prime_rl_init_broadcaster(obj)
        return _response(success, message)

    @http_server.app.post("/update_weights")
    async def prime_rl_update_weights(request: Request):
        data = await request.json()
        obj = PrimeRLUpdateWeightsReqInput(flush_cache=data.get("flush_cache", True))
        success, message = await http_server._global_state.tokenizer_manager.prime_rl_update_weights(obj)
        return _response(success, message)


def _single_prompt_token_ids(adapted_request: GenerateReqInput) -> list[int] | None:
    input_ids = adapted_request.input_ids
    if input_ids is None:
        return None
    if len(input_ids) == 0:
        return []
    if isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def _completion_token_ids(ret_item: dict[str, Any]) -> list[int]:
    output_ids = ret_item.get("output_ids")
    if output_ids is not None:
        return list(output_ids)

    output_token_logprobs = ret_item.get("meta_info", {}).get("output_token_logprobs", [])
    return [token_id for _, token_id, *_ in output_token_logprobs]


def _chat_response_with_token_ids(response: Any, adapted_request: GenerateReqInput, ret: list[dict[str, Any]]) -> Any:
    if not hasattr(response, "model_dump"):
        return response

    payload = response.model_dump()
    prompt_token_ids = _single_prompt_token_ids(adapted_request)
    if prompt_token_ids is not None:
        payload["prompt_token_ids"] = prompt_token_ids

    for choice, ret_item in zip(payload.get("choices", []), ret):
        choice["token_ids"] = _completion_token_ids(ret_item)

    return JSONResponse(content=payload)


def _patch_chat_response_tokens() -> None:
    async def _handle_non_streaming_request(
        self: OpenAIServingChat,
        adapted_request: GenerateReqInput,
        request,
        raw_request: Request,
    ):
        try:
            ret = await self.tokenizer_manager.generate_request(adapted_request, raw_request).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(request, ret, int(time.time()))
        return _chat_response_with_token_ids(response, adapted_request, ret)

    OpenAIServingChat._handle_non_streaming_request = _handle_non_streaming_request


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    _patch_tokenizer_manager()
    _patch_scheduler()
    _patch_routes()
    _patch_chat_response_tokens()
    _PATCHED = True


def run_scheduler_process_with_prime_rl_patches(*args, **kwargs):
    apply_patches()

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
