from __future__ import annotations

"""
Adapters to make Verifiers env parsing robust to non-vLLM backends (e.g., SGLang).

"""

from typing import Any, List, Tuple

from loguru import logger


def apply_verifiers_adapters(server_type: str) -> None:
    """Apply runtime patches to Verifiers for specific server types."""

    if server_type != "sglang":
        return

    # TODO: Upstream these patches to `verifiers` so we can drop the runtime shim.

    try:
        # Delayed import to avoid hard dependency if verifiers isn't installed.
        from verifiers.envs.environment import Environment  # type: ignore
    except Exception as e:
        logger.debug(f"Verifiers not available for patching: {e}")
        return

    # Keep a reference to the original implementation in case we need it later
    _orig = getattr(Environment, "process_chat_format_vllm", None)
    if _orig is None:
        logger.debug("Environment.process_chat_format_vllm missing; skipping SGLang adapter")
        return

    def _responses_have_usable_logprobs(responses: list[Any]) -> bool:
        for response in responses:
            try:
                choices = getattr(response, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                logprobs = getattr(choice, "logprobs", None)
                content = getattr(logprobs, "content", None) if logprobs is not None else None
                if content:
                    return True
            except Exception:
                continue
        return False

    def _encode_single_token(tokenizer: Any, token_text: str | None) -> int:
        if not token_text:
            raise ValueError("token text missing from logprob entry")

        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"token '{token_text}' produced {len(ids)} token ids")

        return ids[0]

    def _map_logprobs_to_token_ids(responses: list[Any], tokenizer: Any) -> None:
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError("tokenizer does not provide an `encode` method for logprob conversion")

        for response in responses:
            choices = getattr(response, "choices", None)
            if not choices:
                continue

            for choice in choices:
                logprobs = getattr(choice, "logprobs", None)
                content = getattr(logprobs, "content", None) if logprobs is not None else None
                if not content:
                    continue

                for item in content:
                    token_id = _encode_single_token(tokenizer, getattr(item, "token", None))
                    item.token = f"token_id:{token_id}"

                    top = getattr(item, "top_logprobs", None)
                    if top:
                        for candidate in top:
                            candidate_id = _encode_single_token(tokenizer, getattr(candidate, "token", None))
                            candidate.token = f"token_id:{candidate_id}"

    def _patched_process_chat_format_vllm(
        self: Any,
        prompt: list[dict[str, Any]],
        completion: list[dict[str, Any]],
        state: dict[str, Any],
        processing_class: Any,
        mask_env_responses: bool = False,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[float]]:
        """Patched version that preserves native logprobs when available."""
        responses = state["responses"]
        tokenizer = getattr(processing_class, "tokenizer", processing_class)

        if _responses_have_usable_logprobs(responses):
            _map_logprobs_to_token_ids(responses, tokenizer)
            return _orig(self, prompt, completion, state, processing_class, mask_env_responses)
        raise ValueError("SGLang response lacked logprobs; enable trainer.recompute_logprobs")

    # Apply the patch
    setattr(Environment, "process_chat_format_vllm", _patched_process_chat_format_vllm)
    logger.info("Applied Verifiers adapter: patched process_chat_format_vllm for SGLang backend")
