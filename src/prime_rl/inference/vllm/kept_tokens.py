"""Kept-token (sampling mask) capture for top-p/top-k replay training.

When rollouts sample with truncation (top-p/top-k/min-p), the effective
sampling distribution is renormalized over the surviving "kept set" of
tokens. The trainer must renormalize its own logits over the same set or
the importance ratio is biased (DeepSeek V3.2 "Keep Sampling Mask",
arXiv:2512.02556 §3.1; Cognition SWE-1.7 "sampling distribution replay").
The rollout-side logprobs are already correct: prime-rl runs vLLM with
``logprobs_mode="processed_logprobs"``, whose values are log-softmax over
the truncated logits. What's missing is the kept-set token IDs — vLLM
computes the mask inside ``apply_top_k_top_p`` and discards it.

Capture rides the existing logprobs pipeline so no vLLM structs need new
fields (``ModelRunnerOutput``/``EngineCoreOutput`` are fixed msgspec/
dataclass schemas that cross process boundaries):

1. Engine-core worker (``monkey_patch_kept_tokens_sampler``): after
   ``Sampler.sample`` returns the processed (masked, renormalized)
   logprobs, the kept set per position is exactly the finite entries.
   Extract them via top-k and append to the step's ``LogprobsTensors``
   rows as ``[stock columns | -1 separator | kept ids, -1 padded]``.
   The downstream D2H, scheduler slicing, and msgspec transport are
   width-agnostic, so the extension crosses to the API process untouched.

2. API process (``monkey_patch_kept_tokens_output_capture``): split the
   extension off each row before vLLM builds per-position logprob dicts
   (stock consumers — chat completions, evals — see exactly the stock
   columns), accumulate the ragged kept rows on the request's
   ``LogprobsProcessor``, and attach them to the final
   ``CompletionOutput`` as a dynamic ``kept_token_ids`` attribute.

3. ``/inference/v1/generate`` (``KeptTokensCapture`` +
   ``serialize_kept_tokens``): encode as compact base64 raw bytes
   ``{"ids": int32 concat, "counts": int32 per completion token}``,
   mirroring the routed_experts wire format. Kept sets are decode-only
   (aligned to sampled tokens), so the PD router passes them through
   unmodified — no router changes needed.

A position's count of 0 means "no usable kept set" (kept set larger than
``PRIME_KEPT_TOKENS_MAX``, i.e. barely-truncated). The trainer falls back
to full-vocab logprobs there; the resulting bias is bounded by
``-log(top_p)`` since the excluded tail mass is at most ``1 - top_p``.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pybase64
from vllm.outputs import RequestOutput

KEPT_TOKENS_ENV = "PRIME_RETURN_KEPT_TOKENS"
KEPT_TOKENS_MAX_ENV = "PRIME_KEPT_TOKENS_MAX"
KEPT_TOKENS_MAX_DEFAULT = 512

# Separator/padding token id in the widened logprobs rows. Never a valid
# vocab id, and stock vLLM never emits it (top-k indices and requested
# logprob_token_ids are always >= 0).
_SEPARATOR = -1


def kept_tokens_enabled() -> bool:
    return os.environ.get(KEPT_TOKENS_ENV) == "1"


def serialize_kept_tokens(kept_token_ids: list[np.ndarray] | None, num_tokens: int) -> dict[str, Any] | None:
    """Encode per-position kept-set rows as compact base64 raw bytes.

    Returns ``{"ids": b64(int32 concat), "counts": b64(int32[num_tokens])}``
    or None when nothing was captured. ``counts[i]`` is the kept-set size
    for completion token i (0 = absent); ``ids`` is the concatenation of
    all rows in position order.
    """
    if not kept_token_ids:
        return None

    # The engine emits one row per sampled token; stop-token trimming can
    # leave the response with fewer tokens than sampling steps.
    rows = kept_token_ids[:num_tokens]
    if len(rows) < num_tokens:
        rows = rows + [np.empty(0, dtype=np.int32)] * (num_tokens - len(rows))

    counts = np.fromiter((len(row) for row in rows), dtype=np.int32, count=num_tokens)
    if not int(counts.sum()):
        return None
    ids = np.ascontiguousarray(np.concatenate(rows).astype(np.int32, copy=False))
    return {
        "ids": pybase64.b64encode(memoryview(ids)).decode("ascii"),
        "counts": pybase64.b64encode(memoryview(np.ascontiguousarray(counts))).decode("ascii"),
    }


class KeptTokensCapture:
    """Records ``kept_token_ids`` off streamed ``RequestOutput``s per choice index."""

    def __init__(self, generator: AsyncIterator[RequestOutput]):
        self._generator = generator
        self.kept_tokens: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                encoded = serialize_kept_tokens(getattr(output, "kept_token_ids", None), len(output.token_ids))
                if encoded is not None:
                    self.kept_tokens[output.index] = encoded
            yield request_output


def monkey_patch_kept_tokens_sampler():
    """Widen sampler logprobs rows with the kept-set extension (engine-core process).

    Wraps ``Sampler.forward``, transiently intercepting ``self.sample`` to
    capture the full ``[num_positions, vocab]`` processed logprobs that the
    stock forward gathers top-k from and then discards. The kept set per
    row is the finite entries (non-kept tokens are exactly ``-inf`` after
    ``apply_top_k_top_p`` + log_softmax). Rows whose kept set exceeds
    ``PRIME_KEPT_TOKENS_MAX`` emit an empty extension (trainer falls back
    to full-vocab logprobs there). Speculative decoding is unsupported:
    vLLM's RejectionSampler builds logprobs via ``gather_logprobs`` and
    never calls the patched forward, so capture would be silently inert —
    the server launcher rejects the combination up front.

    Only active with ``logprobs_mode="processed_logprobs"`` — that mode
    already forces the PyTorch-native sampling path (FlashInfer's fused
    sampler never materializes the mask), so the processed logprobs are
    guaranteed to reflect the truncation actually used for sampling.
    """
    import torch
    from vllm import envs
    from vllm.logger import init_logger
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler

    if not kept_tokens_enabled():
        return
    if envs.VLLM_USE_V2_MODEL_RUNNER:
        # The V2 runner samples through a separate Sampler class this patch
        # doesn't cover; capture would be silently inert and training biased.
        raise ValueError("VLLM_USE_V2_MODEL_RUNNER does not yet support: kept-tokens capture")
    if getattr(Sampler.forward, "_prime_rl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    cap = int(os.environ.get(KEPT_TOKENS_MAX_ENV, str(KEPT_TOKENS_MAX_DEFAULT)))
    original_forward = Sampler.forward

    def _forward(self, logits, sampling_metadata, *args, **kwargs):
        captured: dict[str, torch.Tensor | None] = {}
        original_sample = self.sample

        def capturing_sample(*sample_args, **sample_kwargs):
            sampled, processed_logprobs = original_sample(*sample_args, **sample_kwargs)
            captured["processed_logprobs"] = processed_logprobs
            return sampled, processed_logprobs

        # Instance attribute shadows the bound method for this call only;
        # the model runner drives the sampler single-threaded.
        self.sample = capturing_sample
        try:
            output = original_forward(self, logits, sampling_metadata, *args, **kwargs)
        finally:
            del self.sample

        processed_logprobs = captured.get("processed_logprobs")
        # (logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
        override = kwargs.get("logprobs_mode_override") or (args[1] if len(args) > 1 else None)
        logprobs_mode = override or self.logprobs_mode
        num_logprobs = sampling_metadata.max_num_logprobs
        if (
            processed_logprobs is None
            or logprobs_mode != "processed_logprobs"
            or output.logprobs_tensors is None
            # -1 already returns the full processed logprobs; scoring
            # requests (logprob_token_ids) replace the whole tensors.
            or num_logprobs is None
            or num_logprobs < 0
            or sampling_metadata.logprob_token_ids
        ):
            return output

        stock = output.logprobs_tensors
        num_rows = stock.logprob_token_ids.shape[0]
        if processed_logprobs.shape[0] != num_rows:
            return output

        # Fixed extension width `cap + 1`, fully device-side: the extra column
        # detects overflow (a finite entry there means more than `cap` kept
        # ids), so no host sync (`.item()`/`.any()`) stalls the engine loop.
        # Finite entries all rank above -inf, so the top-`width` columns cover
        # every kept id; surplus columns land on -inf and become padding.
        # Overflow rows ship an empty extension (the trainer falls back to
        # full-vocab logprobs there), which also covers untruncated rows
        # (top_p = 1, all-greedy steps) — only the separator marks alignment.
        ids_dtype = stock.logprob_token_ids.dtype
        device = processed_logprobs.device
        width = min(cap + 1, processed_logprobs.shape[-1])
        ext_logprobs, ext_ids = processed_logprobs.topk(width, dim=-1)
        finite = ext_logprobs > float("-inf")
        overflow = finite[:, -1:]
        valid = finite & ~overflow
        padding_id = torch.tensor(_SEPARATOR, dtype=ids_dtype, device=device)
        ext_ids = torch.where(valid, ext_ids.to(ids_dtype), padding_id)
        ext_logprobs = torch.where(valid, ext_logprobs, torch.tensor(float("-inf"), device=device))

        separator_ids = torch.full((num_rows, 1), _SEPARATOR, dtype=ids_dtype, device=device)
        separator_logprobs = torch.full((num_rows, 1), float("-inf"), device=device)
        output.logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.cat([stock.logprob_token_ids, separator_ids, ext_ids], dim=1),
            logprobs=torch.cat([stock.logprobs, separator_logprobs, ext_logprobs], dim=1),
            selected_token_ranks=stock.selected_token_ranks,
            cu_num_generated_tokens=stock.cu_num_generated_tokens,
        )
        return output

    _forward._prime_rl_kept_tokens = True
    Sampler.forward = _forward
    logger.warning("Installed kept-tokens sampler patch (cap=%d).", cap)


def monkey_patch_kept_tokens_output_capture():
    """Split kept-set extensions off logprobs rows in the API process.

    Patches ``LogprobsProcessor._update_sample_logprobs`` to strip the
    ``-1``-separated extension before vLLM builds per-position logprob
    dicts (stock logprobs stay byte-identical for every consumer) and to
    accumulate the ragged kept rows, and ``RequestState._new_completion_output``
    to attach them to the finished ``CompletionOutput``. Detection is
    data-driven (the separator id), so this is a no-op for engines that
    never widen rows.
    """
    from vllm.logger import init_logger
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.engine.output_processor import RequestState
    from vllm.v1.outputs import LogprobsLists

    if getattr(LogprobsProcessor._update_sample_logprobs, "_prime_rl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    original_update = LogprobsProcessor._update_sample_logprobs
    original_new_completion_output = RequestState._new_completion_output

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        token_ids, logprobs, ranks, cu_num_generated_tokens = logprobs_lists
        # Track positions even on steps without extensions so a request whose
        # rows only start (or stop) carrying separators stays position-aligned.
        seen_positions = getattr(self, "_prime_kept_positions", 0)
        self._prime_kept_positions = seen_positions + len(token_ids)

        has_separator = token_ids.size > 0 and bool((token_ids == _SEPARATOR).any())
        kept_rows: list[np.ndarray] | None = getattr(self, "_prime_kept_token_ids", None)
        if not has_separator and kept_rows is None:
            return original_update(self, logprobs_lists)
        if kept_rows is None:
            # Extensions appeared mid-request: earlier positions carried none.
            kept_rows = [np.empty(0, dtype=np.int32)] * seen_positions
            self._prime_kept_token_ids = kept_rows
        if not has_separator:
            kept_rows.extend([np.empty(0, dtype=np.int32)] * len(token_ids))
            return original_update(self, logprobs_lists)

        stock_token_ids = []
        stock_logprobs = []
        for row_ids, row_logprobs in zip(token_ids, logprobs):
            separators = np.nonzero(row_ids == _SEPARATOR)[0]
            if separators.size:
                split = int(separators[0])
                extension = row_ids[split + 1 :]
                kept_rows.append(np.ascontiguousarray(extension[extension >= 0], dtype=np.int32))
                stock_token_ids.append(row_ids[:split])
                stock_logprobs.append(row_logprobs[:split])
            else:
                kept_rows.append(np.empty(0, dtype=np.int32))
                stock_token_ids.append(row_ids)
                stock_logprobs.append(row_logprobs)

        return original_update(
            self,
            LogprobsLists(
                np.stack(stock_token_ids),
                np.stack(stock_logprobs),
                ranks,
                cu_num_generated_tokens,
            ),
        )

    def _new_completion_output(self, *args, **kwargs):
        output = original_new_completion_output(self, *args, **kwargs)
        if output.finish_reason is not None and self.logprobs_processor is not None:
            kept_rows = getattr(self.logprobs_processor, "_prime_kept_token_ids", None)
            if kept_rows is not None:
                output.kept_token_ids = kept_rows
        return output

    _update_sample_logprobs._prime_rl_kept_tokens = True
    LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs
    RequestState._new_completion_output = _new_completion_output
    logger.info("Installed kept-tokens output capture patch (splits -1-separated logprobs extensions).")
