"""Kept-token (sampling mask) capture for top-p/top-k replay training.

Truncated sampling (top-p/top-k) renormalizes the sampling distribution
over a per-token "kept set"; the trainer replays these sets to renormalize its
own logprobs identically (DeepSeek V3.2 "Keep Sampling Mask", arXiv:2512.02556
§3.1). vLLM materializes the mask (it's the finite entries of the processed
logprobs) but never returns it, and its inter-process output structs are fixed
msgspec/dataclass schemas — so the kept ids ride the existing logprobs channel:

1. Engine-core worker: append ``[-1 separator | kept ids, -1 padded]`` columns
   to each ``LogprobsTensors`` row; everything downstream is width-agnostic.
2. API process: split the extension back off before vLLM builds logprob dicts
   (stock consumers see stock columns), accumulate the ragged rows per request,
   attach to the finished ``CompletionOutput``.
3. ``/inference/v1/generate``: serialize as base64
   ``{"ids": int32 concat, "counts": int32 per completion token}``. Kept sets
   are decode-only, so PD-disaggregated serving needs no router changes.

A count of 0 means no usable kept set (above the capture width, or the
position wasn't truncated); the trainer falls back to full-vocab logprobs.
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
# Fallback only — setup_vllm_env always stamps the env var from inference.kept_tokens.
KEPT_TOKENS_MAX_DEFAULT = 512

# Separator/padding token id in the widened logprobs rows. Never a valid
# vocab id, and stock vLLM never emits it (top-k indices and requested
# logprob_token_ids are always >= 0).
_SEPARATOR = -1

_EMPTY_KEPT_ROW = np.empty(0, dtype=np.int32)


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

    # Stop-token trimming can leave fewer response tokens than sampling steps.
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

    Intercepts ``self.sample`` for the duration of ``Sampler.forward`` to grab
    the full processed logprobs the stock forward discards; the kept set per
    row is their finite entries. Requires ``logprobs_mode="processed_logprobs"``,
    which also forces the sampling path that materializes the mask (FlashInfer's
    fused sampler doesn't). Speculative decoding bypasses this patch entirely —
    the server launcher rejects that combination.
    """
    import torch
    from vllm import envs
    from vllm.logger import init_logger
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler

    if not kept_tokens_enabled():
        return
    if envs.VLLM_USE_V2_MODEL_RUNNER:
        # The V2 runner samples through a separate Sampler class; capture would be inert.
        raise ValueError("VLLM_USE_V2_MODEL_RUNNER does not yet support: kept-tokens capture")
    if getattr(Sampler.forward, "_prime_rl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    cap = int(os.environ.get(KEPT_TOKENS_MAX_ENV, str(KEPT_TOKENS_MAX_DEFAULT)))
    original_forward = Sampler.forward

    def _forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
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
            output = original_forward(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)
        finally:
            del self.sample

        processed_logprobs = captured.get("processed_logprobs")
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        num_logprobs = sampling_metadata.max_num_logprobs
        if (
            processed_logprobs is None
            or logprobs_mode != "processed_logprobs"
            or output.logprobs_tensors is None
            # logprobs=-1 (full vocab) and scoring requests need no extension
            or num_logprobs is None
            or num_logprobs < 0
            or sampling_metadata.logprob_token_ids
        ):
            return output

        stock = output.logprobs_tensors
        num_rows = stock.logprob_token_ids.shape[0]
        if processed_logprobs.shape[0] != num_rows:
            return output

        # Fixed width `cap + 1` keeps this device-side (no host sync to stall the
        # engine loop): a finite entry in the extra column means the kept set
        # exceeds the cap, and such rows — like untruncated/greedy ones — ship an
        # empty extension with only the separator marking alignment.
        ids_dtype = stock.logprob_token_ids.dtype
        device = processed_logprobs.device
        width = min(cap + 1, processed_logprobs.shape[-1])
        ext_logprobs, ext_ids = processed_logprobs.topk(width, dim=-1)
        finite = ext_logprobs > float("-inf")
        valid = finite & ~finite[:, -1:]
        ext_ids = ext_ids.to(ids_dtype).masked_fill_(~valid, _SEPARATOR)

        # The splitter reads only id columns; the logprob extension is -inf filler.
        separator_ids = torch.full((num_rows, 1), _SEPARATOR, dtype=ids_dtype, device=device)
        extension_logprobs = torch.full((num_rows, width + 1), float("-inf"), device=device)
        output.logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.cat([stock.logprob_token_ids, separator_ids, ext_ids], dim=1),
            logprobs=torch.cat([stock.logprobs, extension_logprobs], dim=1),
            selected_token_ranks=stock.selected_token_ranks,
            cu_num_generated_tokens=stock.cu_num_generated_tokens,
        )
        return output

    _forward._prime_rl_kept_tokens = True
    Sampler.forward = _forward
    logger.warning("Installed kept-tokens sampler patch (cap=%d).", cap)


def monkey_patch_kept_tokens_output_capture():
    """Split kept-set extensions off logprobs rows in the API process.

    Strips the extension before vLLM builds per-position logprob dicts and
    attaches the accumulated rows to the finished ``CompletionOutput``.
    Detection is data-driven (the separator id), so rows without extensions
    pass through untouched.
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
        # Append one kept row per position even on extension-less steps, so rows
        # stay position-aligned if steps start (or stop) carrying separators.
        kept_rows: list[np.ndarray] | None = getattr(self, "_prime_kept_token_ids", None)
        if kept_rows is None:
            kept_rows = self._prime_kept_token_ids = []

        # Rows in one update come from one step's batch tensor: same separator column.
        separators = np.nonzero(token_ids[0] == _SEPARATOR)[0] if token_ids.size else np.empty(0, dtype=np.int64)
        if not separators.size:
            kept_rows.extend([_EMPTY_KEPT_ROW] * len(token_ids))
            return original_update(self, logprobs_lists)

        split = int(separators[0])
        for extension in token_ids[:, split + 1 :]:
            kept_rows.append(np.ascontiguousarray(extension[extension >= 0], dtype=np.int32))

        return original_update(
            self,
            LogprobsLists(
                token_ids[:, :split],
                logprobs[:, :split],
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
