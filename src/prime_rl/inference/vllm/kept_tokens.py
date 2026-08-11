"""Kept-token (sampling mask) capture for top-p/top-k replay training.

Truncated sampling (top-p/top-k) renormalizes the sampling distribution
over a per-token "kept set"; the trainer replays these sets to renormalize its
own logprobs identically (DeepSeek V3.2 "Keep Sampling Mask", arXiv:2512.02556
§3.1). vLLM materializes the mask (it's the finite entries of the processed
logprobs) but never returns it, and its inter-process output structs are fixed
msgspec/dataclass schemas — so the kept ids ride the existing logprobs channel:

1. Engine-core worker: append ``[-1 separator | kept ids, -1 padded]`` columns
   to each ``LogprobsTensors`` id row (ids only — nothing between sampler and
   API process pairs ids and logprobs column-wise); everything downstream is
   width-agnostic. Gated per engine on ``additional_config
   ["enable_return_sampling_mask"]``, snapshotted at ``Sampler.__init__``
   (fp32_lm_head-style) — no env vars.
2. API process: split the extension back off before vLLM builds logprob dicts
   (stock consumers see stock columns), accumulate the ragged rows per request,
   attach to the finished ``CompletionOutput``. Purely data-driven off the
   separator id, so it installs unconditionally.
3. ``/inference/v1/generate``: serialize as base64
   ``{"ids": int32 concat, "counts": int32 per completion token}``. Kept sets
   are decode-only, so PD-disaggregated serving needs no router changes.

A count of 0 means no usable kept set (above the capture width, or the
position wasn't truncated); the trainer falls back to full-vocab logprobs.
The orchestrator therefore bounds train-sampling ``top_k`` to
``SAMPLING_MASK_MAX`` so kept sets never overflow the capture width.

vLLM is growing native support under the same flag name —
``enable_return_sampling_mask`` (vllm-project/vllm#49577, unreleased) — with
the same semantics and constraints; once it ships in a released version this
module reduces to the ``routed_experts``-style API-layer glue
(``KeptTokensCapture`` + serializer).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pybase64
from vllm.outputs import RequestOutput

# Enable flag in vLLM's additional_config, named after the in-flight native
# vLLM flag (vllm-project/vllm#49577); set from inference.enable_return_sampling_mask.
SAMPLING_MASK_KEY = "enable_return_sampling_mask"

# Fixed kept-set capture width. Not configurable: the orchestrator rejects
# train-sampling top_k above this, so capture never overflows and replay is
# exact at every position.
SAMPLING_MASK_MAX = 512

# Separator/padding token id in the widened logprobs rows. Never a valid
# vocab id, and stock vLLM never emits it (top-k indices and requested
# logprob_token_ids are always >= 0).
SEPARATOR = -1

EMPTY_KEPT_ROW = np.empty(0, dtype=np.int32)


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

    Self-gates on ``additional_config["enable_return_sampling_mask"]``,
    snapshotted at ``Sampler.__init__`` where vLLM guarantees a
    ``set_current_vllm_config()`` context (same mechanism as fp32_lm_head).
    When enabled, intercepts ``self.sample`` for the duration of
    ``Sampler.forward`` to grab the full processed logprobs the stock forward
    discards; the kept set per row is their finite entries. Requires
    ``logprobs_mode="processed_logprobs"``, which also forces the sampling path
    that materializes the mask (FlashInfer's fused sampler doesn't).
    Speculative decoding bypasses this patch entirely, and the V2 model
    runner's separate Sampler class would leave it inert — the server launcher
    rejects both combinations.
    """
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.logger import init_logger
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler.forward, "_prime_rl_kept_tokens", False):
        return

    logger = init_logger(__name__)
    cap = SAMPLING_MASK_MAX
    original_init = Sampler.__init__
    original_forward = Sampler.forward

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        additional_config = get_current_vllm_config().additional_config or {}
        self._prime_return_sampling_mask = bool(additional_config.get(SAMPLING_MASK_KEY, False))
        if self._prime_return_sampling_mask:
            logger.warning("Kept-set sampling-mask capture ENABLED for this Sampler instance (cap=%d).", cap)

    def patched_forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        if not getattr(self, "_prime_return_sampling_mask", False):
            return original_forward(self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override)

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
        ext_ids = ext_ids.to(ids_dtype).masked_fill_(~valid, SEPARATOR)

        # Only the id tensor grows: the splitter reads ids alone, nothing between
        # sampler and API process pairs the two tensors column-wise (LogprobsLists
        # slices rows), and skipping a float extension halves the IPC overhead.
        separator_ids = torch.full((num_rows, 1), SEPARATOR, dtype=ids_dtype, device=device)
        output.logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.cat([stock.logprob_token_ids, separator_ids, ext_ids], dim=1),
            logprobs=stock.logprobs,
            selected_token_ranks=stock.selected_token_ranks,
            cu_num_generated_tokens=stock.cu_num_generated_tokens,
        )
        return output

    patched_forward._prime_rl_kept_tokens = True
    Sampler.__init__ = patched_init
    Sampler.forward = patched_forward
    logger.info("Installed kept-tokens sampler patch (self-gates on additional_config[%r]).", SAMPLING_MASK_KEY)


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

    def patched_update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        token_ids, logprobs, ranks, cu_num_generated_tokens = logprobs_lists
        # Append one kept row per position even on extension-less steps, so rows
        # stay position-aligned if steps start (or stop) carrying separators.
        kept_rows: list[np.ndarray] | None = getattr(self, "_prime_kept_token_ids", None)
        if kept_rows is None:
            kept_rows = self._prime_kept_token_ids = []

        # Rows in one update come from one step's batch tensor: same separator column.
        separators = np.nonzero(token_ids[0] == SEPARATOR)[0] if token_ids.size else np.empty(0, dtype=np.int64)
        if not separators.size:
            kept_rows.extend([EMPTY_KEPT_ROW] * len(token_ids))
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

    def patched_new_completion_output(self, *args, **kwargs):
        output = original_new_completion_output(self, *args, **kwargs)
        if output.finish_reason is not None and self.logprobs_processor is not None:
            kept_rows = getattr(self.logprobs_processor, "_prime_kept_token_ids", None)
            if kept_rows is not None:
                output.kept_token_ids = kept_rows
        return output

    patched_update_sample_logprobs._prime_rl_kept_tokens = True
    LogprobsProcessor._update_sample_logprobs = patched_update_sample_logprobs
    RequestState._new_completion_output = patched_new_completion_output
    logger.info("Installed kept-tokens output capture patch (splits -1-separated logprobs extensions).")
