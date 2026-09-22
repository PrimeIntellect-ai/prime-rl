# ruff: noqa: F722, F821 -- jaxtyping axis strings.
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.transports.batch.types import MicroBatch, MMRefs


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    ref_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperatures: Float[Tensor, "batch seq"]  # Per-token temperatures
    env_names: list[str]
    sequence_lengths: list[int]

    # Per-sequence branch identity, parallel to sequence_lengths; None on
    # synthetic data. "" / -1 mark an unknown sequence (e.g. a dummy batch).
    trace_ids: list[str] | None
    branch_indices: list[int] | None

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]
    seq_lens: Int[Tensor, "segments"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Sampling-mask token ids per position, padded with -1 to the micro batch's
    # maximum mask size. A row containing only -1 has no mask.
    sampling_mask: Int[Tensor, "batch seq mask"] | None

    # Raw image references, materialized immediately before forward.
    mm_refs: MMRefs | None
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None


def micro_batch_to_tensor(micro_batch: MicroBatch) -> TensorMicroBatch:
    """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
    routed_experts = None
    packed_routed_experts = micro_batch.routed_experts
    if packed_routed_experts is not None:
        routed_experts = (
            torch.frombuffer(
                packed_routed_experts.data,
                dtype=_torch_dtype(packed_routed_experts.dtype),
            )
            .reshape(packed_routed_experts.shape)
            .unsqueeze(0)
        )
    sampling_mask = None
    packed_sampling_mask = micro_batch.sampling_mask
    if packed_sampling_mask is not None:
        counts = np.frombuffer(packed_sampling_mask.counts, dtype=np.int32)
        ids = np.frombuffer(packed_sampling_mask.ids, dtype=np.int32)
        # Boolean assignment fills row-major, matching the flat concat order.
        max_mask_size = max(int(counts.max()), 1) if counts.size else 1
        padded = np.full((len(counts), max_mask_size), -1, dtype=np.int32)
        padded[np.arange(max_mask_size)[None, :] < counts[:, None]] = ids
        sampling_mask = torch.from_numpy(padded).unsqueeze(0)
    return TensorMicroBatch(
        input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
        advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
        inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
        ref_logprobs=torch.tensor(micro_batch.ref_logprobs, dtype=torch.float).unsqueeze(0)
        if micro_batch.ref_logprobs is not None
        else None,
        loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
        temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
        env_names=micro_batch.env_names,
        sequence_lengths=micro_batch.sequence_lengths,
        trace_ids=micro_batch.trace_ids,
        branch_indices=micro_batch.branch_indices,
        # Single adapter: every token in the batch belongs to it (padding included).
        lora_num_tokens=torch.tensor([len(micro_batch.input_ids)], dtype=torch.int32),
        seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long),
        mm_refs=micro_batch.mm_refs,
        mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
        if micro_batch.mm_token_type_ids is not None
        else None,
        routed_experts=routed_experts,
        sampling_mask=sampling_mask,
        rl_weights=torch.tensor(micro_batch.rl_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.rl_weights is not None
        else None,
        ce_weights=torch.tensor(micro_batch.ce_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.ce_weights is not None
        else None,
        ref_kl_weights=torch.tensor(micro_batch.ref_kl_weights, dtype=torch.float).unsqueeze(0)
        if micro_batch.ref_kl_weights is not None
        else None,
    )


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype
