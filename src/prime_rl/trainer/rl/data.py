from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.multimodal import ForwardPolicy, MultimodalAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch import (
    BatchReceiver,
    MicroBatch,
    MMRefs,
    TransportConfig,
    setup_batch_receiver,
)
from prime_rl.transports.batch.types import MICRO_BATCH_DTYPES


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
    env_names: list[str]  # One per sequence, parallel to sequence_lengths
    sequence_lengths: list[int]
    padding: int  # Trailing padding tokens, folded into the last sequence's length

    # Per-sequence branch identity, parallel to sequence_lengths; None on
    # synthetic data. "" / -1 mark an unknown sequence (e.g. a dummy batch).
    trace_ids: list[str] | None
    branch_indices: list[int] | None

    # Batch level
    seq_lens: Int[Tensor, "segments"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Sampling-mask token ids per position, padded with -1 to the micro batch's
    # maximum mask size. A row containing only -1 has no mask.
    sampling_mask: Int[Tensor, "batch seq mask"] | None
    top_logprobs_ids: Int[Tensor, "batch seq head"] | None
    top_logprobs_values: Float[Tensor, "batch seq head"] | None

    # Multimodal inputs. The data loader sets mm_refs (undecoded image references), and prepare_micro_batch
    # replaces them with mm_kwargs (CPU tensors for the vision encoder) and mm_forward_policy.
    mm_refs: MMRefs | None
    mm_kwargs: NotRequired[dict[str, Tensor] | None]
    mm_forward_policy: NotRequired[ForwardPolicy | None]
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        self.world = get_world()
        self.dp_world_size = dp_world_size
        self.non_dp_world_size = self.world.world_size // self.dp_world_size
        self.dp_rank = self.world.rank // self.non_dp_world_size

        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // self.dp_world_size
        self.seq_len = seq_len
        self.generate_samples = config.generate_samples
        self.batch_counter = 0

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        if not self.generate_samples:
            get_micro_batch_fn = self._get_micro_batch
        else:
            get_micro_batch_fn = self._get_sample_micro_batch

        # Seeded by the global micro batch index, so every DP layout trains the same batch and
        # all CP ranks of a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.batch_counter * 1000000 + self.dp_rank * self.num_micro_batches + micro_batch_idx
            generator = torch.Generator().manual_seed(seed)
            micro_batches.append(get_micro_batch_fn(generator))

        self.batch_counter += 1
        return micro_batches

    def _get_sample_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []
        sequence_lengths = []

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            sequence_lengths.append(seq_len_to_generate)
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * len(sequence_lengths),
            "sequence_lengths": sequence_lengths,
            "padding": 0,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": loss_mask.unsqueeze(0),
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "top_logprobs_ids": None,
            "top_logprobs_values": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
                generator=generator,
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "env_names": ["fake"],
            "sequence_lengths": [self.seq_len],
            "padding": 0,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "top_logprobs_ids": None,
            "top_logprobs_values": None,
            "mm_refs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }


def prepare_micro_batch(
    micro_batch: TensorMicroBatch, processor: Any | None, mm_adapter: MultimodalAdapter | None
) -> TensorMicroBatch:
    """Prepare a micro batch for the training loop. Currently only multimodal samples need preparation."""
    micro_batch["mm_kwargs"] = None
    micro_batch["mm_forward_policy"] = None
    mm_refs = micro_batch.get("mm_refs")
    if mm_refs is not None:
        if processor is None or mm_adapter is None:
            raise ValueError("Received multimodal samples but [model.vlm] is not set")
        materialized = materialize_mm_refs(mm_refs, processor, mm_adapter)
        micro_batch["mm_kwargs"] = materialized.kwargs
        micro_batch["mm_forward_policy"] = materialized.forward_policy
        micro_batch["mm_refs"] = None
    return micro_batch


class DataLoader:
    """Receives packed micro batches from the orchestrator, one stream per DP rank."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        config: TransportConfig,
    ):
        self.world = get_world()

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size

        self.receiver: BatchReceiver = setup_batch_receiver(
            output_dir,
            data_rank=dp_rank,
            rank=self.world.rank,
            world_size=self.world.world_size,
            current_step=start_step,
            transport=config,
        )

    def wait_for_batch(self) -> None:
        self.receiver.wait()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct of raw buffers) to a TensorMicroBatch (dict with tensors)."""
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
        top_ids = top_values = None
        # Masked tokens carry their sampler logprobs on the mask, the others on top_logprobs.
        heads = [
            head
            for head in (micro_batch.top_logprobs, packed_sampling_mask)
            if head is not None and head.logprobs is not None
        ]
        if heads:
            head_counts = [np.frombuffer(head.counts, dtype=np.int32) for head in heads]
            counts = sum(head_counts)
            rows = np.concatenate([np.repeat(np.arange(len(c)), c) for c in head_counts])
            order = np.argsort(rows, kind="stable")
            width = max(int(counts.max(initial=0)), 1)
            valid = np.arange(width)[None, :] < counts[:, None]
            ids = np.full((len(counts), width), -1, dtype=np.int32)
            values = np.zeros((len(counts), width), dtype=np.float32)
            ids[valid] = np.concatenate([np.frombuffer(head.ids, dtype=np.int32) for head in heads])[order]
            values[valid] = np.concatenate([np.frombuffer(head.logprobs, dtype=np.float32) for head in heads])[order]
            top_ids = torch.from_numpy(ids).unsqueeze(0)
            top_values = torch.from_numpy(values).unsqueeze(0)

        def stream(name: str, dtype: type[np.generic]) -> Tensor | None:
            """A per-token stream as a [1, seq] tensor: one numpy copy out of the received frame
            (which stays read-only) that also widens it to the trainer's dtype."""
            values = getattr(micro_batch, name)
            if values is None:
                return None
            return torch.from_numpy(np.frombuffer(values, MICRO_BATCH_DTYPES[name]).astype(dtype)).unsqueeze(0)

        return TensorMicroBatch(
            input_ids=stream("input_ids", np.int64),
            position_ids=stream("position_ids", np.int64),
            advantages=stream("advantages", np.float32),
            inference_logprobs=stream("inference_logprobs", np.float32),
            ref_logprobs=stream("ref_logprobs", np.float32),
            loss_mask=stream("loss_mask", np.bool_),
            temperatures=stream("temperatures", np.float32),
            env_names=micro_batch.env_names,
            sequence_lengths=micro_batch.sequence_lengths,
            padding=micro_batch.padding,
            trace_ids=micro_batch.trace_ids,
            branch_indices=micro_batch.branch_indices,
            seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long),
            mm_refs=micro_batch.mm_refs,
            mm_token_type_ids=stream("mm_token_type_ids", np.int64),
            routed_experts=routed_experts,
            sampling_mask=sampling_mask,
            top_logprobs_ids=top_ids,
            top_logprobs_values=top_values,
            rl_weights=stream("rl_weights", np.float32),
            ce_weights=stream("ce_weights", np.float32),
            ref_kl_weights=stream("ref_kl_weights", np.float32),
        )


SCORE_CENTERING_DATA_ERRORS = (
    "Missing or out-of-vocabulary sampler top logprobs",
    "Sampler top logprobs must be finite and nonpositive",
    "Duplicate sampler head ids",
    "Sampler head mass exceeds one",
    "Sampler probabilities must normalize over the recorded support",
    "Sampled token is missing from the recorded support",
)


def score_centering_data_errors(
    input_ids: Int[Tensor, "batch seq"],
    loss_mask: Bool[Tensor, "batch seq"],
    rl_weights: Float[Tensor, "batch seq"] | None,
    top_ids: Int[Tensor, "batch seq head"],
    top_values: Float[Tensor, "batch seq head"],
    sampling_mask: Int[Tensor, "batch seq mask"] | None,
    *,
    vocab_size: int,
) -> Bool[Tensor, " checks"]:
    """Flag each check in ``SCORE_CENTERING_DATA_ERRORS`` that the sampler evidence on RL tokens
    fails. Runs on the tensors' device without a host sync, so the trainer can validate on the GPU
    and check the flags once per step with ``raise_score_centering_data_errors``."""
    rl_mask = loss_mask if rl_weights is None else loss_mask & (rl_weights != 0)
    valid = top_ids >= 0
    ordered = top_ids.sort(-1).values
    head_mass = top_values.exp().masked_fill(~valid, 0.0).sum(-1)
    if sampling_mask is None:
        unnormalized = unsampled = torch.zeros_like(rl_mask)
    else:
        replay = (sampling_mask >= 0).any(-1)
        unnormalized = replay & ~torch.isclose(head_mass, torch.ones_like(head_mass), atol=1e-4, rtol=0)
        unsampled = replay & ~(sampling_mask == input_ids.unsqueeze(-1)).any(-1)
    errors = torch.stack(
        [
            ~valid.any(-1) | (top_ids >= vocab_size).any(-1),
            (valid & (~top_values.isfinite() | (top_values > 1e-6))).any(-1),
            ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any(-1),
            head_mass > 1.0001,
            unnormalized,
            unsampled,
        ],
        dim=-1,
    )
    return (errors & rl_mask.unsqueeze(-1)).flatten(0, -2).any(0)


def raise_score_centering_data_errors(errors: list[Bool[Tensor, " checks"]]) -> None:
    """Raise the first failed check of the first failing micro batch, given each micro batch's
    ``score_centering_data_errors`` in order. Syncs once."""
    if not errors:
        return
    failed = torch.stack(errors).flatten().cpu().nonzero()
    if failed.numel():
        raise ValueError(SCORE_CENTERING_DATA_ERRORS[int(failed[0]) % len(SCORE_CENTERING_DATA_ERRORS)])


def validate_score_centering_data(batch: TensorMicroBatch, *, vocab_size: int) -> None:
    """Validate sampler evidence before the LM head consumes it."""
    ids, values = batch["top_logprobs_ids"], batch["top_logprobs_values"]
    if ids is None or values is None:
        mask = batch["loss_mask"]
        if batch["rl_weights"] is not None:
            mask = mask & (batch["rl_weights"] != 0)
        if mask.any():
            raise ValueError("Score centering requires top logprobs on every RL token")
        return
    errors = score_centering_data_errors(
        batch["input_ids"],
        batch["loss_mask"],
        batch["rl_weights"],
        ids,
        values,
        batch["sampling_mask"],
        vocab_size=vocab_size,
    )
    raise_score_centering_data_errors([errors])


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype
