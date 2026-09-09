from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.routing_replay import RoutingReplay, validate_routing_replay
from prime_rl.trainer.world import get_world
from prime_rl.transports.batch import (
    BatchReceiver,
    MicroBatch,
    TransportConfig,
    setup_batch_receiver,
)
from prime_rl.transports.batch.routing import validate_routed_experts


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
    routed_experts: Int[Tensor, "batch seq layers topk"] | RoutingReplay | None

    # Sampling-mask token ids per position, padded with -1 to the micro batch's
    # maximum mask size. A row containing only -1 has no mask.
    sampling_mask: Int[Tensor, "batch seq mask"] | None

    # Generic multimodal kwargs — flat dict matching the model's forward
    # signature (e.g. ``{"pixel_values": ..., "image_grid_thw": ...}`` for
    # Qwen3-VL; ``{"pixel_values": ...}`` for Gemma3-VL). The trainer
    # ``**`` -unpacks this into the forward call, so any HF VLM whose
    # processor and forward agree on kwarg names works out of the box.
    mm_kwargs: dict[str, Tensor] | None
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

        # This is a pretty ugly hack to ensure that all CP ranks in a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.dp_rank * 1000000 + self.batch_counter * 1000 + micro_batch_idx
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
            "env_names": ["fake"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": torch.tensor([input_ids.shape[0]], dtype=torch.int32),
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "mm_kwargs": None,
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
            "env_names": ["fake"] * self.seq_len,
            "sequence_lengths": [self.seq_len],
            "trace_ids": None,
            "branch_indices": None,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": torch.tensor([self.seq_len], dtype=torch.int32),
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "sampling_mask": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
        }


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

        self.receiver: BatchReceiver = setup_batch_receiver(output_dir, dp_rank, start_step, config)

    def wait_for_batch(self) -> None:
        self.receiver.wait()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        mm_kwargs: dict[str, Tensor] | None = None
        if micro_batch.mm_kwargs:
            # Each value is an EncodedTensor (dtype, shape, raw bytes).
            # No batch dim — the orchestrator concatenates per-image along
            # dim=0 generically, matching what each HF VLM's forward expects.
            mm_kwargs = {
                key: torch.frombuffer(bytearray(payload.data), dtype=_torch_dtype(payload.dtype)).reshape(payload.shape)
                for key, payload in micro_batch.mm_kwargs.items()
            }
        routed_experts = None
        packed_routed_experts = micro_batch.routed_experts
        if packed_routed_experts is not None:
            ids, weights, valid = validate_routed_experts(packed_routed_experts)
            if ids.shape[0] != len(micro_batch.input_ids):
                raise ValueError("Routing rows must match the packed microbatch token count")
            # Own the storage: bytes/NumPy views can be read-only or reused by a
            # receiver while a previous microbatch awaits checkpoint backward.
            id_tensor = torch.from_numpy(ids.astype(np.int32, copy=True)).unsqueeze(0)
            if weights is not None:
                _validate_packed_routing_validity(micro_batch, ids, weights, valid)
                weight_tensor = torch.from_numpy(weights.astype(np.float32, copy=True)).unsqueeze(0)
                routed_experts = RoutingReplay(id_tensor, weight_tensor)
            else:
                routed_experts = id_tensor
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
            mm_kwargs=mm_kwargs,
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


def _validate_packed_routing_validity(
    micro_batch: MicroBatch, ids: np.ndarray, weights: np.ndarray, valid: np.ndarray | None
) -> None:
    """Every input before a participating target needs captured routing.

    A target token's loss uses its preceding input row. Thus the final target
    row itself may be uncaptured, and zero-loss tails/padding need no routes.
    Raw-sample ingress separately restricts missing captures to terminal rows;
    this packed check protects the actual causal inputs of the training loss.
    """
    num_tokens = len(micro_batch.input_ids)
    if valid is None or valid.shape != (num_tokens,):
        raise ValueError("Full routing replay requires one validity flag per packed token")
    if micro_batch.mm_kwargs or micro_batch.mm_token_type_ids is not None:
        raise ValueError("Full routing replay does not support multimodal microbatches")
    lengths = micro_batch.seq_lens
    if (
        not lengths
        or any(length <= 0 for length in lengths)
        or sum(lengths) != num_tokens
        or lengths != micro_batch.sequence_lengths
    ):
        raise ValueError("Full routing replay requires aligned positive sequence_lengths and seq_lens")
    for name in ("loss_mask", "rl_weights", "ce_weights", "ref_kl_weights"):
        values = getattr(micro_batch, name)
        if values is not None and len(values) != num_tokens:
            raise ValueError(f"Full routing replay requires token-aligned {name}")
    # The byte codec has already checked finite values and zero placeholders.
    # Match the loss's actual RL membership; a zero-weight RL token is inactive.
    participating = np.asarray(micro_batch.loss_mask, dtype=bool).copy()
    if micro_batch.rl_weights is not None:
        participating &= np.asarray(micro_batch.rl_weights) != 0
    for stream in (micro_batch.ce_weights, micro_batch.ref_kl_weights):
        if stream is not None:
            participating |= np.asarray(stream) != 0
    start = 0
    for length in lengths:
        end = start + length
        targets = np.flatnonzero(participating[start:end])
        if len(targets):
            if targets[0] == 0:
                raise ValueError("Full routing replay requires loss-masked sequence starts")
            if not np.all(valid[start : start + int(targets[-1])]):
                raise ValueError("Full routing replay is missing a causal input row before a participating target")
        start = end

    # Full v1 is normalized Qwen3 top-k, not arbitrary mixture coefficients.
    # Check once at trainer ingress, while capture validity is still available.
    # Do not renormalize/reorder the actual payload or inspect GPU values later.
    if valid.any():
        if not np.allclose(weights.sum(axis=-1)[valid], 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Captured full-replay rows must have normalized top-k coefficients")
        ordered = np.sort(ids, axis=-1)
        # K is small; check one slot pair at a time to bound temporary memory.
        if any(
            np.any((ordered[..., slot - 1] == ordered[..., slot]) & valid[:, None]) for slot in range(1, ids.shape[-1])
        ):
            raise ValueError("Captured full-replay rows must contain unique expert IDs")


def prepare_router_replay(
    micro_batch: TensorMicroBatch,
    *,
    enabled: bool,
    mode: Literal["ids", "ids_and_weights"],
    model_config: Any,
) -> Tensor | RoutingReplay | None:
    """Choose the objective and validate full routing bounds before any H2D copy."""
    if mode not in ("ids", "ids_and_weights"):
        raise ValueError(f"Unknown router replay mode: {mode}")
    if mode == "ids_and_weights" and not enabled:
        raise ValueError("ids_and_weights requires enable_router_replay=true")
    if not enabled:
        return None
    replay = micro_batch["routed_experts"]
    if replay is None:
        raise ValueError("Router replay requires routed experts from inference (enable_return_routed_experts=True)")
    if mode == "ids":
        # Explicit IDs mode keeps the legacy recomputed-coefficient objective,
        # even when the inference server supplied both streams.
        return replay.ids if isinstance(replay, RoutingReplay) else replay
    if not isinstance(replay, RoutingReplay):
        raise ValueError("ids_and_weights requires captured routing weights; IDs-only data is insufficient")
    if micro_batch.get("mm_kwargs") or micro_batch.get("mm_token_type_ids") is not None:
        raise ValueError("Full routing replay does not support multimodal microbatches")
    if getattr(model_config, "model_type", None) != "qwen3_moe":
        raise ValueError("Full routing replay only supports custom Qwen3-MoE")
    expected_shape = (
        *micro_batch["input_ids"].shape,
        model_config.num_hidden_layers,
        model_config.num_experts_per_tok,
    )
    validate_routing_replay(replay, expected_shape=expected_shape, device=torch.device("cpu"))
    if not torch.isfinite(replay.weights).all():
        raise ValueError("Full routing replay weights must be finite")
    if replay.ids.numel() and (replay.ids.min() < 0 or replay.ids.max() >= model_config.num_experts):
        raise ValueError("Full routing replay expert IDs are outside the model's expert range")
    return replay


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype
