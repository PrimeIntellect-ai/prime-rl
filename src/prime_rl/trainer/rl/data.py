import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.trainer import FakeDataLoaderConfig, MissingMMImagePolicy
from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport import (
    MicroBatch,
    MicroBatchReceiver,
    TransportConfig,
    setup_micro_batch_receiver,
)
from prime_rl.transport.types import MMRefs
from prime_rl.utils.logger import get_logger
from prime_rl.utils.mm import RawImageMaterializer, missing_file_uris


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

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Generic multimodal kwargs materialized by the trainer's model processor.
    mm_kwargs: dict[str, Tensor] | None
    mm_forward_policy: ForwardPolicy | None
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None

    # Packer-derived metadata used for run-local debug exports.
    run_id: str | None
    run_step: int | None


MaterializedMMParts = tuple[dict[str, Tensor] | None, ForwardPolicy | None]


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
        self.multi_run_manager = get_multi_run_manager()
        self.last_mm_materialize_time = 0.0
        self.last_mm_images_materialized = 0
        self.last_mm_images_placeholdered = 0

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
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = input_ids.shape[0]

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_forward_policy": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = self.seq_len
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
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_forward_policy": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        bin_cost: Callable[[Sequence[int]], int],
        config: TransportConfig,
        model_name: str,
        model_trust_remote_code: bool,
        missing_mm_image_policy: MissingMMImagePolicy = "placeholder_zero_loss",
    ):
        self.world = get_world()

        if self.world.is_master:
            self.packer: BasePacker = setup_packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                tokenizer=tokenizer,
                transport_config=config,
                pad_to_multiple_of=pad_to_multiple_of,
                bin_cost=bin_cost,
                start_step=start_step,
            )

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.multi_run_manager = get_multi_run_manager()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)
        self.mm_materializer = RawImageMaterializer(model_name, trust_remote_code=model_trust_remote_code)
        self.missing_mm_image_policy = missing_mm_image_policy
        self._reset_mm_stats()

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer._arm_watchdog()
            try:
                self.packer.pack()
            finally:
                self.packer._disarm_watchdog()
        self.receiver.wait()
        self.multi_run_manager.synchronize_state()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        self._reset_mm_stats()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _reset_mm_stats(self) -> None:
        self.last_mm_materialize_time = 0.0
        self.last_mm_images_materialized = 0
        self.last_mm_images_placeholdered = 0

    @staticmethod
    def _materialized_mm_parts(materialized: MaterializedMM | None) -> MaterializedMMParts:
        if materialized is None:
            return None, None
        return materialized.kwargs, materialized.forward_policy

    @staticmethod
    def _mm_run_context(micro_batch: MicroBatch) -> str:
        run_idx = next((i for i, n in enumerate(micro_batch.lora_num_tokens or []) if n > 0), None)
        return f"run_idx={run_idx}, run_id={micro_batch.run_id}, run_step={micro_batch.run_step}"

    def _materialize_mm_refs(self, micro_batch: MicroBatch, refs: MMRefs) -> MaterializedMMParts:
        materialize_start = time.perf_counter()
        try:
            materialized = self.mm_materializer.materialize(refs)
        except FileNotFoundError as exc:
            self.last_mm_materialize_time += time.perf_counter() - materialize_start
            if self.missing_mm_image_policy == "error":
                get_logger().error(
                    f"raw image materialization failed ({self._mm_run_context(micro_batch)}, uris={refs.uris}): {exc!r}"
                )
                raise
            return self._synthesize_missing_mm_placeholder(micro_batch, refs)

        self.last_mm_materialize_time += time.perf_counter() - materialize_start
        self.last_mm_images_materialized += len(refs.uris)
        return self._materialized_mm_parts(materialized)

    def _synthesize_missing_mm_placeholder(self, micro_batch: MicroBatch, refs: MMRefs) -> MaterializedMMParts:
        placeholder_start = time.perf_counter()
        materialized = self.mm_materializer.synthesize_placeholder(refs)
        self.last_mm_materialize_time += time.perf_counter() - placeholder_start
        self.last_mm_images_placeholdered += len(refs.uris)
        micro_batch.loss_mask = [False] * len(micro_batch.loss_mask)
        micro_batch.advantages = [0.0] * len(micro_batch.advantages)

        missing_uris = missing_file_uris(refs.uris)
        get_logger().warning(
            "raw image materialization missing image(s); using zero-loss placeholder "
            f"({self._mm_run_context(micro_batch)}, "
            f"missing_uris={missing_uris or ['<unknown: disappeared during read>']}, "
            f"uris={refs.uris})"
        )
        return self._materialized_mm_parts(materialized)

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.multi_run_manager.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        mm_kwargs: dict[str, Tensor] | None = None
        mm_forward_policy: ForwardPolicy | None = None
        if micro_batch.mm_kwargs is not None:
            raise ValueError("Processed multimodal mm_kwargs are unsupported in v1; use raw mm_refs")
        if micro_batch.mm_refs is not None:
            mm_kwargs, mm_forward_policy = self._materialize_mm_refs(micro_batch, micro_batch.mm_refs)
        routed_experts = None
        packed_routed_experts = micro_batch.routed_experts
        if packed_routed_experts is not None:
            routed_experts = (
                torch.frombuffer(
                    packed_routed_experts.data,
                    dtype=_torch_dtype(packed_routed_experts.dtype),
                )
                .reshape(packed_routed_experts.shape)
                .to(torch.int32)
                .unsqueeze(0)
            )
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
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
            mm_kwargs=mm_kwargs,
            mm_forward_policy=mm_forward_policy,
            mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
            if micro_batch.mm_token_type_ids is not None
            else None,
            routed_experts=routed_experts,
            rl_weights=torch.tensor(micro_batch.rl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.rl_weights is not None
            else None,
            ce_weights=torch.tensor(micro_batch.ce_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ce_weights is not None
            else None,
            ref_kl_weights=torch.tensor(micro_batch.ref_kl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ref_kl_weights is not None
            else None,
            run_id=micro_batch.run_id,
            run_step=micro_batch.run_step,
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
