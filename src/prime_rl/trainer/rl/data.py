from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport import MicroBatch, MicroBatchReceiver, TransportConfig, setup_micro_batch_receiver


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    rewards: Float[Tensor, "batch seq"] | None
    inference_logprobs: Float[Tensor, "batch seq"]
    teacher_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperatures: Float[Tensor, "batch seq"]  # Per-token temperatures
    # Per-token top-k / top-p used during generation (-1 / 1.0 means no
    # truncation). The trainer applies the same truncation to its logits so
    # the importance ratio against ``inference_logprobs`` stays unbiased.
    top_k: Int[Tensor, "batch seq"] | None
    top_p: Float[Tensor, "batch seq"] | None
    env_names: list[str]

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Generic multimodal kwargs — flat dict matching the model's forward
    # signature (e.g. ``{"pixel_values": ..., "image_grid_thw": ...}`` for
    # Qwen3-VL; ``{"pixel_values": ...}`` for Gemma3-VL). The trainer
    # ``**`` -unpacks this into the forward call, so any HF VLM whose
    # processor and forward agree on kwarg names works out of the box.
    mm_kwargs: dict[str, Tensor] | None
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Selects loss dispatch (rl/opd → default loss with mode-specific taus,
    # sft → sft loss). All samples in a micro batch share the same mode.
    training_mode: str


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

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
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
            "rewards": None,
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "teacher_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "top_k": None,
            "top_p": None,
            "env_names": ["fake"] * input_ids.shape[0],
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "training_mode": "rl",
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
            "rewards": None,
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "teacher_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "top_k": None,
            "top_p": None,
            "env_names": ["fake"] * self.seq_len,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "training_mode": "rl",
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
        config: TransportConfig,
    ):
        self.world = get_world()

        if self.world.is_master:
            self.packer: BasePacker = setup_packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                tokenizer=tokenizer,
                transport_config=config,
                pad_to_multiple_of=pad_to_multiple_of,
                start_step=start_step,
            )

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.multi_run_manager = get_multi_run_manager()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)

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
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.multi_run_manager.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        mm_kwargs: dict[str, Tensor] | None = None
        if micro_batch.mm_kwargs:
            # Each value is an EncodedTensor (dtype, shape, raw bytes).
            # No batch dim — the orchestrator concatenates per-image along
            # dim=0 generically, matching what each HF VLM's forward expects.
            mm_kwargs = {
                key: torch.frombuffer(bytearray(payload.data), dtype=_torch_dtype(payload.dtype)).reshape(payload.shape)
                for key, payload in micro_batch.mm_kwargs.items()
            }
        return TensorMicroBatch(
            input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
            position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
            advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
            rewards=torch.tensor(micro_batch.rewards, dtype=torch.float).unsqueeze(0)
            if micro_batch.rewards is not None
            else None,
            inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
            teacher_logprobs=torch.tensor(micro_batch.teacher_logprobs, dtype=torch.float).unsqueeze(0)
            if micro_batch.teacher_logprobs is not None
            else None,
            loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
            temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
            top_k=torch.tensor(micro_batch.top_k, dtype=torch.long).unsqueeze(0)
            if micro_batch.top_k is not None
            else None,
            top_p=torch.tensor(micro_batch.top_p, dtype=torch.float).unsqueeze(0)
            if micro_batch.top_p is not None
            else None,
            env_names=micro_batch.env_names,
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
            if micro_batch.mm_token_type_ids is not None
            else None,
            routed_experts=torch.tensor(micro_batch.routed_experts, dtype=torch.int32).unsqueeze(
                0
            )  # [1, seq_len, layers, topk]
            if micro_batch.routed_experts is not None
            else None,
            training_mode=micro_batch.training_mode,
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
