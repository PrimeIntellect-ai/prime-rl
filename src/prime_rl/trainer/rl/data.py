import pickle
import time
from pathlib import Path
from typing import TypedDict

import torch
import torch.distributed.distributed_c10d as c10d
from jaxtyping import Bool, Float, Int
from renderers import RendererConfig
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.trainer import FakeDataLoaderConfig, MissingMMImagePolicy
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport import MicroBatch, MicroBatchReceiver, TransportConfig, setup_micro_batch_receiver
from prime_rl.utils.logger import get_logger

# Poll interval for the worker wait on the master's micro-batch publish status.
_PUBLISH_POLL_SECONDS = 1.0


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
    env_names: list[str]

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]
    seq_lens: Int[Tensor, "segments"] | None

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

    def synchronize_state(self) -> None:
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
            "env_names": ["fake"] * input_ids.shape[0],
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": None,
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
            "env_names": ["fake"] * self.seq_len,
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": None,
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
        defer_mm_materialization: bool = False,
        renderer_config: RendererConfig | None = None,
        pack_multimodal: bool = False,
        micro_batch_transport_config: TransportConfig | None = None,
        missing_mm_image_policy: MissingMMImagePolicy = "placeholder_zero_loss",
    ):
        self.world = get_world()
        self._current_step = start_step
        self._micro_batch_transport_config = micro_batch_transport_config or config
        self._store = c10d._get_default_store()

        if self.world.is_master:
            self.packer: BasePacker = setup_packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                tokenizer=tokenizer,
                transport_config=config,
                pad_to_multiple_of=pad_to_multiple_of,
                start_step=start_step,
                pack_multimodal=pack_multimodal,
                micro_batch_transport_config=self._micro_batch_transport_config,
            )

        non_dp_world_size = self.world.world_size // dp_world_size
        dp_rank = self.world.rank // non_dp_world_size
        self.multi_run_manager = get_multi_run_manager()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(
            output_dir, dp_rank, start_step, self._micro_batch_transport_config
        )

        # Deferred materialization: each rank builds its own renderer once and
        # materializes pixels from the shipped image references in get_batch.
        self.defer_mm_materialization = defer_mm_materialization
        self.missing_mm_image_policy = missing_mm_image_policy
        self._renderer = None
        # Build the renderer only when one is configured. With default-on defer,
        # text-only runs leave renderer_config None and never receive mm_refs, so
        # the materialize path below is simply never hit.
        if defer_mm_materialization and renderer_config is not None:
            from renderers.base import create_renderer

            self._renderer = create_renderer(tokenizer, renderer_config)
        # Per-step materialization cost, surfaced as wandb time/mm_materialize.
        self.last_mm_materialize_time = 0.0
        self.last_mm_images_materialized = 0
        self.last_mm_images_placeholdered = 0

    def _publish_status_key(self) -> str:
        return f"micro_batch_publish/{self._current_step}"

    def _publish_micro_batch_status(self, *, ok: bool, error: str = "") -> None:
        self._store.set(self._publish_status_key(), pickle.dumps({"ok": ok, "error": error}))

    def _wait_for_micro_batch_status(self) -> None:
        # No deadline here: packing is generation-bound (the master blocks on the
        # orchestrator) and can legitimately take arbitrarily long, so a fixed
        # timeout would crash the run on slow generation rather than a real fault.
        # Liveness is already covered: a wedged master is killed by the packer
        # watchdog -> torchrun tears down the group, and a master pack error sets
        # ok=False below for an immediate coordinated fail. Genuine ZMQ delivery
        # is still bounded by the receiver's recv_timeout once published.
        key = self._publish_status_key()
        while not self._store.check([key]):
            time.sleep(_PUBLISH_POLL_SECONDS)

        status = pickle.loads(self._store.get(key))
        if not status.get("ok", False):
            error = status.get("error") or "unknown error"
            raise RuntimeError(f"Trainer master failed to pack micro-batch step {self._current_step}: {error}")

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer._arm_watchdog()
            try:
                self.packer.pack()
                self._publish_micro_batch_status(ok=True)
            except Exception as exc:
                self._publish_micro_batch_status(ok=False, error=repr(exc))
                raise
            finally:
                self.packer._disarm_watchdog()

        self._wait_for_micro_batch_status()
        self.receiver.wait()

    def synchronize_state(self) -> None:
        self.multi_run_manager.synchronize_state()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        self.last_mm_materialize_time = 0.0
        self.last_mm_images_materialized = 0
        self.last_mm_images_placeholdered = 0
        tensor_batches = [self._micro_batch_to_tensor(mb) for mb in micro_batches]
        self._current_step += 1
        return tensor_batches

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.multi_run_manager.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        mm_kwargs: dict[str, Tensor] | None = None
        if micro_batch.mm_kwargs:
            raise ValueError("Eager multimodal mm_kwargs transport is unsupported; use mm_refs")
        elif micro_batch.mm_refs is not None:
            # Deferred path: materialize pixels here from the shipped image
            # references. Returns torch tensors directly (no decode needed).
            # SCOPE (16a): this runs in every rank that holds the shard, so with
            # TP/CP/EP the same images are read+processed non_dp_world_size times.
            # Fine for DP-only; a per-DP-group materializer + broadcast is a 16b
            # perf item for large model-parallel runs.
            if self._renderer is None:
                raise ValueError(
                    "Received mm_refs but the trainer has no renderer: orchestrator/trainer "
                    "defer_mm_materialization config mismatch (trainer flag is off)."
                )
            from prime_rl.utils.mm import materialize_mm_refs, missing_file_uris, synthesize_placeholder_mm_kwargs

            materialize_start = time.perf_counter()
            try:
                mm_kwargs = materialize_mm_refs(self._renderer, micro_batch.mm_refs)
                self.last_mm_materialize_time += time.perf_counter() - materialize_start
                self.last_mm_images_materialized += len(micro_batch.mm_refs.uris)
            except FileNotFoundError as exc:
                self.last_mm_materialize_time += time.perf_counter() - materialize_start
                run_idx = next((i for i, n in enumerate(micro_batch.lora_num_tokens or []) if n > 0), None)
                if self.missing_mm_image_policy == "error":
                    get_logger().error(
                        f"mm materialization failed (run_idx={run_idx}, run_id={micro_batch.run_id}, "
                        f"run_step={micro_batch.run_step}, uris={micro_batch.mm_refs.uris}): {exc!r}"
                    )
                    raise
                placeholder_start = time.perf_counter()
                try:
                    mm_kwargs = synthesize_placeholder_mm_kwargs(self._renderer, micro_batch.mm_refs)
                except Exception as placeholder_exc:
                    get_logger().error(
                        f"mm placeholder synthesis failed after missing image "
                        f"(run_idx={run_idx}, run_id={micro_batch.run_id}, run_step={micro_batch.run_step}, "
                        f"uris={micro_batch.mm_refs.uris}): {placeholder_exc!r}"
                    )
                    raise placeholder_exc from exc
                self.last_mm_materialize_time += time.perf_counter() - placeholder_start
                self.last_mm_images_placeholdered += len(micro_batch.mm_refs.uris)
                micro_batch.loss_mask = [False] * len(micro_batch.loss_mask)
                missing_uris = missing_file_uris(micro_batch.mm_refs.uris)
                get_logger().warning(
                    "mm materialization missing image(s); using zero-loss placeholder "
                    f"(run_idx={run_idx}, run_id={micro_batch.run_id}, run_step={micro_batch.run_step}, "
                    f"missing_uris={missing_uris or ['<unknown: disappeared during read>']}, "
                    f"uris={micro_batch.mm_refs.uris})"
                )
            except Exception as exc:
                # The pre-forward all-reduce will fail-fast every rank, so make the
                # culprit obvious: which run (from lora_num_tokens) and which images.
                run_idx = next((i for i, n in enumerate(micro_batch.lora_num_tokens or []) if n > 0), None)
                get_logger().error(
                    f"mm materialization failed (run_idx={run_idx}, run_id={micro_batch.run_id}, "
                    f"run_step={micro_batch.run_step}, uris={micro_batch.mm_refs.uris}): {exc!r}"
                )
                raise
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
            rewards=torch.tensor(micro_batch.rewards, dtype=torch.float).unsqueeze(0)
            if micro_batch.rewards is not None
            else None,
            inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
            teacher_logprobs=torch.tensor(micro_batch.teacher_logprobs, dtype=torch.float).unsqueeze(0)
            if micro_batch.teacher_logprobs is not None
            else None,
            loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
            temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
            env_names=micro_batch.env_names,
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
            seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long) if micro_batch.seq_lens is not None else None,
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
            if micro_batch.mm_token_type_ids is not None
            else None,
            routed_experts=routed_experts,
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
