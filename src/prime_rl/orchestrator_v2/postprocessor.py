"""PostProcessor: turns a popped batch into ``TrainingSample`` records and ships
them to the trainer.

Single responsibility. The orchestrator hands it a popped batch and a step
number; it runs the post-batch filters, persists rollouts to disk, tokenizes,
attaches metadata, optionally computes teacher logprobs (opd), and ships via
the configured sender. Returns a ``ProcessResult`` carrying the per-rollout
stats the metrics builder needs — so this class doesn't import pandas or
build the W&B dict itself.

No batch buffering, no step tracking, no checkpoint logic, no barriers — those
live on the orchestrator.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.filters import RolloutFilter, apply_filters
from prime_rl.orchestrator.trajectories import (
    backfill_rollout_tokens,
    interleave_rollout,
    offload_images_to_disk,
)
from prime_rl.orchestrator.utils import compute_teacher_logprobs
from prime_rl.orchestrator.vf_utils import save_rollouts
from prime_rl.transport import TrainingBatch, TrainingSample
from prime_rl.transport.base import TrainingBatchSender
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_rollout_dir, get_step_path


@dataclass
class ProcessResult:
    """Per-batch stats the metrics builder + orchestrator log line need.

    ``n_trainable == 0`` means every rollout was post-filtered; the orchestrator
    should drop the batch and retry on the next batch without incrementing the
    step counter.
    """

    n_trainable: int
    num_prefill_tokens: int
    num_decode_tokens: int
    rollout_prefill_lens: list[int]
    rollout_decode_lens: list[int]
    samples_per_rollout: list[int]
    parallel_preprocess_time: float
    teacher_logprobs_time: float
    samples_shipped: int


class PostProcessor:
    """Stateless across batches. ``process(batch, step)`` is the only public method."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        tokenizer,
        renderer,
        mm_token_type_ids_mapping: dict[int, int] | None,
        sender: TrainingBatchSender,
        teacher_inference: InferencePool | None,
        post_filters: list[RolloutFilter],
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.mm_token_type_ids_mapping = mm_token_type_ids_mapping
        self.sender = sender
        self.teacher_inference = teacher_inference
        self.post_filters = post_filters
        self.logger = get_logger()

    async def process(self, batch: list[vf.RolloutOutput], step: int) -> ProcessResult:
        """Run filter → tokenize → ship pipeline. Mutates ``batch`` in place
        with post-filter annotations and resolved samples."""
        # Post-batch filter annotation. Filtered rollouts stay in the batch
        # so metrics aggregations see them; they're excluded from ``train_examples``.
        if self.post_filters:
            await asyncio.to_thread(apply_filters, self.post_filters, batch)
        else:
            for r in batch:
                r.setdefault("filters", {})
                r.setdefault("is_filtered", False)

        n_trainable = sum(1 for r in batch if not r.get("is_filtered"))
        if n_trainable == 0:
            return ProcessResult(
                n_trainable=0,
                num_prefill_tokens=0,
                num_decode_tokens=0,
                rollout_prefill_lens=[],
                rollout_decode_lens=[],
                samples_per_rollout=[],
                parallel_preprocess_time=0.0,
                teacher_logprobs_time=0.0,
                samples_shipped=0,
            )

        # Persist rollouts (fire-and-forget background thread).
        step_path = get_step_path(get_rollout_dir(self.config.output_dir), step)
        await asyncio.to_thread(save_rollouts, batch, step_path / "train_rollouts.jsonl", exclude_keys={"trajectory"})

        offload_start = time.perf_counter()
        num_offloaded = offload_images_to_disk(batch, self.config.output_dir)
        if num_offloaded:
            self.logger.info(
                f"Offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
            )

        # Tokenize.
        parallel_start = time.perf_counter()
        needs_backfill = any(s["tokens"] is None for r in batch for s in r["trajectory"])
        if needs_backfill:
            self.logger.info(
                "Backfilling tokens for rollout trajectories (expected for "
                "training_mode=sft against an external teacher API)"
            )
            await asyncio.gather(
                *(asyncio.to_thread(backfill_rollout_tokens, r, self.tokenizer, renderer=self.renderer) for r in batch)
            )
        interleaved = await asyncio.gather(
            *(
                asyncio.to_thread(interleave_rollout, r, mm_token_type_ids_mapping=self.mm_token_type_ids_mapping)
                for r in batch
            )
        )

        result = ProcessResult(
            n_trainable=n_trainable,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            rollout_prefill_lens=[],
            rollout_decode_lens=[],
            samples_per_rollout=[],
            parallel_preprocess_time=0.0,
            teacher_logprobs_time=0.0,
            samples_shipped=0,
        )

        train_examples: list[TrainingSample] = []
        for rollout, samples in zip(batch, interleaved):
            prefill = 0
            decode = 0
            if samples is None:
                samples = []
            result.samples_per_rollout.append(len(samples))
            for sample in samples:
                sample.advantage = rollout.get("advantage")
                sample.reward = rollout.get("reward")
                sample.env_name = rollout.get("env_name")
                sample.training_mode = self.config.training_mode
                sample_decode = sum(sample.completion_mask)
                sample_prefill = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode
                decode += sample_decode
                prefill += sample_prefill
                if not rollout.get("is_filtered"):
                    train_examples.append(sample)
            result.rollout_prefill_lens.append(prefill)
            result.rollout_decode_lens.append(decode)
            result.num_prefill_tokens += prefill
            result.num_decode_tokens += decode

        result.parallel_preprocess_time = time.perf_counter() - parallel_start

        if self.config.training_mode == "opd" and self.teacher_inference is not None:
            assert self.config.teacher is not None
            t = time.perf_counter()
            teacher_logprobs_list = await compute_teacher_logprobs(
                clients=self.teacher_inference.train_clients,
                model_name=self.config.teacher.model.name,
                samples=train_examples,
            )
            for ex, lp in zip(train_examples, teacher_logprobs_list):
                ex.teacher_logprobs = lp
            result.teacher_logprobs_time = time.perf_counter() - t

        # Ship.
        await self.sender.send(TrainingBatch(examples=train_examples, step=step))
        result.samples_shipped = len(train_examples)
        return result
