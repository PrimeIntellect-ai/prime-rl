"""Dataset-backed ``TrainingBatch`` producer for the ``rl`` trainer path."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, cast

import tomli_w
from renderers.base import Renderer, build_training_sample, create_renderer

from prime_rl.configs.algorithm import DatasetSourceConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.algo.routing import stamp_loss_routing
from prime_rl.orchestrator.ckpt import CheckpointManager, setup_ckpt_manager
from prime_rl.orchestrator.trajectories import _encode_mm_kwargs
from prime_rl.orchestrator.types import Progress
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.chat_template import resolve_sft_messages, resolve_sft_tools
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import resolve_latest_ckpt_step

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.monitor.base import Monitor


STATIC_SFT_ENV_NAME = "static_sft"


class DatasetSampleSource:
    """Loads and renders a static SFT dataset as training samples."""

    def __init__(
        self,
        config: DatasetSourceConfig,
        *,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        seq_len: int,
        cursor: int = 0,
    ) -> None:
        self.config = config
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.cursor = cursor
        self.dataset: Dataset | None = None
        self._epoch = -1
        self._epoch_dataset: Dataset | None = None

    async def load(self) -> None:
        from prime_rl.trainer.sft.data import load_sft_dataset

        dataset = await asyncio.to_thread(load_sft_dataset, self.config)
        if self.config.max_examples is not None:
            dataset = dataset.take(min(self.config.max_examples, len(dataset)))
        if len(dataset) == 0:
            raise ValueError(f"Static SFT dataset {self.config.name!r} is empty")
        self.dataset = dataset
        get_logger().success(f"Static SFT dataset ready ({self.config.name}, {len(dataset)} rows)")

    def _next_example(self) -> dict:
        assert self.dataset is not None
        epoch, index = divmod(self.cursor, len(self.dataset))
        if epoch != self._epoch:
            self._epoch = epoch
            self._epoch_dataset = (
                self.dataset.shuffle(seed=self.config.seed + epoch) if self.config.shuffle else self.dataset
            )
        assert self._epoch_dataset is not None
        example = dict(self._epoch_dataset[index])
        self.cursor += 1
        return example

    def _render(self, example: dict) -> TrainingSample | None:
        messages = resolve_sft_messages(example)
        loss_mask = self.config.loss_mask

        def role_to_mask(message: dict) -> bool:
            role = message.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError(f"Invalid message role: {role}")
            return cast(bool, getattr(loss_mask, role))

        rendered = build_training_sample(
            self.renderer,
            messages,
            role_to_mask=role_to_mask,
            tools=resolve_sft_tools(example),
        )
        token_ids = list(rendered.token_ids)
        mask = list(rendered.loss_mask)
        mm_token_type_ids = list(rendered.mm_token_type_ids) if rendered.mm_token_type_ids is not None else None
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and eos_token_id not in token_ids:
            token_ids.append(eos_token_id)
            mask.append(True)
            if mm_token_type_ids is not None:
                mm_token_type_ids.append(0)
        max_length = min(self.config.max_length or self.seq_len, self.seq_len)
        if len(token_ids) > max_length:
            return None
        if mask:
            mask[0] = False
        if not any(mask):
            return None

        mm_kwargs = None
        if rendered.multi_modal_data is not None:
            mm_kwargs = _encode_mm_kwargs(rendered.multi_modal_data.mm_items)
        sample = TrainingSample(
            token_ids=token_ids,
            mask=mask,
            logprobs=[0.0] * len(token_ids),
            temperatures=[1.0] * len(token_ids),
            env_name=STATIC_SFT_ENV_NAME,
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=mm_token_type_ids,
        )
        stamp_loss_routing(sample, "ce")
        return sample

    def _build_batch(self, batch_size: int | None, token_batch_size: int | None) -> tuple[list[TrainingSample], int]:
        assert self.dataset is not None
        samples: list[TrainingSample] = []
        tokens = 0
        attempts = 0
        consecutive_failures = 0
        while (len(samples) < batch_size) if batch_size is not None else (tokens < cast(int, token_batch_size)):
            example = self._next_example()
            attempts += 1
            sample = self._render(example)
            if sample is None:
                consecutive_failures += 1
                if consecutive_failures >= len(self.dataset):
                    raise ValueError("Static SFT dataset produced no trainable samples in a full pass")
                continue
            consecutive_failures = 0
            samples.append(sample)
            tokens += len(sample.token_ids)
        return samples, attempts

    async def build_batch(
        self, batch_size: int | None, token_batch_size: int | None
    ) -> tuple[list[TrainingSample], int]:
        return await asyncio.to_thread(self._build_batch, batch_size, token_batch_size)


class DatasetBatchProducer:
    """Produces dataset-backed training batches for the trainer."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.dataset_config = config.static_sft_source
        assert self.dataset_config is not None
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        self.progress = Progress()
        self.ckpt_manager: CheckpointManager | None = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.sender: TrainingBatchSender | None = None
        self.monitor: Monitor | None = None
        self.heart = Heartbeat(config.heartbeat.url) if config.heartbeat is not None else None

    async def setup(self) -> DatasetSampleSource:
        config = self.config
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(to_toml_dict(config), f)

        resume_step = None
        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            resume_step = (
                resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
                if config.ckpt.resume_step == -1
                else config.ckpt.resume_step
            )
        if resume_step is not None and self.ckpt_manager is not None:
            self.ckpt_manager.load(self.progress, step=resume_step)
            self.progress.step = resume_step + 1
            get_logger().info(f"Resuming static SFT from step {resume_step}")

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        from prime_rl.trainer.model import setup_tokenizer

        tokenizer = setup_tokenizer(config.tokenizer)
        renderer = create_renderer(tokenizer, config.renderer)
        source = DatasetSampleSource(
            self.dataset_config,
            renderer=renderer,
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            cursor=self.progress.total_problems,
        )
        await source.load()
        from prime_rl.utils.monitor import setup_monitor

        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            prime_config=config.prime_monitor,
            output_dir=config.output_dir,
            tokenizer=tokenizer,
            run_config=config,
            keep_full_history=config.bench,
            train_env_names=[STATIC_SFT_ENV_NAME],
            eval_env_names=[],
        )
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)
        return source

    async def start(self) -> None:
        source = await self.setup()
        assert self.sender is not None and self.monitor is not None
        config = self.config
        get_logger().info(f"Starting static SFT loop (max_steps={config.max_steps or 'infinite'})")
        last_completed_step = self.progress.step - 1
        try:
            while config.max_steps is None or self.progress.step <= config.max_steps:
                step = self.progress.step
                started = time.perf_counter()
                samples, attempts = await source.build_batch(config.batch_size, config.token_batch_size)
                await self.sender.send(TrainingBatch(examples=samples, step=step))

                stable = get_step_path(get_broadcast_dir(config.output_dir), step) / "STABLE"
                await wait_for_path(stable)

                num_tokens = sum(len(sample.token_ids) for sample in samples)
                self.progress.total_tokens += num_tokens
                self.progress.total_samples += len(samples)
                self.progress.total_problems += attempts
                self.progress.step += 1
                last_completed_step = step
                if (
                    self.ckpt_manager is not None
                    and config.ckpt is not None
                    and config.ckpt.interval is not None
                    and step % config.ckpt.interval == 0
                    and (config.max_steps is None or step < config.max_steps)
                ):
                    await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)

                elapsed = time.perf_counter() - started
                self.monitor.log(
                    {
                        "progress/tokens": num_tokens,
                        "progress/samples": len(samples),
                        "progress/total_tokens": self.progress.total_tokens,
                        "progress/total_samples": self.progress.total_samples,
                        "time/step": elapsed,
                        "step": step,
                    },
                    step=step,
                )
                get_logger().success(
                    f"Step {step} | {format_time(elapsed):>7} | {len(samples)} samples | {num_tokens} tokens"
                )
                if self.heart is not None:
                    self.heart.beat()
        finally:
            self.monitor.save_final_summary()
            if self.ckpt_manager is not None and last_completed_step > 0:
                await asyncio.to_thread(self.ckpt_manager.save, self.progress, last_completed_step)
            self.sender.close()
