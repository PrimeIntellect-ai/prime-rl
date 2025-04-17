import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # type: ignore
import torch.distributed.tensor
import wandb
import shardcast

from zeroband.models import AttnImpl, ModelName, ModelType, get_model_and_tokenizer
from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.data import BatchOutput, CollateMode, DataConfig, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.loss import grpo_loss, kl_penalty, selective_log_softmax, entropy_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import (
    PerfCounter,
    apply_ac_ckpt,
    MetricsAverager,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
import logging

from zeroband.utils.http_monitor import HttpMonitor


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.01
    betas1: float = 0.9
    betas2: float = 0.99


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512
    grad_norm_clip: float = 1.0

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = False  #  disabling torch compile because its too unstable for RL
    liger_qwen: bool = False

    attn_impl: AttnImpl = "flex_attention"


class CkptConfig(BaseConfig):
    path: str | None = None
    interval: int | None = None
    resume: str | None = None

    rollout_path: str | None = None  # if rollout path is set we saved at each step

    @model_validator(mode="after")
    def check_path_and_interval(self):
        if (self.path is None) != (self.interval is None):
            raise ValueError("path and interval must be either both None or both not None")
        return self


class Config(BaseConfig):
    name_model: ModelName = "150M"

    ckpt: CkptConfig = CkptConfig()

    project: str = "prime_simple"
    wandb: bool = True

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    gpus_ids: list[int] | None = None

    temperature: float = 0.6  # todo remove this and add this to the data

    grpo_epsilon_low: float = 0.2
    grpo_epsilon_high: float = 0.2
    entropy_loss_coeff: float = 0.001
    clamp_log_prob_coef: float = 4.0

    max_async_level: int = 2  # the amount of rollout checkpoints to keep

    masked_mean_axis: int | None = None  # the axis to compute the mean of the masked values

    collate_mode: CollateMode = "padding"

    kl_coef: float | None = None

    start_step: int = 0

    @model_validator(mode="after")
    def check_liger(self):
        if self.train.liger_qwen:
            assert "Qwen" in self.name_model, "train.liger_qwen can only be applied to Qwen2 models."
        return self

    @model_validator(mode="after")
    def check_ckpt_interval(self):
        if self.ckpt.interval is not None:
            assert self.ckpt.interval % self.optim.step_per_rollout == 0, "ckpt.interval must be divisible by train.step_per_rollout"
        return self


def get_local_batch_size(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank


def get_logprobs(model: ModelType, input_ids: torch.Tensor, position_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids, position_ids=position_ids).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


def train(config: Config):
    if "ZERO_BAND_DEV" not in os.environ:
        torch._logging.set_logs(dynamo=logging.CRITICAL)  # silent flex attn error
        torch_log.setLevel(logging.CRITICAL)

    logger = get_logger()
    world_info = get_world_info()

    logger.info(f"start training on {world_info.world_size} rank(s)")

    # Allow eager fallback during production so that training runs don't die if compile fails
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info)

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        origin_data_dir = os.environ.get("SHARDCAST_OUTPUT_DIR", "./origin_data")
        shardcast.initialize(origin_data_dir, max_distribution_folders=config.max_async_level)

    model, tokenizer = get_model_and_tokenizer(config.name_model, config.train.attn_impl)

    perf_counter = PerfCounter(window_size=min(10, 2 * config.optim.step_per_rollout), model=model, seq_len=config.data.seq_length)

    if config.train.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    apply_fsdp(model, config.train.reshard_after_forward)

    if config.kl_coef is not None:
        model_reference, _ = get_model_and_tokenizer(config.name_model, config.train.attn_impl)
        apply_fsdp(model_reference, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.optim.lr,
        weight_decay=config.optim.optim.weight_decay,
        betas=(config.optim.optim.betas1, config.optim.optim.betas2),
    )

    scheduler = get_scheduler(
        sched_type=config.optim.sched_type,
        optimizer=optimizer,
        num_warmup_steps=config.optim.warmup_steps,
        num_stable_steps=config.optim.stable_steps,
        num_training_steps=config.optim.total_steps,
    )

    training_progress = TrainingProgress(total_tokens=0, step=config.start_step, total_problems=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if envs.PRIME_API_BASE_URL is not None:
        monitor = HttpMonitor()

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

        if config.kl_coef is not None:
            model_reference = torch.compile(model_reference) if not TYPE_CHECKING else model_reference

    if config.kl_coef is not None:
        logger.info(f"memory before model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        tensors_offloaded_reference = offload_model_to_cpu(model_reference)
        logger.info(f"memory after model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if config.ckpt.resume:
        logger.info(f"loading checkpoint from {config.ckpt.resume}")
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, scheduler, config.ckpt.resume)

    if training_progress.step % config.optim.step_per_rollout != 0:
        logger.warning(
            f"Resuming training from step {training_progress.step} seems invalid, as it should be multiple of train.step_per_rollout ({config.optim.step_per_rollout})"
            f"training will continue as if it was from step {training_progress.step - training_progress.step % config.optim.step_per_rollout}"
        )

    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        local_batch_size=local_batch_size,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
        step_count_init=training_progress.step // config.optim.step_per_rollout,
    )
    train_dataloader_iterator = iter(train_dataloader)

    previous_ckpt_rollout = []

    logger.info("Starting training loop")

    while True:
        time_start = time.time()

        total_time_data_loading = 0
        total_time_packing = 0

        # here we want to pre-compute the logprobs with the model before update
        with torch.no_grad():
            if config.kl_coef is not None:
                wake_up_model_from_cpu(model_reference, tensors_offloaded_reference)
                del tensors_offloaded_reference

            data: list[list[BatchOutput]] = []

            for rollout_step in range(config.optim.step_per_rollout):
                logger.debug(f"start rollout step {rollout_step} / {config.optim.step_per_rollout}")
                time_data_loading = time.time()

                batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                time_data_loading = time.time() - time_data_loading
                total_time_data_loading += time_data_loading

                time_0 = time.time()

                batch_packed = packed_batch(
                    batch_rollout, config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs, config.collate_mode
                )
                num_grad_acc_steps = len(batch_packed)

                time_1 = time.time()
                total_time_packing += time_1 - time_0

                for grad_acc_step in range(num_grad_acc_steps):
                    batch = batch_packed[grad_acc_step]
                    logger.debug(f"log prob grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")

                    input_ids = batch["input_ids"].to("cuda")

                    per_token_logps = get_logprobs(model, input_ids, batch["position_ids"], config.temperature)

                    batch["logprobs"] = per_token_logps.to("cpu")

                    if config.kl_coef is not None:
                        logger.debug(f"kl grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")
                        per_token_logps_reference = get_logprobs(model_reference, input_ids, batch["position_ids"], config.temperature)
                        batch["ref_logprobs"] = per_token_logps_reference.to("cpu")

                data.append(batch_packed)

            if config.kl_coef is not None:
                # if we don't manually reshard the the embed and lm head will conflict with the offloading because they will stay unshard until backward which we never call
                reshard_module(model_reference)
                tensors_offloaded_reference = offload_model_to_cpu(model_reference)

            logprobs_aware_iterator = iter(data)

            time_logprob = time.time() - time_start
            logger.info(f"Time to compute logprobs: {time_logprob:.2f} seconds")

        logger.debug("start training rollout")
        for rollout_step in range(config.optim.step_per_rollout):
            logger.debug(f"training rollout step {rollout_step} / {config.optim.step_per_rollout}")
            metric_averager = MetricsAverager()
            loss_batch = torch.tensor(0.0, device="cuda")

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            data_per_rollout = next(logprobs_aware_iterator)
            num_grad_acc_steps = len(data_per_rollout)

            for grad_acc_step in range(num_grad_acc_steps):
                logger.debug(f"training grad_acc_step {grad_acc_step} / {num_grad_acc_steps}")
                batch = data_per_rollout[grad_acc_step]

                input_ids = batch["input_ids"].to("cuda")
                loss_mask = batch["loss_mask"]

                for target_len, len_penalty, rewards, task_rewards, seq_lens in zip(
                    batch["target_lengths"], batch["length_penalties"], batch["rewards"], batch["task_rewards"], batch["seq_lens"]
                ):
                    metric_averager.update("target_lengths", target_len)
                    metric_averager.update("length_penalties", len_penalty)
                    metric_averager.update("sample_reward", rewards)
                    metric_averager.update("task_reward", task_rewards)
                    metric_averager.update("seq_lens", seq_lens)

                    metric_averager.update(f"length_penalty_{target_len}", len_penalty, need_sync=False)
                    metric_averager.update(f"sample_reward_{target_len}", rewards, need_sync=False)
                    metric_averager.update(f"task_reward_{target_len}", task_rewards, need_sync=False)
                    metric_averager.update(f"seq_lens_{target_len}", seq_lens, need_sync=False)

                # Forward
                logits: Float[torch.Tensor, "batch seq vocab"] = model(
                    input_ids=input_ids, position_ids=batch["position_ids"]
                ).logits.contiguous()

                # Gather args for grpo loss
                advantages = batch["advantages"].to("cuda")
                loss_mask = loss_mask.to("cuda")
                original_logprobs = batch["logprobs"].to("cuda")

                # Loss
                pg_loss, clip_ratio = grpo_loss(
                    logits,
                    input_ids,
                    advantages,
                    original_logprobs,
                    loss_mask,
                    config.temperature,
                    config.grpo_epsilon_low,
                    config.grpo_epsilon_high,
                    config.masked_mean_axis,
                    config.clamp_log_prob_coef,
                )

                entropy = entropy_loss(logits, loss_mask, config.temperature, config.masked_mean_axis)

                loss = pg_loss - config.entropy_loss_coeff * entropy

                if config.kl_coef is not None:
                    kl = kl_penalty(original_logprobs, batch["ref_logprobs"].to("cuda"), loss_mask, config.masked_mean_axis)
                    kl_scaled = kl * config.kl_coef
                    metric_averager.update("kl", kl_scaled)
                    loss = loss + kl_scaled

                loss = loss / num_grad_acc_steps

                inputs_ids_shape = input_ids.shape
                del batch, logits, input_ids, advantages, loss_mask, original_logprobs

                # Backward
                loss.backward()
                loss_batch += loss.detach().clone()

                metric_averager.update("pg_loss", pg_loss.detach().clone())
                metric_averager.update("entropy_loss", entropy.detach().clone())
                metric_averager.update("clip_ratio", clip_ratio.detach().clone())

                del loss, pg_loss, entropy, clip_ratio

            metric_averager.sync()

            dist.all_reduce(loss_batch, op=dist.ReduceOp.AVG)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_norm_clip).full_tensor()  # type: ignore (is a dtensor)

            logger.debug(f"loss: {loss_batch.item()}, grad_norm: {grad_norm.item()}")

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            logger.debug("optimizer step")

            training_progress.step += 1
            inner_lr = [group["lr"] for group in optimizer.param_groups][0]

            token_per_gpu = inputs_ids_shape[0] * inputs_ids_shape[1] * num_grad_acc_steps
            new_tokens = world_info.world_size * token_per_gpu
            perf_counter.count_tokens(new_tokens)
            training_progress.total_tokens += new_tokens
            training_progress.total_problems += config.optim.batch_size

            padding_proportion = (config.data.seq_length - metric_averager["seq_lens"].item() - 1) / config.data.seq_length

            metrics = {
                "Loss": loss_batch.item(),
                "step": training_progress.step,
                "rollout_step": rollout_step,
                "inner_lr": inner_lr,
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "grad_norm": grad_norm.item(),
                "padding_proportion": padding_proportion,
                "grad_acc_steps": num_grad_acc_steps,
                "total_problems": training_progress.total_problems,
            }

            for key, value in metric_averager.items():
                metrics[key] = value.item()

            log = (
                f"step: {training_progress.step}, "
                f"rollout_step: {training_progress.step // config.optim.step_per_rollout}, "
                f"loss: {loss_batch.item():.4f}, "
                f"clip_ratio: {metric_averager['clip_ratio'].item():.4f}, "
                f"sample_reward: {metric_averager['sample_reward'].item():.4f}, "
            )

            del loss_batch, grad_norm

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
                metrics["tokens_per_second"] = tokens_per_second
                metrics["tokens_per_second_per_gpu"] = tokens_per_second_per_gpu

                metrics["mfu"] = perf_counter.get_mfu()

                log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {metrics['mfu']:.2f}"

            if world_info.rank == 0:
                if config.wandb:
                    wandb.log(metrics)
                if envs.PRIME_API_BASE_URL is not None:
                    monitor.log(metrics)

            logger.info(log)

            # Lets do this first so that clients can start downloading as soon as possible
            if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
                logger.debug("saving rollout ckpt")
                rollout_step = training_progress.step // config.optim.step_per_rollout
                path = Path(config.ckpt.rollout_path) / f"step_{rollout_step}"
                previous_ckpt_rollout.append(path)
                safetensor_path = save_ckpt_for_rollout(model, path)
                if world_info.rank == 0:
                    logger.info(f"Broadcasting {safetensor_path}")
                    shardcast.broadcast(safetensor_path)  # TODO: Is this blocking?

                if len(previous_ckpt_rollout) > config.max_async_level:
                    path_to_delete = previous_ckpt_rollout.pop(0)
                    if path_to_delete.exists():
                        logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                        shutil.rmtree(path_to_delete, ignore_errors=True)

            if config.train.memory_profile and (training_progress.step == 2) and world_info.rank == 0:
                logger.info("Dumping memory snapshot.")
                pickle_path: str = config.train.memory_profile
                if not pickle_path.endswith(".pickle"):
                    pickle_path += ".pickle"
                torch.cuda.memory._dump_snapshot(pickle_path)
                torch.cuda.memory._record_memory_history(enabled=False)

            if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
                logger.info(
                    f"Saving checkpoint at step {training_progress.step}, rollout_step {training_progress.step // config.optim.step_per_rollout}"
                )
                save_checkpoint_fsdp_state(model, [optimizer], training_progress, scheduler, config.ckpt.path)

        time_rollout_step = time.time() - time_start
        logger.info(f"Finished rollout {rollout_step} step {training_progress.step}")
        if world_info.rank == 0 and config.wandb:
            new_metrics = {"rollout_step": rollout_step, "step": training_progress.step, "time_rollout_step": time_rollout_step}
            new_metrics["time_logprob"] = time_logprob
            new_metrics["time_data_loading"] = total_time_data_loading
            new_metrics["time_packing"] = total_time_packing
            wandb.log(new_metrics)

        if training_progress.step >= config.optim.total_steps:
            break

    if prefetcher is not None:
        prefetcher.shutdown()

    if world_info.rank == 0 and envs.PRIME_API_BASE_URL is not None:
        monitor.finish()

    logger.info("Training finished, exiting ...")
    logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    train(Config(**parse_argv()))
