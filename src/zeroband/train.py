import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import shardcast
import torch
import torch.distributed as dist
import torch.distributed.tensor
import wandb
from jaxtyping import Float
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from pydantic_config import parse_argv
from torch._guards import log as torch_log
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard  # type: ignore

from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.config import Config
from zeroband.training.data import BatchOutput, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty, selective_log_softmax
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import (
    MetricsAverager,
    PerfCounter,
    apply_ac_ckpt,
    log_prompt_response_samples,
    log_to_wandb,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.utils.http_monitor import HttpMonitor
from zeroband.utils.logger import get_logger
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.world_info import WorldInfo, get_world_info


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
        torch_log.setLevel(logging.CRITICAL)

    logger = get_logger("TRAIN")
    world_info = get_world_info()
    wandb_sample_history = None

    if config.ckpt.clean_rollout_path and config.ckpt.rollout_path is not None:
        logger.info(f"Cleaning rollout path {config.ckpt.rollout_path}")
        shutil.rmtree(config.ckpt.rollout_path, ignore_errors=True)

    logger.info(f"start training on {world_info.world_size} rank(s)")

    # Allow eager fallback during production so that training runs don't die if compile fails
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info)

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            shardcast.initialize(envs.SHARDCAST_OUTPUT_DIR, max_distribution_folders=config.max_async_level)

    model, tokenizer = get_model_and_tokenizer(config.model_name, config.train.attn_impl)

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
        model_reference, _ = get_model_and_tokenizer(config.model_name, config.train.attn_impl)
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

    total_samples = config.start_total_samples if config.start_total_samples is not None else 0
    training_progress = TrainingProgress(total_tokens=0, step=config.start_step, total_samples=total_samples)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump(), dir="wandb_logs")

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

    step_count_init = (
        config.start_rollout_step if config.start_rollout_step is not None else training_progress.step // config.optim.step_per_rollout
    )
    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        local_batch_size=local_batch_size,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
        step_count_init=step_count_init,
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

        # In the training loop
        for rollout_step in range(config.optim.step_per_rollout):
            logger.debug(f"training rollout step {rollout_step} / {config.optim.step_per_rollout}")
            metric_averager = MetricsAverager()
            loss_batch = torch.tensor(0.0, device="cuda")

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            data_per_rollout = next(logprobs_aware_iterator)
            num_grad_acc_steps = len(data_per_rollout)

            # Collect samples for WandB logging - do this ONCE per step
            if world_info.rank == 0 and config.wandb:
                # Use the first batch for logging (could be configurable if needed)
                batch = data_per_rollout[0]

                # Log the samples to WandB with history management
                try:
                    # Pass and update the sample history
                    wandb_sample_history = log_prompt_response_samples(tokenizer, batch, training_progress.step, wandb_sample_history)
                except Exception as e:
                    logger.warning(f"Error logging samples to WandB: {e}")

            # Now here's the complete grad_acc_step loop WITHOUT the WandB logging inside it:
            for grad_acc_step in range(num_grad_acc_steps):
                logger.debug(f"training grad_acc_step {grad_acc_step} / {num_grad_acc_steps}")
                batch = data_per_rollout[grad_acc_step]

                input_ids = batch["input_ids"].to("cuda")
                max_tokens = input_ids.shape[0] * input_ids.shape[1]

                loss_mask = batch["loss_mask"]

                ## correct aggregated metrics
                for rewards in batch["rewards"]:
                    metric_averager.update("sample_reward", rewards)
                for task_rewards in batch["task_rewards"]:
                    metric_averager.update("task_reward", task_rewards)
                for seq_lens in batch["seq_lens"]:
                    metric_averager.update("seq_lens", seq_lens)
                for length_penalties in batch["length_penalties"]:
                    metric_averager.update("length_penalties", length_penalties)
                for target_lengths in batch["target_lengths"]:
                    metric_averager.update("target_lengths", target_lengths)

                ## per micro batch metrics

                metric_averager.update("batch_reward", batch["rewards"].float().mean())
                metric_averager.update("batch_task_reward", batch["task_rewards"].float().mean())
                metric_averager.update("batch_seq_lens", batch["seq_lens"].float().mean())
                metric_averager.update("batch_length_penalties", batch["length_penalties"].float().mean())
                metric_averager.update("batch_target_lengths", batch["target_lengths"].float().mean())

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
                    config.clamp_log_prob_coef,
                    max_tokens,
                )

                entropy = entropy_loss(logits, loss_mask, config.temperature, max_tokens)

                loss = pg_loss - config.entropy_loss_coeff * entropy

                if config.kl_coef is not None:
                    kl = kl_penalty(original_logprobs, batch["ref_logprobs"].to("cuda"), loss_mask, max_tokens)
                    kl_scaled = kl * config.kl_coef
                    metric_averager.update("kl", kl_scaled)
                    loss = loss + kl_scaled

                loss = loss / num_grad_acc_steps

                inputs_ids_shape = input_ids.shape

                # Now we can delete the batch data
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
            training_progress.total_samples += config.optim.batch_size

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
                "total_samples": training_progress.total_samples,
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
                    log_to_wandb(metrics)
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
                    if envs.SHARDCAST_OUTPUT_DIR is not None:
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
            new_metrics = {
                "rollout_step": rollout_step,
                "step": training_progress.step,
                "time_rollout_step": time_rollout_step,
                "time_logprob": time_logprob,
                "time_data_loading": total_time_data_loading,
                "time_packing": total_time_packing,
            }
            log_to_wandb(new_metrics)

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
