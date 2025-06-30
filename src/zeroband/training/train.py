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
from jaxtyping import Float
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from zeroband.training import envs
from zeroband.training.checkpoint import (
    TrainingProgress,
    load_checkpoint_fsdp_state,
    save_checkpoint_fsdp_state,
    save_ckpt_for_rollout,
)
from zeroband.training.config import Config as TrainingConfig
from zeroband.training.data import DataLoader, FakeDataLoader
from zeroband.training.logger import setup_logger
from zeroband.training.loss import entropy_loss, grpo_loss, selective_log_softmax
from zeroband.training.utils import (
    MetricsAverager,
    OffloadedTensor,
    PerfCounter,
    apply_ac_ckpt,
    copy_model_to_cpu,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


def get_local_batch_size(batch_size: int, micro_bs: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    return batch_size


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=layer_reshard_after_forward,
        )
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank


def get_logprobs(
    model: ModelType,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(
        input_ids=input_ids, position_ids=position_ids
    ).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


@clean_exit
def train(config: TrainingConfig):
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)

    world_info = get_world_info()
    logger = setup_logger(config.log, world_info)

    if config.ckpt.clean_rollout_path and config.ckpt.rollout_path is not None:
        logger.info(f"Cleaning rollout path {config.ckpt.rollout_path}")
        shutil.rmtree(config.ckpt.rollout_path, ignore_errors=True)

    logger.info(f"start training on {world_info.world_size} rank(s)")

    # Allow eager fallback during production so that training runs don't die if compile fails
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, world_info)

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            shardcast.initialize(
                envs.SHARDCAST_OUTPUT_DIR,
                max_distribution_folders=config.max_async_level,
            )

    model, tokenizer = get_model_and_tokenizer(config.model.name, config.train.attn_impl)

    perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length)

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

    if config.recompute_logprobs:
        model_for_logprob_only, _ = get_model_and_tokenizer(config.model.name, config.train.attn_impl)
        apply_fsdp(model_for_logprob_only, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.optim.lr,
        weight_decay=config.optim.optim.weight_decay,
        betas=(config.optim.optim.betas1, config.optim.optim.betas2),
    )

    total_samples = config.start_total_samples if config.start_total_samples is not None else 0
    training_progress = TrainingProgress(total_tokens=0, step=config.start_step, total_samples=total_samples)

    # Setup the monitor
    monitor = setup_monitor(config.monitor, run_config=config)

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

        if config.recompute_logprobs:
            model_for_logprob_only: ModelType = torch.compile(model_for_logprob_only)

    tensor_offloaded_repository: dict[int, OffloadedTensor] = {}

    if config.recompute_logprobs:
        tensor_offloaded_repository[0] = offload_model_to_cpu(model_for_logprob_only)

    if config.ckpt.resume:
        logger.info(f"loading checkpoint from {config.ckpt.resume}")
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.resume)

    if config.data.fake:
        train_dataloader = FakeDataLoader(
            config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs, local_batch_size
        )
    else:
        train_dataloader = DataLoader(config.data.path, config.start_step)

    previous_ckpt_rollout = []

    logger.info("Starting training loop")

    while True:
        time_start = time.time()
        logger.info(f"start training step {training_progress.step}")

        micro_batches = train_dataloader.get_batch()
        time_data_loading = time.time() - time_start

        # here we want to pre-compute the logprobs with the model before update
        time_logprob_start = time.time()
        with torch.no_grad():
            if config.recompute_logprobs:
                og_infer_step = training_progress.step - config.max_async_level
                infer_step = max(og_infer_step, 0)
                wake_up_model_from_cpu(model_for_logprob_only, tensor_offloaded_repository[infer_step])

                if og_infer_step == infer_step:
                    del tensor_offloaded_repository[infer_step]

            logger.info(f"start logprob recomputation step {training_progress.step}")

            num_grad_acc_steps = len(micro_batches)

            for grad_acc_step in range(num_grad_acc_steps):
                batch = micro_batches[grad_acc_step]

                # Only compute logprobs if not using vllm logprobs or if the batch doesn't have them
                if config.recompute_logprobs:
                    logger.debug(
                        f"log prob grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}"
                    )

                    input_ids = batch["input_ids"].to("cuda")

                    model_for_logprob = model_for_logprob_only if config.recompute_logprobs else model
                    per_token_logps = get_logprobs(
                        model_for_logprob, input_ids, batch["position_ids"], batch["temperature"]
                    )

                    batch["logprobs"] = per_token_logps.to("cpu")

            if config.recompute_logprobs:
                # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
                # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
                reshard_module(model_for_logprob_only)
                offload_model_to_cpu(model_for_logprob_only)

        time_logprobs = time.time() - time_logprob_start
        time_data_and_logprob = time.time() - time_start
        logger.info(f"Time to compute logprobs and data preprocessing: {time_data_and_logprob:.2f} seconds")
        logger.debug(f"Time to data loading: {time_data_loading:.2f} seconds")
        logger.debug(f"Time to logprob computation: {time_logprobs:.2f} seconds")

        metric_averager = MetricsAverager()
        loss_batch = torch.tensor(0.0, device="cuda")

        if config.train.memory_profile and world_info.rank == 0:
            torch.cuda.memory._record_memory_history()

        num_grad_acc_steps = len(micro_batches)

        for grad_acc_step in range(num_grad_acc_steps):
            logger.debug(f"training grad_acc_step {grad_acc_step} / {num_grad_acc_steps}")
            batch = micro_batches[grad_acc_step]

            input_ids = batch["input_ids"].to("cuda")
            if config.normalize_batch_to_token_count:
                max_tokens = int(batch["total_tokens"])
            else:
                max_tokens = input_ids.shape[0] * input_ids.shape[1]

            loss_mask = batch["loss_mask"]

            # Forward
            logits: Float[torch.Tensor, "batch seq vocab"] = model(
                input_ids=input_ids, position_ids=batch["position_ids"]
            ).logits.contiguous()

            # Gather args for grpo loss
            advantages = batch["advantages"].to("cuda")
            loss_mask = loss_mask.int().to("cuda")
            original_logprobs = batch["logprobs"].to("cuda")

            # Loss

            loss, clip_ratio = grpo_loss(
                logits,
                input_ids,
                advantages,
                original_logprobs,
                loss_mask,
                batch["temperature"],
                max_tokens,
                config.grpo.off_policy,
            )

            with torch.no_grad():
                entropy = entropy_loss(logits, loss_mask, batch["temperature"], max_tokens)

            loss = loss / num_grad_acc_steps

            inputs_ids_shape = input_ids.shape

            # Now we can delete the batch data
            del batch, logits, input_ids, advantages, loss_mask, original_logprobs

            # Backward
            loss.backward()
            loss_batch += loss.detach().clone()

            metric_averager.update("losses/entropy_loss", entropy.detach().clone())

            if clip_ratio is not None:
                metric_averager.update("losses/clip_ratio", clip_ratio.detach().clone())

            del loss, entropy, clip_ratio

        metric_averager.sync()

        dist.all_reduce(loss_batch, op=dist.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_norm_clip).full_tensor()  # type: ignore (is a dtensor)

        logger.debug(f"loss: {loss_batch.item()}, grad_norm: {grad_norm.item()}")

        optimizer.step()
        optimizer.zero_grad()

        logger.debug("optimizer step")

        training_progress.step += 1
        inner_lr = [group["lr"] for group in optimizer.param_groups][0]

        token_per_gpu = inputs_ids_shape[0] * inputs_ids_shape[1] * num_grad_acc_steps
        new_tokens = world_info.world_size * token_per_gpu
        perf_counter.count_tokens(new_tokens)
        training_progress.total_tokens += new_tokens
        training_progress.total_samples += config.optim.batch_size

        metrics = {
            "step": training_progress.step,
            "losses/Loss": loss_batch.item(),
            "train/inner_lr": inner_lr,
            "train/total_tokens": training_progress.total_tokens,
            "train/total_samples": training_progress.total_samples,
            "losses/grad_norm": grad_norm.item(),
        }

        for key, value in metric_averager.items():
            metrics[key] = value.item()

        log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}, "

        del loss_batch, grad_norm

        tokens_per_second = perf_counter.get_tokens_per_second()
        if tokens_per_second is not None:
            tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
            mfu = perf_counter.get_mfu()
            metrics.update(
                {
                    "perf/tokens_per_second": tokens_per_second,
                    "perf/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "perf/mfu": mfu,
                }
            )

            log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {mfu:.2f}"

        if world_info.rank == 0:
            monitor.log(metrics)

        logger.info(log)

        time_rollout_ckpt = None
        time_shardcast = None
        time_rollout_delete = None

        # Lets do this first so that clients can start downloading as soon as possible
        if config.ckpt.rollout_path is not None:
            logger.debug("saving rollout ckpt")
            path = Path(config.ckpt.rollout_path) / f"step_{training_progress.step}"
            previous_ckpt_rollout.append(path)
            t0 = time.time()
            safetensor_path = save_ckpt_for_rollout(model, tokenizer, path, async_save=config.ckpt.async_save)
            time_rollout_ckpt = time.time() - t0

            time_shardcast = time.time()
            if world_info.rank == 0:
                if envs.SHARDCAST_OUTPUT_DIR is not None:
                    logger.info(f"Broadcasting {safetensor_path}")
                    shardcast.broadcast(safetensor_path)  # TODO: Is this blocking?
            time_shardcast = time.time() - time_shardcast

            time_rollout_delete = time.time()
            if len(previous_ckpt_rollout) > config.max_async_level:
                path_to_delete = previous_ckpt_rollout.pop(0)
                ckpt_step = int(str(path_to_delete).split("_")[-1])

                should_keep = config.ckpt.interval_rollout is not None and ckpt_step % config.ckpt.interval_rollout == 0
                if path_to_delete.exists() and not should_keep:
                    logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                    shutil.rmtree(path_to_delete, ignore_errors=True)
            time_rollout_delete = time.time() - time_rollout_delete
        if config.train.memory_profile and (training_progress.step == 2) and world_info.rank == 0:
            logger.info("Dumping memory snapshot.")
            pickle_path: str = config.train.memory_profile
            if not pickle_path.endswith(".pickle"):
                pickle_path += ".pickle"
            torch.cuda.memory._dump_snapshot(pickle_path)
            torch.cuda.memory._record_memory_history(enabled=False)

        if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
            logger.info(f"Saving checkpoint at step {training_progress.step}")
            save_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.path)

        if config.recompute_logprobs:
            reshard_module(model_for_logprob_only)
            tensor_offloaded_repository[training_progress.step] = copy_model_to_cpu(model)

        time_rollout_step = time.time() - time_start
        logger.success(f"Finished training step {training_progress.step} in {time_rollout_step:.2f}s")
        if world_info.rank == 0:
            time_metrics = {
                "step": training_progress.step,
                "perf/time_rollout_step": time_rollout_step,
                "perf/time_logprob": time_logprobs,
                "perf/time_data_loading": time_data_loading,
                "perf/time_data_preprocessing": time_data_and_logprob,
                "perf/time_rollout_delete": time_rollout_delete,
            }
            if time_rollout_ckpt is not None:
                time_metrics["perf/time_rollout_ckpt"] = time_rollout_ckpt
            if time_shardcast is not None:
                time_metrics["perf/time_shardcast"] = time_shardcast
            if time_rollout_delete is not None:
                time_metrics["perf/time_rollout_delete"] = time_rollout_delete

            monitor.log(time_metrics)

        if config.stop_after_steps is not None and training_progress.step >= config.stop_after_steps:
            break

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")


def main():
    train(parse_argv(TrainingConfig))


if __name__ == "__main__":
    main()
