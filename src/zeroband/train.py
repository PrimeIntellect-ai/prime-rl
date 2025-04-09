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
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.data import BatchOutput, DataConfig, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.loss import grpo_loss, selective_log_softmax, entropy_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import PerfCounter, apply_ac_ckpt

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
import logging


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

    step_per_rollout: int = 1


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    memory_profile: str | None = None
    torch_compile: bool = False  #  disabling torch compile because its too unstable for RL
    liger_qwen: bool = False
    ignore_zero_advantages: bool = False  # don't use in local setup

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

    on_policy_log_prob: bool = True
    max_async_level: int = 2  # the amount of rollout checkpoints to keep

    masked_mean_axis: int | None = None  # the axis to compute the mean of the masked values

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

    print(f"batch_size: {batch_size}, micro_bs: {micro_bs}, data_workers: {data_workers}")
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

    if world_info.local_rank >= len(gpus_ids):
        raise ValueError(f"Local rank {world_info.local_rank} is greater than the number of available GPUs ({len(gpus_ids)})")

    return gpus_ids[world_info.local_rank]


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

    training_progress = TrainingProgress(total_tokens=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    if config.ckpt.resume:
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
        ignore_zero_advantages=config.train.ignore_zero_advantages,
    )
    train_dataloader_iterator = iter(train_dataloader)

    previous_ckpt_rollout = []

    while True:
        time_start = time.time()

        total_time_data_loading = 0

        # here we want to pre-compute the logprobs with the model before update
        with torch.no_grad():
            if config.on_policy_log_prob:
                data: list[tuple[list[BatchOutput], int]] = []

                for rollout_step in range(config.optim.step_per_rollout):
                    batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                    logger.info(f"{len(batch_rollout)}=")

                    time_0 = time.time()

                    batch_packed, num_grad_acc_steps = packed_batch(
                        batch_rollout, config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs
                    )
                    time_1 = time.time()
                    logger.info(f"time to pack batch: {time_1 - time_0:.2f} seconds")

                    logger.info(f"policy log prob rollout_step: {rollout_step} num_grad_acc_steps: {num_grad_acc_steps}")
                    data_per_rollout = []
                    for grad_acc_step in range(num_grad_acc_steps):
                        time_data_loading = time.time()

                        batch = batch_packed[grad_acc_step]

                        time_data_loading = time.time() - time_data_loading
                        total_time_data_loading += time_data_loading

                        input_ids = batch["input_ids"].to("cuda")
                        logger.info(f"policy log prob grad_acc_step: {grad_acc_step} batch: {input_ids.shape}")

                        logits: Float[torch.Tensor, "batch seq vocab"] = model(
                            input_ids=input_ids, position_ids=batch["position_ids"]
                        ).logits.contiguous()

                        input_ids = input_ids[:, 1:]
                        logits = logits[:, :-1, :] / config.temperature

                        per_token_logps = selective_log_softmax(logits, input_ids)
                        batch["logprobs"] = per_token_logps.to("cpu")

                        del logits, per_token_logps
                        data_per_rollout.append(batch)

                    data.append((data_per_rollout, num_grad_acc_steps))

                logprobs_aware_iterator = iter(data)

                time_logprob = time.time() - time_start
                logger.info(f"Time to compute logprobs: {time_logprob:.2f} seconds")
            else:
                data = []
                for rollout_step in range(config.optim.step_per_rollout):
                    batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                    data.append(packed_batch(batch_rollout, config.data.seq_length, tokenizer.pad_token_id, 1))
                logprobs_aware_iterator = iter(data)

        for rollout_step in range(config.optim.step_per_rollout):
            loss_batch = torch.tensor(0.0, device="cuda")
            pg_loss_batch = torch.tensor(0.0, device="cuda")
            entropy_loss_batch = torch.tensor(0.0, device="cuda")
            clip_ratio_batch = torch.tensor(0.0, device="cuda")
            seq_lens_batch = torch.tensor(0.0)
            clip_seq_lens = torch.tensor(0.0)
            sample_reward_batch = torch.tensor(0.0)

            rewards_sum = torch.tensor(0.0)
            rewards_token_count = torch.tensor(0.0)

            task_rewards_sum = torch.tensor(0.0)
            task_rewards_token_count = torch.tensor(0.0)

            length_pen_sum = torch.tensor(0.0)
            length_pen_token_count = torch.tensor(0.0)

            sample_task_reward_batch = torch.tensor(0.0)
            sample_length_penalty_batch = torch.tensor(0.0)

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            data_per_rollout, num_grad_acc_steps = next(logprobs_aware_iterator)

            logger.info(f"rollout_step: {rollout_step} num_grad_acc_steps: {num_grad_acc_steps}")

            for grad_acc_step in range(num_grad_acc_steps):
                logger.info(f"grad_acc_step: {grad_acc_step}/{num_grad_acc_steps}")
                batch = data_per_rollout[grad_acc_step]
                input_ids = batch["input_ids"].to("cuda")
                loss_mask = batch["loss_mask"]

                rewards_this = batch["rewards"][loss_mask.bool()]
                rewards_sum += rewards_this.sum()
                rewards_token_count += rewards_this.numel()

                sample_reward_batch += batch["rewards"][:, 0].sum() / batch["rewards"].shape[0] / num_grad_acc_steps

                task_this = batch["task_rewards"][loss_mask.bool()]
                task_rewards_sum += task_this.sum()
                task_rewards_token_count += task_this.numel()

                len_pen_this = batch["length_penalties"][loss_mask.bool()]
                length_pen_sum += len_pen_this.sum()
                length_pen_token_count += len_pen_this.numel()

                sample_task_reward_batch += batch["task_rewards"][:, 0].sum() / batch["task_rewards"].shape[0] / num_grad_acc_steps

                sample_length_penalty_batch += (
                    batch["length_penalties"][:, 0].sum() / batch["length_penalties"].shape[0] / num_grad_acc_steps
                )

                seq_lens_batch += batch["seq_lens"].float().mean() / num_grad_acc_steps
                clip_seq_lens += (batch["seq_lens"] >= config.data.seq_length).sum() / batch["seq_lens"].shape[0] / num_grad_acc_steps

                # Forward
                logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

                # Gather args for grpo loss
                advantages = batch["advantages"].to("cuda")
                loss_mask = loss_mask.to("cuda")
                original_logprobs = batch["logprobs"].to("cuda")
                if not config.on_policy_log_prob:
                    original_logprobs = original_logprobs[:, 1:]

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
                )
                entropy = entropy_loss(logits, loss_mask, config.temperature, config.masked_mean_axis)

                loss = pg_loss - config.entropy_loss_coeff * entropy
                loss = loss / num_grad_acc_steps
                clip_ratio = clip_ratio / num_grad_acc_steps

                del batch, logits, input_ids, advantages, loss_mask, original_logprobs

                # Backward
                loss.backward()
                loss_batch += loss.detach().clone()
                pg_loss_batch += (pg_loss / num_grad_acc_steps).detach().clone()
                entropy_loss_batch += (entropy / num_grad_acc_steps).detach().clone()
                clip_ratio_batch += clip_ratio.detach().clone()
                del loss, clip_ratio, pg_loss, entropy

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=pg_loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=entropy_loss_batch, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=clip_ratio_batch, op=dist.ReduceOp.AVG)

            seq_lens_batch = seq_lens_batch / world_info.world_size
            dist.all_reduce(tensor=seq_lens_batch, op=dist.ReduceOp.SUM)

            clip_seq_lens = clip_seq_lens / world_info.world_size
            dist.all_reduce(tensor=clip_seq_lens, op=dist.ReduceOp.SUM)

            sample_reward_batch = sample_reward_batch / world_info.world_size
            dist.all_reduce(tensor=sample_reward_batch, op=dist.ReduceOp.SUM)

            dist.all_reduce(rewards_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(rewards_token_count, op=dist.ReduceOp.SUM)
            average_rewards = rewards_sum / rewards_token_count

            dist.all_reduce(task_rewards_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(task_rewards_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(length_pen_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(length_pen_token_count, op=dist.ReduceOp.SUM)

            dist.all_reduce(sample_task_reward_batch, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_length_penalty_batch, op=dist.ReduceOp.SUM)

            average_task_rewards = task_rewards_sum / task_rewards_token_count
            average_length_penalties = length_pen_sum / length_pen_token_count

            sample_task_rewards = sample_task_reward_batch / world_info.world_size
            sample_length_penalties = sample_length_penalty_batch / world_info.world_size

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore (is a dtensor)
            if isinstance(grad_norm, torch.distributed.tensor.DTensor):
                grad_norm.full_tensor()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            training_progress.step += 1
            inner_lr = [group["lr"] for group in optimizer.param_groups][0]

            new_tokens = config.data.seq_length * config.optim.batch_size
            perf_counter.count_tokens(new_tokens)
            training_progress.total_tokens += new_tokens

            padding_proportion = (config.data.seq_length - seq_lens_batch.item() - 1) / config.data.seq_length

            metrics = {
                "Loss": loss_batch.item(),
                "pg_loss": pg_loss_batch.item(),
                "entropy_loss": entropy_loss_batch.item(),
                "step": training_progress.step,
                "rollout_step": rollout_step,
                "seq_lens": seq_lens_batch.item(),
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
                "time": time.time(),
                "grad_norm": grad_norm.item(),
                "average_rewards": average_rewards.item(),
                "clip_ratio": clip_ratio_batch.item(),
                "padding_proportion": padding_proportion,
                "sample_reward": sample_reward_batch.item(),
                "clip_seq_lens": clip_seq_lens.item(),
                "average_task_rewards": average_task_rewards.item(),
                "average_length_penalties": average_length_penalties.item(),
                "sample_task_rewards": sample_task_rewards.item(),
                "sample_length_penalties": sample_length_penalties.item(),
            }

            log = (
                f"step: {training_progress.step}, "
                f"rollout_step: {training_progress.step // config.optim.step_per_rollout}, "
                f"loss: {loss_batch.item():.4f}, "
                f"average_rewards: {average_rewards.item():.4f}, "
                f"clip_ratio: {clip_ratio_batch.item():.4f}, "
                f"average_task_rewards: {average_task_rewards.item():.4f}, "
                f"average_length_penalties: {average_length_penalties.item():.4f}"
            )

            del loss_batch, average_rewards, grad_norm, pg_loss_batch, entropy_loss_batch

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
                metrics["tokens_per_second"] = tokens_per_second
                metrics["tokens_per_second_per_gpu"] = tokens_per_second_per_gpu

                metrics["mfu"] = perf_counter.get_mfu()

                log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {metrics['mfu']:.2f}"

            if world_info.rank == 0 and config.wandb:
                wandb.log(metrics)

            logger.info(log)

            # Lets do this first so that clients can start downloading as soon as possible
            if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
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
            if config.on_policy_log_prob:
                new_metrics["time_logprob"] = time_logprob
                new_metrics["time_data_loading"] = total_time_data_loading

            wandb.log(new_metrics)

        if training_progress.step >= config.optim.total_steps:
            break

    if prefetcher is not None:
        prefetcher.shutdown()

    logger.info("Training finished, exiting ...")
    logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    train(Config(**parse_argv()))
