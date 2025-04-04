import os
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy  # type: ignore
import wandb
import shardcast

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    # PrepareModuleInput,
    # SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from zeroband.models import AttnImpl, ModelName, ModelType, get_model_and_tokenizer
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.data import DataConfig, get_dataloader
from zeroband.training.loss import grpo_loss, selective_log_softmax, entropy_loss
from zeroband.training.lr_scheduler import get_scheduler
from zeroband.training.utils import PerfCounter, apply_ac_ckpt, clip_grad_norm_

from zeroband.logger import get_logger

from pydantic_config import BaseConfig, parse_argv
from jaxtyping import Float

from zeroband.training.world_info import WorldInfo, get_world_info

from pydantic import model_validator

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
    torch_compile: bool = True

    attn_impl: AttnImpl = "flex_attention"

    tp: int = 1 # Tensor parallel shard the model
    more_tp: bool = False # Also TP shard the embedding and lm_head


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
    grpo_epsilon: float = 0.2
    entropy_loss_coeff: float = 0.001

    on_policy_log_prob: bool = False
    max_async_level: int = 2  # the amount of rollout checkpoints to keep

    masked_mean_axis: int | None = None  # the axis to compute the mean of the masked values

    @model_validator(mode="after")
    def check_ckpt_interval(self):
        if self.ckpt.interval is not None:
            assert self.ckpt.interval % self.optim.step_per_rollout == 0, "ckpt.interval must be divisible by train.step_per_rollout"
        return self


def get_gradient_accumulation_steps(batch_size: int, micro_bs: int, data_workers: int, dp_world_size: int) -> int:
    assert batch_size % dp_world_size == 0
    batch_size = batch_size // dp_world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size // micro_bs


def apply_tp(model: ModelType, config: TrainConfig, device_mesh: DeviceMesh):
    # Qwen2 only, no sequence parallel

    if config.more_tp:
        parallelize_module(
            model.model,
            device_mesh,
            {
                "embed_tokens": RowwiseParallel(
                    input_layouts=Replicate(),
                    use_local_output=True,
                ),
            }
        )

        parallelize_module(
            model,
            device_mesh,
            {
                "lm_head": ColwiseParallel(
                    output_layouts=Replicate(),
                    use_local_output=True,
                ),
            },
        )

    parallelize_module(
        model.model,
        device_mesh,
        {
            'layers.*.self_attn.q_proj':   ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=-1), use_local_output=True),
            'layers.*.self_attn.k_proj':   ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=-1), use_local_output=True),
            'layers.*.self_attn.v_proj':   ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=-1), use_local_output=True),
            'layers.*.self_attn.o_proj':   RowwiseParallel(input_layouts=Shard(dim=-1), output_layouts=Replicate(), use_local_output=True),
            'layers.*.mlp.gate_proj':      ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=-1), use_local_output=True),
            'layers.*.mlp.up_proj':        ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=-1), use_local_output=True),
            'layers.*.mlp.down_proj':      RowwiseParallel(input_layouts=Shard(dim=-1), output_layouts=Replicate(), use_local_output=True),
        }
    )


def apply_fsdp(model: ModelType, reshard_after_forward: bool, device_mesh: DeviceMesh | None):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward, mesh=device_mesh)
    fully_shard(model.get_input_embeddings(), mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, mesh=device_mesh)
    fully_shard(model.get_output_embeddings(), mp_policy=mp_policy, reshard_after_forward=False, mesh=device_mesh)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, mesh=device_mesh)


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
        torch._logging.set_logs(dynamo=logging.CRITICAL)  # type: ignore (silence flex attn error)
        torch_log.setLevel(logging.CRITICAL)  #
        logging.getLogger("transformers.modeling_utils").setLevel(logging.CRITICAL) # Silence dtype and device warnings

    logger = get_logger()
    world_info = get_world_info()

    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        origin_data_dir = os.environ.get("SHARDCAST_OUTPUT_DIR", "./origin_data")
        shardcast.initialize(origin_data_dir, max_distribution_folders=config.max_async_level)

    model, tokenizer = get_model_and_tokenizer(config.name_model, config.train.attn_impl)

    # Create device mesh and apply fsdp/tp.
    assert world_info.world_size % config.train.tp == 0, "world size must be divisible by tp"
    tp_world_size = config.train.tp
    dp_world_size = world_info.world_size // config.train.tp

    world_mesh: DeviceMesh = init_device_mesh("cuda", mesh_shape=(dp_world_size, tp_world_size), mesh_dim_names=("fsdp", "tp"))
    logger.info(f"World device mesh: {world_mesh}")

    tp_mesh = world_mesh["tp"]
    tp_rank = tp_mesh.get_local_rank() if tp_world_size > 1 else 0
    logger.info(f"tp_rank: {tp_rank}, tp_mesh: {tp_mesh}")
    if tp_world_size > 1:
        apply_tp(model, config.train, device_mesh=world_mesh["tp"])

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    dp_mesh = world_mesh["fsdp"]
    dp_rank = dp_mesh.get_local_rank() if dp_world_size > 1 else 0
    logger.info(f"dp_rank: {dp_rank}, dp_mesh: {dp_mesh}")
    apply_fsdp(model, config.train.reshard_after_forward, device_mesh=dp_mesh) # Always enabled for Mixed Precision

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=config.optim.optim.lr,weight_decay=config.optim.optim.weight_decay,betas=(config.optim.optim.betas1, config.optim.optim.betas2), foreach=False)  # fmt: skip

    scheduler = get_scheduler(sched_type=config.optim.sched_type,optimizer=optimizer,num_warmup_steps=config.optim.warmup_steps,num_stable_steps=config.optim.stable_steps,num_training_steps=config.optim.total_steps)  # fmt: skip

    training_progress = TrainingProgress(total_tokens=0, step=0)
    gradient_accumulation_steps = get_gradient_accumulation_steps(
        config.optim.batch_size, config.train.micro_bs, config.data.num_workers, dp_world_size
    )

    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        micro_batch_size=config.train.micro_bs,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
        step_count_init=training_progress.step // config.optim.step_per_rollout,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    train_dataloader_iterator = iter(train_dataloader)

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

    perf_counter = PerfCounter(window_size=10, model=model, seq_len=config.data.seq_length, tp_world_size=tp_world_size)

    previous_ckpt_rollout = []

    while True:
        time_start = time.time()

        # here we want to pre-compute the logprobs with the model before update
        with torch.no_grad():
            if config.on_policy_log_prob:
                data = []

                for rollout_step in range(config.optim.step_per_rollout):
                    for grad_acc_step in range(gradient_accumulation_steps):
                        batch = next(train_dataloader_iterator)
                        input_ids = batch["input_ids"].to("cuda")

                        logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids).logits.contiguous()

                        input_ids = input_ids[:, 1:]
                        logits = logits[:, :-1, :] / config.temperature

                        per_token_logps = selective_log_softmax(logits, input_ids)
                        batch["logprobs"] = per_token_logps.to("cpu")

                        del input_ids, logits, per_token_logps
                        data.append(batch)

                logprobs_aware_iterator = iter(data)
            else:
                logprobs_aware_iterator = train_dataloader_iterator

        for rollout_step in range(config.optim.step_per_rollout):
            loss_batch = 0
            pg_loss_batch = 0
            entropy_loss_batch = 0
            clip_ratio_batch = 0
            seq_lens_batch = 0
            sample_reward_batch = 0
            clip_seq_lens = 0
            rewards_sum = torch.tensor(0.0, device="cuda") # On GPU for allreduce sum
            rewards_token_count = torch.tensor(0.0, device="cuda")

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            for _grad_acc_step in range(gradient_accumulation_steps):
                # Load args
                batch = next(logprobs_aware_iterator)
                input_ids = batch["input_ids"].to("cuda")
                loss_mask = batch["loss_mask"]

                rewards = batch["rewards"][loss_mask.bool()]
                rewards_sum += rewards.sum().to("cuda")

                rewards_token_count += rewards.numel()
                seq_lens_batch += batch["seq_lens"].float().mean().to('cuda') / gradient_accumulation_steps
                clip_seq_lens += (
                    (batch["seq_lens"] >= config.data.seq_length).sum() / batch["seq_lens"].shape[0] / gradient_accumulation_steps
                ).to('cuda')

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
                    config.grpo_epsilon,
                    config.masked_mean_axis,
                )
                entropy = entropy_loss(logits, loss_mask, config.temperature, config.masked_mean_axis)

                loss = pg_loss - config.entropy_loss_coeff * entropy
                loss = loss / gradient_accumulation_steps
                clip_ratio = clip_ratio / gradient_accumulation_steps

                sample_reward_batch += (batch["rewards"][:, 0].sum() / batch["rewards"].shape[0] / gradient_accumulation_steps).to("cuda")

                del batch, input_ids, logits, advantages, loss_mask, original_logprobs

                # Backward
                loss.backward()
                loss_batch += loss.detach().clone()
                pg_loss_batch += (pg_loss / gradient_accumulation_steps).detach().clone()
                entropy_loss_batch += (entropy / gradient_accumulation_steps).detach().clone()
                clip_ratio_batch += clip_ratio.detach().clone()
                del loss, clip_ratio, pg_loss, entropy

            dp_group = dp_mesh.get_group("fsdp") if dp_world_size > 1 else None
            if dp_group is not None:
                dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=pg_loss_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=entropy_loss_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=clip_ratio_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=seq_lens_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=clip_seq_lens, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=sample_reward_batch, op=dist.ReduceOp.AVG, group=dp_group)
                dist.all_reduce(tensor=rewards_sum, op=dist.ReduceOp.SUM, group=dp_group)
                dist.all_reduce(tensor=rewards_token_count, op=dist.ReduceOp.SUM, group=dp_group)
            average_rewards = rewards_sum / rewards_token_count

            grad_norm = clip_grad_norm_(model.parameters(), 1.0, dp_mesh=dp_mesh, tp_mesh=tp_mesh)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in optimizer.param_groups][0]

            # syncing loss across all data parallel rank within a nodes
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
            }

            log = f"step: {training_progress.step}, rollout_step: {training_progress.step // config.optim.step_per_rollout}, loss: {loss_batch.item():.4f}, average_rewards: {average_rewards.item():.4f}"

            del loss_batch, average_rewards, grad_norm, pg_loss_batch, entropy_loss_batch

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = perf_counter.get_mfu()
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

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

        logger.info(f"Finished rollout {rollout_step} step {training_progress.step}")
        if world_info.rank == 0 and config.wandb:
            wandb.log({"rollout_step": rollout_step, "step": training_progress.step, "time_rollout_step": time.time() - time_start})

        if training_progress.step >= config.optim.total_steps:
            break

    if prefetcher is not None:
        prefetcher.shutdown()

    logger.info("Training finished, exiting ...")
    logger.info(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    train(Config(**parse_argv()))
