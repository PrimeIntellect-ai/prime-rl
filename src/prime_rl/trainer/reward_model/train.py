import time
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from renderers.base import create_renderer
from torchtitan.distributed.utils import clip_grad_norm_

import prime_rl._compat  # noqa: F401
from prime_rl.configs.reward_model import RewardModelConfig
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.reward_model.data import (
    load_bradley_terry_dataset,
    setup_dataloader,
    setup_dataset,
)
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import GarbageCollection, get_zero_gradient_ratio, setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.config import cli
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


def bradley_terry_losses(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Per-pair negative log-likelihood under the Bradley-Terry model."""
    return -F.logsigmoid(chosen_rewards.float() - rejected_rewards.float())


@clean_exit
def train(config: RewardModelConfig):
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting Bradley-Terry reward-model trainer in {world}")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=False)
    torch.set_float32_matmul_precision(config.matmul_precision)
    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)
    ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt)

    total_pairs = config.data.batch_size
    pairs_per_micro_step = world.world_size * config.data.micro_batch_size
    if total_pairs % pairs_per_micro_step:
        raise ValueError(
            f"data.batch_size ({total_pairs}) must be divisible by world_size * data.micro_batch_size "
            f"({pairs_per_micro_step})."
        )
    grad_accum_steps = total_pairs // pairs_per_micro_step

    logger.info(f"Initializing scalar reward model ({config.model})")
    model = setup_model(config.model, parallel_dims, task="reward_model")
    tokenizer = setup_tokenizer(config.tokenizer)
    renderer = create_renderer(tokenizer, config.renderer)

    optimizer = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        cpu_offload=config.model.optim_cpu_offload,
    )
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)

    dataset = setup_dataset(config.data, renderer)
    dataloader = setup_dataloader(dataset, config.data, tokenizer.pad_token_id)
    dataiter = iter(dataloader)
    progress = Progress(step=0)

    def save_checkpoint(step: int):
        progress.step = step
        progress.total_samples = dataset.step
        progress.total_tokens = sum(dataset.num_tokens.values())
        if ckpt_manager is not None and not config.ckpt.weights_only:
            ckpt_manager.save(step, model, [optimizer], scheduler, progress, dataloader=dataloader)
        if weight_ckpt_manager is not None:
            weight_ckpt_manager.save(step, model, tokenizer)
        if ckpt_manager is not None:
            ckpt_manager.maybe_clean()
        if weight_ckpt_manager is not None:
            weight_ckpt_manager.maybe_clean()

    val_raw_dataset = load_bradley_terry_dataset(config.val.data) if config.val is not None else None
    dp_group = parallel_dims.get_mesh("dp").get_group()

    def score_pairs(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = micro_batch["input_ids"].to("cuda")
        attention_mask = micro_batch["attention_mask"].to("cuda")
        position_ids = micro_batch["position_ids"].to("cuda")
        num_pairs = micro_batch["num_pairs"]
        pair_weights = micro_batch["pair_weights"].to("cuda")
        with maybe_activation_offloading(config.model.ac_offloading):
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            rewards = output.logits.squeeze(-1).float()
            chosen_rewards = rewards[:num_pairs]
            rejected_rewards = rewards[num_pairs:]
            losses = bradley_terry_losses(chosen_rewards, rejected_rewards)
        margin = chosen_rewards - rejected_rewards
        if pair_weights.shape != losses.shape:
            raise ValueError(
                f"Pair weights and losses must have the same shape, got {tuple(pair_weights.shape)} and "
                f"{tuple(losses.shape)}."
            )
        return (
            (losses * pair_weights).sum(),
            pair_weights.sum(),
            ((margin > 0) * pair_weights).sum(),
            (margin * pair_weights).sum(),
        )

    def run_validation(step: int):
        assert config.val is not None and val_raw_dataset is not None
        val_dataset = setup_dataset(
            config.val.data,
            renderer,
            raw_dataset=val_raw_dataset,
            max_epochs=1,
            pad_to_data_world_size=True,
        )
        val_loader = setup_dataloader(val_dataset, config.val.data, tokenizer.pad_token_id)
        totals = torch.zeros(4, dtype=torch.float64, device="cuda")
        with torch.no_grad():
            iterator = iter(val_loader)
            while True:
                batch = next(iterator, None)
                has_data = torch.tensor(batch is not None, dtype=torch.int32, device="cuda")
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if not has_data.item():
                    break
                loss_sum, pair_count, correct, margin_sum = score_pairs(batch)
                totals += torch.stack([loss_sum, pair_count, correct, margin_sum]).double()
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_group)
        loss, count, correct, margin = totals.tolist()
        metrics = {
            "val/loss": loss / count,
            "val/accuracy": correct / count,
            "val/reward_margin": margin / count,
            "step": step,
        }
        logger.success(
            f"Validation | Step {step} | Loss {metrics['val/loss']:.4f} | "
            f"Accuracy {metrics['val/accuracy']:.1%} | Margin {metrics['val/reward_margin']:.4f}"
        )
        monitor.log(metrics, step=step)

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None
    maybe_record_function = nullcontext
    max_steps = config.max_steps
    if max_steps is None:
        raise ValueError("Reward-model training requires max_steps.")

    if config.val is not None and config.val.eval_on_start:
        run_validation(0)

    logger.info(
        f"Training for {max_steps} optimizer steps with {total_pairs} pairs/step "
        f"({grad_accum_steps} accumulation steps)"
    )
    for step in range(1, max_steps + 1):
        step_start = time.perf_counter()
        if gc_handler is not None:
            gc_handler.run(step)
        optimizer.zero_grad()
        local_totals = torch.zeros(4, dtype=torch.float64, device="cuda")

        for _ in range(grad_accum_steps):
            micro_batch = next(dataiter)
            with maybe_record_function("forward"):
                loss_sum, pair_count, correct, margin_sum = score_pairs(micro_batch)
            (loss_sum / grad_accum_steps).backward()
            local_totals += torch.stack([loss_sum.detach(), pair_count, correct, margin_sum.detach()]).double()

        global_pair_count = local_totals[1].clone()
        dist.all_reduce(global_pair_count, op=dist.ReduceOp.SUM, group=dp_group)
        grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_pair_count.item()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(grad_scale)

        grad_norm = None
        if config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(
                model.parameters(), max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled
            )
        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        dist.all_reduce(local_totals, op=dist.ReduceOp.SUM, group=dp_group)
        loss, count, correct, margin = local_totals.tolist()
        metrics = {
            "loss/mean": loss / count,
            "train/accuracy": correct / count,
            "train/reward_margin": margin / count,
            "optim/lr": current_lr,
            "optim/zero_grad_ratio": get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate),
            "perf/peak_memory": torch.cuda.max_memory_reserved() / 1024**3,
            "step": step,
        }
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm.item()
        monitor.log(metrics, step=step)
        logger.success(
            f"Step {step} | {format_time(time.perf_counter() - step_start):>7} | "
            f"Loss {metrics['loss/mean']:.4f} | Accuracy {metrics['train/accuracy']:.1%} | "
            f"Margin {metrics['train/reward_margin']:.4f} | LR {current_lr:.2e}"
        )

        if config.val is not None and step % config.val.interval == 0:
            run_validation(step)
        if config.ckpt is not None and config.ckpt.interval and step < max_steps and step % config.ckpt.interval == 0:
            save_checkpoint(step)

    if config.ckpt is not None:
        logger.info("Writing final reward-model checkpoint")
        save_checkpoint(max_steps)
    logger.success("Reward-model training finished!")


def main():
    set_proc_title("RewardModelTrainer")
    train(cli(RewardModelConfig))


if __name__ == "__main__":
    main()
