import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

import asyncio
import json
import math
import time
from datetime import timedelta

# ruff: noqa: I001

from prime_rl.utils.act_offloading import maybe_activation_offloading
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from prime_rl import monitors
from prime_rl.configs.pretrain import PretrainConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.model import (
    forward,
    is_tt_moe_model,
    iter_moe_blocks,
    reshard_module,
    setup_model,
    setup_tokenizer,
)
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.models.init import multipliers, param_role
from prime_rl.trainer.models.prime_moe.moe import histogram_quantile_biases
from prime_rl.trainer.pretrain.parametrization import param_groups
from prime_rl.trainer.pretrain.bytes import token_byte_lengths
from prime_rl.trainer.pretrain.data import (
    TokenShards,
    check_eos,
    setup_dataloader,
    setup_train_dataset,
    source_tokens,
    setup_val_dataset,
    starts_with_bos,
)
from prime_rl.trainer.pretrain.suite import (
    length_label,
    prepare_multi_length_set,
    score_sets,
    set_metrics,
    scoring_head_active,
    summarize,
)
from prime_rl.trainer.pretrain.step import run_micro_batches
from prime_rl.trainer.pretrain.metrics import (
    DeadExperts,
    SpikeDetector,
    grad_norms_by_group,
    reduce_stats,
    routed_k_metrics,
    take_routing_counts,
)
from prime_rl.trainer.sft.data import get_dataset_state
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import (
    get_grad_accum_steps,
    get_dcp_group,
    GarbageCollection,
    clip_grad_norm_,
    get_ckpt_disk_metrics,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import resolve_resume_step
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit
from prime_rl.utils.weights import resolve_fqn


MOE_TRACE_INTERVAL = 32


@torch.no_grad()
def update_selection_biases(model: nn.Module, tokens_per_expert: Tensor, rate: float) -> None:
    """Aux-loss-free load balancing (https://arxiv.org/abs/2408.15664): move each layer's selection bias by
    ``rate`` towards underloaded experts, given the step's global ``[num_moe_layers, num_experts]`` counts.
    The update is mean-centred (as in Moonlight and torchtitan), which leaves the selection unchanged but keeps
    the biases from drifting together."""
    for moe, load in zip(iter_moe_blocks(model), tokens_per_expert):
        if moe.router.selection_bias is not None:
            delta = torch.sign(load.mean() - load)
            moe.router.selection_bias += rate * (delta - delta.mean())


@torch.no_grad()
def quantile_update_selection_biases(model: nn.Module, group: dist.ProcessGroup) -> None:
    """Quantile balancing: set each layer's selection bias to minus the step's balancing thresholds of its experts
    (``PrimeRouter.record_quantiles``), averaged over the micro-batches and ranks and mean-centred."""
    routers = [moe.router for moe in iter_moe_blocks(model)]
    sums = torch.stack([torch.cat([router.quantile_sum, router.quantile_count[None]]) for router in routers])
    dist.all_reduce(sums, group=group)
    for router, row in zip(routers, sums):
        thresholds = row[:-1] / row[-1]
        router.selection_bias.copy_(thresholds.mean() - thresholds)
        router.quantile_sum.zero_()
        router.quantile_count.zero_()


@torch.no_grad()
def histogram_quantile_update_selection_biases(model: nn.Module, group: dist.ProcessGroup) -> dict[str, float]:
    """Quantile balancing with the histogram estimator: sum the step's histograms over the ranks, set each layer's
    selection bias from them (``histogram_quantile_biases``) and use the step's global range of the required biases
    as the next step's grid. Returns the grid's mean bin width and the share of values in its edge bins, which also
    hold the values outside it."""
    routers = [moe.router for moe in iter_moe_blocks(model)]
    hists = torch.stack([router.quantile_hist for router in routers])
    dist.all_reduce(hists, group=group)
    ranges = torch.stack([torch.stack([-router.quantile_range[0], router.quantile_range[1]]) for router in routers])
    dist.all_reduce(ranges, op=dist.ReduceOp.MAX, group=group)
    bin_width = clamped = 0.0
    for router, hist, (neg_lo, hi) in zip(routers, hists, ranges):
        router.selection_bias.copy_(histogram_quantile_biases(hist, router.quantile_grid, router.top_k))
        bin_width += ((router.quantile_grid[1] - router.quantile_grid[0]) / hist.shape[1]).item()
        clamped += ((hist[:, 0].sum() + hist[:, -1].sum()) / hist.sum()).item()
        router.quantile_grid = torch.stack([-neg_lo, torch.maximum(hi, -neg_lo + 1e-6)])
        router.quantile_hist.zero_()
        router.quantile_range.copy_(torch.tensor([float("inf"), float("-inf")]))
    return {"router/qb/bin_width": bin_width / len(routers), "router/qb/edge_share": clamped / len(routers)}


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total, embedding (input embedding and LM head), non-embedding and active non-embedding parameter counts.
    Active counts the routed experts of each MoE layer at ``top_k / num_experts``, the share each token uses."""
    total = embedding = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param_role(resolve_fqn(model, name), param) in ("embedding", "output"):
            embedding += param.numel()
    inactive = 0
    for moe in iter_moe_blocks(model):
        experts = sum(param.numel() for param in moe.experts.parameters())
        inactive += experts - experts * moe.router.top_k // moe.router.num_experts
    non_embedding = total - embedding
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "active_non_embedding": non_embedding - inactive,
    }


@clean_exit
def train(config: PretrainConfig):
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting pretrain trainer in {world} (output_dir={config.run_dir})")

    asyncio.run(
        monitors.setup(
            producer="trainer",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.run_dir,
            run_config=config,
            overview_flavor="pretrain",
        )
    )

    heart = Heartbeat(config.heartbeat) if config.heartbeat is not None and world.rank == 0 else None

    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision(config.matmul_precision)

    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)

    grad_accum_steps = get_grad_accum_steps(
        config.data.batch_size, config.data.micro_batch_size, config.model.cp, world.world_size
    )

    ckpt_manager = setup_ckpt_manager(config.run_dir, config.ckpt, resume=config.resume)
    checkpoint_step = resolve_resume_step(config.resume, ckpt_manager.ckpt_dir)

    logger.info(f"Initializing model ({config.model})")
    # Weights come from the checkpoint when resuming and from the random init otherwise.
    model = setup_model(
        config.model,
        parallel_dims,
        loading_from_checkpoint_later=checkpoint_step is not None,
        init=config.model.init if checkpoint_step is None else None,
        parametrization=config.parametrization,
        compile_inside_full_checkpoint=True,
    )
    m_n, m_l = multipliers(config.parametrization, model)
    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)
    num_params = count_parameters(model)
    logger.info(
        f"Parameters: {num_params['active_non_embedding'] / 1e6:,.1f}M active non-embedding, "
        f"{num_params['non_embedding'] / 1e6:,.1f}M non-embedding, {num_params['embedding'] / 1e6:,.1f}M embedding, "
        f"{num_params['total'] / 1e6:,.1f}M total"
    )
    if checkpoint_step is None:
        asyncio.run(monitors.log({f"params/{name}": count for name, count in num_params.items()} | {"step": 0}, step=0))

    tokenizer = setup_tokenizer(config.tokenizer)

    optimizer, _ = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        model=model,
        param_groups=param_groups(model, config.optim, m_n, m_l, parallel_dims.ep_enabled),
    )

    # With ckpt.skip_scheduler, the schedule is rebuilt over the remaining steps.
    skip_scheduler = (config.ckpt or CheckpointConfig()).skip_scheduler
    scheduler_steps = config.max_steps - checkpoint_step if skip_scheduler and checkpoint_step else config.max_steps
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    train_dataset = setup_train_dataset(
        config.data, config.max_steps, len(tokenizer), tokenizer.eos_token_id, non_dp_size=config.model.cp
    )
    dataloader = setup_dataloader(train_dataset, config.data.micro_batch_size, config.data.num_workers)

    val_sources = {}
    if config.val is not None:
        val_sources = {name: TokenShards(source) for name, source in config.val.sources.items()}
        check_eos(val_sources, tokenizer.eos_token_id)
        byte_lengths = token_byte_lengths(tokenizer).to("cuda")
        # Whole micro-batches on every data-parallel rank, as every forward is a collective.
        windows_per_round = parallel_dims.get_mesh("dp").size() * config.data.micro_batch_size
        val_windows = {}
        for name, source in val_sources.items():
            val_windows[name] = (source.num_tokens - 1) // config.data.seq_len // windows_per_round * windows_per_round
            if val_windows[name] == 0:
                raise ValueError(f"Val source {name!r} has fewer than {windows_per_round} windows of data.seq_len")
            logger.info(f"Val source {name!r}: {val_windows[name] * config.data.seq_len:,} tokens")

    progress = Progress()
    restore_dataloader = checkpoint_step is not None and not (config.ckpt or CheckpointConfig()).skip_dataloader
    if checkpoint_step is not None:
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            None if skip_scheduler else scheduler,
            progress,
            dataloader=dataloader if restore_dataloader else None,
            path=config.resume.dir / "trainer" if config.resume.dir is not None else None,
        )
        progress.step += 1
        if skip_scheduler:
            # Loading the optimizer state replaces the param groups the scheduler was built on.
            scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
        logger.info(f"Resuming from step {checkpoint_step} (dataset_state={get_dataset_state(dataloader)})")

    # Step s trains on the global windows [(s - 1) * batch_size, s * batch_size). The tokens per source are counted
    # from the window sequence itself, which every data rank walks, so they are global without communication.
    next_window = 0
    source_totals = dict.fromkeys(train_dataset.names, 0)
    if restore_dataloader:
        next_window = checkpoint_step * config.data.batch_size
        # The loaded state counts the windows before its own position, which is near next_window.
        state = next(iter(get_dataset_state(dataloader).values()))
        source_totals = dict(state["tokens"])
        start, end = sorted((state["step"], next_window))
        sign = 1 if next_window > state["step"] else -1
        for name, tokens in source_tokens(train_dataset, start, end).items():
            source_totals[name] += sign * tokens

    # Create the iterator only after a potential resume: iter() forks the workers with the
    # dataset's current state.
    dataiter = iter(dataloader)

    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_size = parallel_dims.cp
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    is_moe_model = is_tt_moe_model(model)
    if is_moe_model and model.config.model_type != "prime_moe":
        raise ValueError(f"Pretraining MoE models use model_type 'prime_moe', got {model.config.model_type!r}")
    balance = config.model.moe.balance
    for moe in iter_moe_blocks(model):
        # Allocated uninitialized with the rest of the model.
        if moe.routed_k_counts is not None:
            moe.routed_k_counts.zero_()
        moe.router.quantile_sum.zero_()
        moe.router.quantile_count.zero_()
        moe.router.quantile_balance = balance.bias_update == "quantile"
        if moe.router.quantile_balance and balance.quantile_estimator == "histogram":
            device = moe.router.selection_bias.device
            moe.router.quantile_hist = torch.zeros(moe.router.num_experts, balance.quantile_bins, device=device)
            # The first step's grid; later steps use the previous step's global range.
            moe.router.quantile_grid = torch.tensor([-1.0, 1.0], device=device)
            moe.router.quantile_range = torch.tensor([float("inf"), float("-inf")], device=device)
    moes_with_seq_balance = list(iter_moe_blocks(model)) if balance.seq_aux_coeff > 0 else []
    for moe in moes_with_seq_balance:
        # A micro-batch is `micro_batch_size` packed rows concatenated into one sample; the sequences are the rows.
        moe.balance_rows = config.data.micro_batch_size
    bias_freeze_step = config.max_steps - round(balance.bias_freeze * config.max_steps)
    # Sum and count of the per-row balance of every MoE layer over the current step's micro-batches, reduced with
    # the step's statistics.
    seq_balance_sums: dict[str, torch.Tensor] = {}

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens and, in training, the MoE
        sequence-wise balance loss summed over tokens (each token carries its row's loss, so after the division
        by the token count it is the mean over rows)."""
        input_ids = micro_batch["input_ids"].to("cuda", non_blocking=True)
        position_ids = micro_batch["position_ids"].to("cuda", non_blocking=True)
        target_ids = micro_batch["target_ids"].to("cuda", non_blocking=True)
        loss_mask = micro_batch["loss_mask"].to("cuda", non_blocking=True)
        seq_lens = micro_batch["seq_lens"].to("cuda", non_blocking=True)
        if cp_enabled:
            input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, seq_lens=seq_lens)
            target_ids = shard_for_cp(target_ids, cp_rank=cp_rank, cp_world_size=cp_size)
            loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)
        labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)
        with maybe_activation_offloading(config.model.ac_offloading):
            out = forward(
                model, input_ids, position_ids, seq_lens=seq_lens, labels=labels, seq_lens_are_pre_shard=cp_enabled
            )
        aux_loss_sum = torch.zeros((), device="cuda")
        if torch.is_grad_enabled():
            row_len = input_ids.shape[1] // config.data.micro_batch_size
            for moe in moes_with_seq_balance:
                aux_loss_sum += balance.seq_aux_coeff * moe.seq_balance.sum() * row_len
                seq_balance_sums["balance"] = seq_balance_sums.get("balance", 0) + moe.seq_balance.detach().sum()
                seq_balance_sums["rows"] = seq_balance_sums.get("rows", 0) + moe.seq_balance.new_tensor(
                    moe.seq_balance.numel()
                )
        return out["loss"], loss_mask.sum(dtype=torch.int64), aux_loss_sum

    def finish_no_grad_forwards() -> None:
        """No-grad forwards leave the [lm_head, norm] FSDP group unsharded, and their routing must not count toward
        the next step's MoE statistics."""
        reshard_module(model)
        if is_moe_model:
            take_routing_counts(model)

    @torch.no_grad()
    def run_val(step: int) -> None:
        """Held-out loss, perplexity and bits per byte on the weights after ``step`` optimizer updates."""
        metrics = {}
        for name, source in val_sources.items():
            val_dataset = setup_val_dataset(source, config.data.seq_len, val_windows[name], config.model.cp)
            totals = torch.zeros(3, dtype=torch.float64, device="cuda")  # loss sum, tokens, bytes
            for micro_batch in setup_dataloader(val_dataset, config.data.micro_batch_size, num_workers=1):
                loss_sum, token_count, _ = compute_loss(micro_batch)
                # The bytes are counted on the unsharded targets, so every CP rank counts them once.
                num_bytes = byte_lengths[micro_batch["target_ids"].to("cuda")].sum() / cp_size
                totals += torch.stack([loss_sum.double(), token_count.double(), num_bytes.double()])
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_cp_group)
            loss_sum, num_tokens, num_bytes = totals.tolist()
            loss = loss_sum / num_tokens
            metrics[f"val/{name}/loss"] = loss
            metrics[f"val/{name}/ppl"] = math.exp(min(loss, 20))
            metrics[f"val/{name}/bpb"] = loss_sum / (math.log(2) * num_bytes)
        finish_no_grad_forwards()
        logger.success(f"Val | Step {step} | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(metrics.items())))
        asyncio.run(monitors.log({**metrics, "step": step}, step=step))

    eval_sets = []
    if config.evals is not None:
        start_time = time.perf_counter()
        bos = config.data.type == "tokens" and starts_with_bos(train_dataset.sources, tokenizer.bos_token_id)
        eval_sets = [
            prepare_multi_length_set(name, config.evals.data_dir, tokenizer, config.evals.seq_lens, bos)
            for name in config.evals.sets
        ]
        logger.info(f"Prepared {len(eval_sets)} eval sets (BOS: {bos}) in {time.perf_counter() - start_time:.0f}s")

    @torch.no_grad()
    def run_suite(step: int) -> None:
        """The eval suite on the weights after ``step`` optimizer updates: the packed rows of all sets are dealt
        over the ranks, the per-token results gathered on rank 0, which summarizes and logs them."""
        start_time = time.perf_counter()
        metrics = {}
        # Eager, also for compiled layers: the packed rows vary in length and would recompile at every forward.
        with scoring_head_active(model), torch.compiler.set_stance("force_eager"):
            scored = score_sets(model, eval_sets, config.evals.micro_batch_tokens, get_dcp_group())
            for eval_set, (logprobs, hits) in zip(eval_sets, scored or []):
                for seq_len, version in eval_set.versions.items():
                    index = eval_set.index[seq_len]
                    summary, arrays = summarize(version, [logprobs[i] for i in index], [hits[i] for i in index])
                    label = length_label(seq_len)
                    metrics |= set_metrics(eval_set.name, seq_len, summary)
                    if config.evals.save_arrays:
                        out = config.run_dir / "evals" / f"step_{step}"
                        out.mkdir(parents=True, exist_ok=True)
                        np.savez(out / f"{eval_set.name}@{label}.npz", **arrays)
                        (out / f"{eval_set.name}@{label}.json").write_text(json.dumps(summary, default=float))
        finish_no_grad_forwards()
        metrics["time/eval_suite"] = time.perf_counter() - start_time
        logger.success(f"Eval suite | Step {step} | {metrics['time/eval_suite']:.0f}s | {len(eval_sets)} sets")
        asyncio.run(monitors.log({**metrics, "step": step}, step=step))

    if checkpoint_step is None and config.val is not None and config.val.eval_on_start:
        run_val(step=0)
    if config.evals is not None and config.evals.eval_on_start:
        run_suite(step=checkpoint_step or 0)
    if checkpoint_step == config.max_steps:
        # The last step's checkpoint is written before its val and evals: resuming from it only runs those.
        logger.info(f"Resuming from the last step ({checkpoint_step}): running its val and evals only")
        if config.val is not None:
            run_val(checkpoint_step)
        if config.evals is not None and not config.evals.eval_on_start:
            run_suite(checkpoint_step)
        logger.success("Pretrain trainer finished")
        asyncio.run(monitors.finalize())
        return

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3
    dead_experts = DeadExperts()
    loss_spikes, grad_norm_spikes = SpikeDetector(), SpikeDetector()
    last_bias_moves = None
    # Per-expert selection bias, router-row norm and load of every MoE layer, every MOE_TRACE_INTERVAL steps.
    moe_trace: dict[str, list] = {"step": [], "bias": [], "router_norm": [], "load": []}
    logger.info(f"Starting training loop (max_steps={config.max_steps})")
    while True:
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = progress.step >= config.max_steps
        step_start_time = time.perf_counter()

        seq_balance_sums.clear()
        result = run_micro_batches(
            model,
            dataiter,
            compute_loss,
            grad_accum_steps,
            parallel_dims,
            extra_sums=seq_balance_sums,
        )
        forward_backward_time = time.perf_counter() - step_start_time
        moe_metrics = {}
        if is_moe_model:
            moe_metrics["expert_load/dead"] = dead_experts.update(result.tokens_per_expert.cpu())
            if result.sums:
                moe_metrics["loss/seq_balance"] = result.sums["balance"] / result.sums["rows"]
            biases = [
                moe.router.selection_bias for moe in iter_moe_blocks(model) if moe.router.selection_bias is not None
            ]
            updates_bias = balance.bias_update == "quantile" or balance.bias_update_rate > 0
            if updates_bias and progress.step <= bias_freeze_step:
                previous_biases = [bias.clone() for bias in biases]
                if balance.bias_update == "quantile" and balance.quantile_estimator == "histogram":
                    moe_metrics.update(histogram_quantile_update_selection_biases(model, dp_cp_group))
                elif balance.bias_update == "quantile":
                    quantile_update_selection_biases(model, dp_cp_group)
                else:
                    update_selection_biases(model, result.tokens_per_expert, balance.bias_update_rate)
                if biases:
                    # Share of experts whose bias moved in the opposite direction to the previous step's move.
                    moves = torch.stack([bias - previous for bias, previous in zip(biases, previous_biases)]).sign()
                    if last_bias_moves is not None:
                        moe_metrics["router/bias/flip_frac"] = (moves * last_bias_moves < 0).float().mean().item()
                    last_bias_moves = moves
            if biases:
                stacked = torch.stack(biases)
                moe_metrics["router/bias/absmax"] = stacked.abs().max().item()
                moe_metrics["router/bias/std"] = stacked.std(dim=1).mean().item()
                if progress.step % MOE_TRACE_INTERVAL == 0:
                    gates = [
                        moe.router.gate.weight
                        for moe in iter_moe_blocks(model)
                        if moe.router.selection_bias is not None
                    ]
                    gates = [gate.full_tensor() if isinstance(gate, DTensor) else gate for gate in gates]
                    moe_trace["step"].append(progress.step)
                    moe_trace["bias"].append(stacked.float().cpu().numpy())
                    moe_trace["router_norm"].append(
                        torch.stack([gate.detach().float().norm(dim=1) for gate in gates]).cpu().numpy()
                    )
                    moe_trace["load"].append(result.tokens_per_expert.float().cpu().numpy())
            moe_metrics |= routed_k_metrics(model, dp_cp_group)

        group_grad_norms = grad_norms_by_group(model)
        grad_norm = clip_grad_norm_(None, model, config.optim.max_norm) if config.optim.max_norm else None
        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # Peak memory of the step's forward, backward and optimizer passes, over all ranks.
        local_peak_memory = torch.tensor(torch.cuda.max_memory_reserved() / 1024**3, device="cuda")
        peak_memory = reduce_stats({"sum": local_peak_memory}, {"max": local_peak_memory}, dp_cp_group)
        peak_memory_metrics = {
            "perf/peak_memory/max": peak_memory["max"].item(),
            "perf/peak_memory/mean": peak_memory["sum"].item() / dist.get_world_size(dp_cp_group),
        }

        num_tokens = result.num_tokens
        progress.total_tokens += num_tokens
        step_tokens = source_tokens(train_dataset, next_window, next_window + config.data.batch_size)
        next_window += config.data.batch_size
        for name, tokens in step_tokens.items():
            source_totals[name] += tokens
        progress.total_samples = next_window

        # The last step's checkpoint is written before its val and evals, so a timeout there does not lose it.
        save_ckpt_time = 0.0
        disk_metrics = {}
        if config.ckpt and (is_last_step or (config.ckpt.interval and progress.step % config.ckpt.interval == 0)):
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
            ckpt_manager.maybe_clean()
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time
            disk_metrics = get_ckpt_disk_metrics(config.run_dir)

        eval_start_time = time.perf_counter()
        if config.val is not None and (is_last_step or progress.step % config.val.interval == 0):
            run_val(progress.step)
        if config.evals is not None and (is_last_step or progress.step % config.evals.interval == 0):
            run_suite(progress.step)
        eval_time = time.perf_counter() - eval_start_time

        step_time = time.perf_counter() - step_start_time - eval_time
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, step_time)
        mfu = perf_counter.get_step_mfu(num_tokens, step_time)
        step_message = f"Step {progress.step} | {step_time:.2f}s | Loss {result.loss:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += (
            f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}%"
            f" | Peak Mem. {peak_memory_metrics['perf/peak_memory/max']:.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        metrics = {
            "loss/mean": result.loss,
            "loss/perplexity": math.exp(min(result.loss, 20)),
            "loss/nan_count": result.nan_count,
            "loss/spike": loss_spikes.update(result.loss),
            "optim/lr": current_lr,
            **{f"optim/grad_norm/{group}": norm.item() for group, norm in group_grad_norms.items()},
            **{f"progress/{name}/step_tokens": tokens for name, tokens in step_tokens.items()},
            **{f"progress/{name}/num_tokens": tokens for name, tokens in source_totals.items()},
            **{
                f"progress/{name}/epochs": tokens / train_dataset.sources[name].num_tokens
                for name, tokens in source_totals.items()
            },
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "time/step": step_time,
            "time/forward_backward": forward_backward_time,
            "time/save_ckpt": save_ckpt_time,
            "time/evals": eval_time,
            **disk_metrics,
            **peak_memory_metrics,
            **result.metrics,
            **moe_metrics,
            "step": progress.step,
        }
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm.item()
            metrics["optim/grad_norm_spike"] = grad_norm_spikes.update(grad_norm.item())
        asyncio.run(monitors.log(metrics, step=progress.step))

        if heart is not None:
            heart.beat()
        if moe_trace["step"] and world.rank == 0 and (is_last_step or progress.step % (16 * MOE_TRACE_INTERVAL) == 0):
            np.savez(config.run_dir / "moe_trace.npz", **{name: np.array(values) for name, values in moe_trace.items()})
        if is_last_step:
            break
        progress.step += 1

    logger.success("Pretrain trainer finished")
    asyncio.run(monitors.finalize())


def main():
    set_proc_title("PretrainTrainer")
    train(cli(PretrainConfig))


if __name__ == "__main__":
    main()
