import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

import math
import time
import asyncio
from contextlib import nullcontext
from datetime import timedelta
from functools import partial

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.utils.act_offloading import maybe_activation_offloading
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.configs.sft import PackedChoiceDataConfig, SFTConfig, SFTDataConfig
from prime_rl.configs.trainer import CheckpointConfig, validate_scheduler
from prime_rl.transports.weights import prune_broadcasts_beyond, setup_weight_sender
from prime_rl.utils.cp import setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.trainer.lora import get_lora_state
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    forward,
    get_full_offload_dtype_policy,
    get_global_moe_stats,
    get_load_balance_stats,
    is_tt_moe_model,
    no_grad_forwards,
    setup_processor,
    setup_tokenizer,
    setup_model,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.sft.choice import choice_loss_inputs
from prime_rl.trainer.sft.choice_eval import (
    ChoiceEvalSite,
    choice_eval_step_dir,
    choice_row_ids,
    due_choice_evals,
    score_choice_predictions,
    unscored_choice_evals,
)
from prime_rl.trainer.sft.data import (
    PackedChoiceEvalBatch,
    get_dataset_progress,
    get_dataset_state,
    load_sft_dataset,
    packed_choice_max_steps,
    packed_choice_mix_components,
    setup_choice_eval_sets,
    setup_dataloader,
    setup_dataset,
)
from prime_rl.trainer.utils import (
    GarbageCollection,
    MemoryProfiler,
    begin_backward,
    clip_grad_norm_,
    finish_backward,
    get_ckpt_disk_metrics,
    prepare_gradient_offload,
    print_sample,
    scale_gradients_,
    setup_full_cpu_optimizer_offload,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl import monitors
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, import_object
import torch.distributed as dist


@clean_exit
def train(config: SFTConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting SFT trainer in {world} (output_dir={config.run_dir})")

    # Packed-choice data trains the [loss] function over choice logits; the step normalizer is its weight sum
    choice_loss_fn = None
    if isinstance(config.data, PackedChoiceDataConfig):
        assert config.loss is not None
        mix_components = packed_choice_mix_components(config.data, config.val.data if config.val else None)
        choice_loss_fn = partial(
            import_object(config.loss.import_path), mix_components=mix_components, **config.loss.kwargs
        )
        config.max_steps = packed_choice_max_steps(config.data, config.max_steps)
        validate_scheduler(config.scheduler, config.max_steps)
    normalizer_dtype = torch.int64 if choice_loss_fn is None else torch.float64

    # Setup the monitors
    asyncio.run(
        monitors.setup(
            producer="trainer",
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.run_dir,
            run_config=config,
            eval_env_names=[source.resolved_name for source in config.eval.source] if config.eval else [],
            overview_flavor="sft",
        )
    )

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.rank == 0:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat)

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.full_offload is not None,
    )
    if config.model.full_offload is not None:
        setup_full_cpu_optimizer_offload(config.model.full_offload)
    # Configurable to support ROCm/AMD GPUs where reduced precision
    # matmul corrupts softmax over large vocabularies. Override via config
    # (e.g. matmul_precision = "highest") on ROCm.
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Resolve ep="auto" to a concrete integer before creating parallel dims
    resolve_ep(config.model)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.data.seq_len)

    total_micro_batches = config.data.batch_size * config.model.cp
    micro_batches_per_step = world.world_size * config.data.micro_batch_size
    assert total_micro_batches % micro_batches_per_step == 0, (
        f"batch_size * cp ({total_micro_batches}) must be divisible by "
        f"world_size * micro_batch_size ({micro_batches_per_step})"
    )
    grad_accum_steps = total_micro_batches // micro_batches_per_step

    if parallel_dims.cp_enabled:
        assert config.data.seq_len % parallel_dims.cp == 0, "Sequence length must be divisible by CP degree"

    # Set up checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.run_dir, config.ckpt, resume=config.resume)

    checkpoint_step = None
    if config.resume is not None:
        if config.resume.dir is not None:
            checkpoint_step = config.resume.dir_step
        else:
            checkpoint_step = config.resume.step
            if checkpoint_step is None:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)

    if config.model.lora is not None:
        get_lora_state().reset_adapter_parameters()

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)
    processor = setup_processor(config.model)
    if config.model.vlm is not None and processor is None:
        raise ValueError(f"[model.vlm] is set but no multimodal processor could be loaded for {config.model.name!r}")

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer, gradient_manager = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        cpu_offload=config.model.optim_cpu_offload,
        full_offload_config=config.model.full_offload,
        model=model,
        full_offload_dtype_policy=(
            get_full_offload_dtype_policy(model, config.model) if config.model.full_offload is not None else None
        ),
    )
    assert choice_loss_fn is None or gradient_manager is None, "Packed choice data requires model.full_offload = None"

    # Set up the learning rate scheduler
    # skip_scheduler rebuilds a fresh schedule over the remaining steps: size it from the
    # resolved checkpoint step (bare --resume and --resume.dir carry no explicit step).
    scheduler_steps = (
        config.max_steps - checkpoint_step
        if config.max_steps is not None and (config.ckpt and config.ckpt.skip_scheduler and checkpoint_step is not None)
        else config.max_steps
    )
    logger.info(f"Initializing scheduler with {scheduler_steps} steps ({config.scheduler})")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    # Set up the dataset and dataloader
    logger.info(f"Initializing data ({config.data})")
    multimodal = config.model.vlm is not None
    dataset = setup_dataset(
        tokenizer,
        config.data,
        config.model.cp,
        renderer_config=config.renderer,
        processor=processor,
        multimodal=multimodal,
    )
    dataloader = setup_dataloader(dataset, config.data)

    val_raw_dataset = None
    if config.val is not None and isinstance(config.val.data, SFTDataConfig):
        logger.info(f"Loading validation dataset ({config.val.data})")
        val_raw_dataset = load_sft_dataset(config.val.data)

    # Every rank imports the scoring function and checks every eval bin, so a bad config fails all ranks at startup
    choice_eval_fn = None
    choice_eval_sets = {}
    if config.choice_eval is not None:
        logger.info(f"Loading choice-eval sets ({config.choice_eval})")
        if config.choice_eval.import_path is not None:
            choice_eval_fn = partial(import_object(config.choice_eval.import_path), **config.choice_eval.kwargs)
        choice_eval_sets = setup_choice_eval_sets(
            config.choice_eval, seq_len=config.data.seq_len, non_dp_size=config.model.cp
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()

    if checkpoint_step is not None:
        resume_dir = config.resume.dir if config.resume else None
        skip = config.ckpt or CheckpointConfig()
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            scheduler if not skip.skip_scheduler else None,
            progress if not skip.skip_progress else None,
            dataloader=dataloader if not skip.skip_dataloader else None,
            path=resume_dir / "trainer" if resume_dir is not None else None,
        )
        # The checkpoint finished step ``checkpoint_step``; resume training at the next step.
        if not skip.skip_progress:
            progress.step += 1
        # This redundant setup is necessary because loading the optimizer's state has side effects on the scheduler state dict
        if skip.skip_scheduler:
            scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)
        logger.info(
            f"Resuming from step {checkpoint_step} (total_tokens={progress.total_tokens}, "
            f"total_samples={progress.total_samples}, dataset_state={get_dataset_state(dataloader)})"
        )
    else:
        logger.info("Starting from scratch")

    # A checkpoint at max_steps finished training: only the final choice evals can still be missing
    resumed_past_end = config.max_steps is not None and progress.step > config.max_steps
    if resumed_past_end:
        logger.info(f"Checkpoint step {checkpoint_step} reached max_steps = {config.max_steps}: skipping training")

    # Create the iterator only after a potential resume: iter() forks workers with a
    # copy of the dataset's *current* state, so a later load_state_dict never reaches
    # an already-running worker (the run silently restarts the data from the beginning
    # and re-saves the stale position).
    dataiter = iter(dataloader)

    cp_enabled = parallel_dims.cp_enabled
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
    cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None
    cp_size = parallel_dims.cp

    def shard_micro_batch(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        per_position: list[torch.Tensor],
        defer_cp_to_model: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], bool]:
        """This rank's context-parallel shard of a micro batch's inputs and ``[1, seq, ...]`` per-position
        tensors, and whether ``seq_lens`` still describes the unsharded sequence. Sizes LoRA to the local tokens."""
        seq_lens_are_pre_shard = False
        if cp_enabled:
            # CP requires the sequence length to be divisible by cp_size. CatDataset
            # pads every pack to seq_len; shard_for_cp raises on violations.
            if not defer_cp_to_model:
                input_ids, position_ids = setup_cp_params(
                    input_ids,
                    position_ids,
                    cp_rank,
                    cp_size,
                    cp_group,
                    seq_lens=seq_lens,
                    cp_style=config.model.cp_style,
                )
            seq_lens_are_pre_shard = True
            per_position = [shard_for_cp(tensor, cp_rank=cp_rank, cp_world_size=cp_size) for tensor in per_position]

        if config.model.lora is not None:
            set_lora_num_tokens(torch.full((1,), per_position[0].shape[1], dtype=torch.int32, device="cuda"))
        return input_ids, position_ids, per_position, seq_lens_are_pre_shard

    def compute_loss(micro_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (loss_sum, token_count) over unmasked tokens, or (loss_sum, weight_sum)
        from the choice loss."""
        input_ids = micro_batch["input_ids"].to("cuda", non_blocking=True)
        position_ids = micro_batch["position_ids"].to("cuda", non_blocking=True)
        seq_lens = micro_batch["seq_lens"].to("cuda", non_blocking=True)
        mm_kwargs = micro_batch.get("mm_kwargs")
        if mm_kwargs is not None:
            mm_kwargs = {key: value.to("cuda", non_blocking=True) for key, value in mm_kwargs.items()}
        mm_type_ids = micro_batch.get("mm_token_type_ids")
        if mm_type_ids is not None:
            mm_type_ids = mm_type_ids.to("cuda", non_blocking=True)

        per_position_keys = ["target_ids", "loss_mask"]
        if choice_loss_fn is not None:
            per_position_keys += ["choice_ids", "choice_targets", "choice_weights", "choice_mix_weights"]
        input_ids, position_ids, per_position, seq_lens_are_pre_shard = shard_micro_batch(
            input_ids,
            position_ids,
            seq_lens,
            [micro_batch[key].to("cuda", non_blocking=True) for key in per_position_keys],
            defer_cp_to_model=(
                mm_kwargs is not None and "image_grid_thw" in mm_kwargs and config.model.cp_style == "ulysses"
            ),
        )
        target_ids, loss_mask, *choice_tensors = per_position

        if choice_loss_fn is not None:
            choice_ids, choice_targets, choice_weights, choice_mix_weights = choice_tensors
            with maybe_activation_offloading(config.model.ac_offloading):
                out = forward(
                    model,
                    input_ids,
                    position_ids,
                    seq_lens=seq_lens,
                    choice_ids=choice_ids,
                    seq_lens_are_pre_shard=seq_lens_are_pre_shard,
                )
            loss_sum, weight_sum = choice_loss_fn(
                **choice_loss_inputs(out["logits"], choice_ids, choice_targets, choice_weights, choice_mix_weights)
            )
            del out
            return loss_sum, weight_sum

        token_count = loss_mask.sum(dtype=torch.int64)

        # Labels without a temperature make the LM head return the summed cross-entropy directly.
        labels = target_ids.masked_fill(~loss_mask, IGNORE_INDEX)

        with maybe_activation_offloading(config.model.ac_offloading):
            out = forward(
                model,
                input_ids,
                position_ids,
                seq_lens=seq_lens,
                labels=labels,
                mm_kwargs=mm_kwargs,
                mm_token_type_ids=mm_type_ids,
                seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            )
            loss_sum = out["loss"]

        del out
        return loss_sum, token_count

    maybe_record_function = nullcontext

    def run_eval_loop(data_iter):
        """Validation forward loop. Returns the global mean loss, weighted by the step normalizer (tokens,
        or the choice loss's weight sum), and the count of NaN batches."""
        total_loss_sum = torch.tensor(0.0, device="cuda")
        total_token_count = torch.tensor(0, dtype=normalizer_dtype, device="cuda")
        nan_count = torch.tensor(0, device="cuda")

        # Variable-length packing yields different per-rank batch counts. Under FSDP
        # every forward is a collective, so all ranks must agree on when to stop —
        # otherwise the first rank to exit deadlocks the rest in the next all-gather.
        # Sync per batch and exit together as soon as any rank exhausts its iterator.
        data_iter = iter(data_iter)

        with no_grad_forwards(model):
            while True:
                micro_batch = next(data_iter, None)
                has_data = torch.tensor(micro_batch is not None, dtype=torch.int32, device="cuda")
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0:
                    break
                loss_sum, token_count = compute_loss(micro_batch)
                if not torch.isnan(loss_sum.detach()):
                    total_loss_sum += loss_sum.detach()
                    total_token_count += token_count
                else:
                    nan_count += 1

        dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(total_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(nan_count, op=dist.ReduceOp.SUM)

        mean_loss = (total_loss_sum / total_token_count).item() if total_token_count.item() > 0 else float("nan")
        return mean_loss, nan_count.item()

    def run_validation(step: int) -> None:
        val_dataset = setup_dataset(
            tokenizer,
            config.val.data,
            config.model.cp,
            max_epochs=1,
            raw_dataset=val_raw_dataset,
            renderer_config=config.renderer,
            processor=processor,
            multimodal=multimodal,
        )
        val_dataloader = setup_dataloader(val_dataset, config.val.data)

        # No train/eval switch: no dropout in these models, and toggling would trigger torch.compile recompilation
        mean_loss, nan_count = run_eval_loop(val_dataloader)
        if is_tt_moe_model(model):
            # Keep validation routing out of the next training step's statistics.
            get_load_balance_stats(model)
        if nan_count > 0:
            logger.warning(f"Validation at step {step}: {nan_count} batches had NaN loss")
        if mean_loss != mean_loss:
            logger.warning(f"Validation at step {step} had no valid tokens")
        else:
            logger.success(f"Validation | Step {step} | Loss {mean_loss:.4f}")
        asyncio.run(
            monitors.log(
                {"val/loss": mean_loss, "val/perplexity": math.exp(min(mean_loss, 20)), "step": step}, step=step
            )
        )

    def predict_choices(batch: PackedChoiceEvalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Row ids ``[M]`` and fp32 choice logits ``[M, K]`` of the rows in this rank's shard of an eval bin."""
        seq_lens = batch["seq_lens"].to("cuda", non_blocking=True)
        input_ids, position_ids, (row_ids, choice_ids), seq_lens_are_pre_shard = shard_micro_batch(
            batch["input_ids"].to("cuda", non_blocking=True),
            batch["position_ids"].to("cuda", non_blocking=True),
            seq_lens,
            [batch["row_ids"].to("cuda", non_blocking=True), batch["choice_ids"].to("cuda", non_blocking=True)],
        )
        out = forward(
            model,
            input_ids,
            position_ids,
            seq_lens=seq_lens,
            choice_ids=choice_ids,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        return choice_row_ids(choice_ids, row_ids), out["logits"]

    def log_metrics(metrics: dict[str, float], step: int) -> None:
        asyncio.run(monitors.log(metrics, step=step))

    def run_choice_evals(step: int, site: ChoiceEvalSite) -> None:
        """Score the choice-eval sets due at ``site`` for weights with ``step`` optimizer updates, skipping sets
        already scored at ``step``. Collective: every rank runs the same forwards, then waits while rank 0 scores."""
        assert config.choice_eval is not None
        names: list[list[str]] = [[]]
        if world.is_master:
            due = due_choice_evals(config.choice_eval, step, site, config.max_steps)
            names = [unscored_choice_evals(config.run_dir, due, step)]
        dist.broadcast_object_list(names, src=0)
        for name in names[0]:
            eval_set = choice_eval_sets[name]
            logger.info(f"Running choice eval {name} at step {step} ({eval_set.num_forwards} forwards per rank)")
            # The start site runs inside the first step, whose own peak memory is logged after it
            if site != "start":
                torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()
            row_ids, logits = [], []
            # No train/eval switch (see run_validation), and no_grad because FSDP would keep inference_mode
            # tensors that break the next training step
            with no_grad_forwards(model):
                for batch, replay in eval_set.batches():
                    batch_row_ids, batch_logits = predict_choices(batch)
                    if not replay:
                        row_ids.append(batch_row_ids)
                        logits.append(batch_logits)
            if is_tt_moe_model(model):
                # Keep eval routing out of the next training step's statistics.
                get_load_balance_stats(model)
            peak_memory = torch.tensor(torch.cuda.max_memory_reserved() / 1024**3, device="cuda")
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            num_choices = eval_set.manifest.max_choices
            part = (
                torch.cat(row_ids).cpu().numpy() if row_ids else np.empty(0, dtype=np.int64),
                torch.cat(logits).cpu().numpy() if logits else np.empty((0, num_choices), dtype=np.float32),
            )
            parts = [None] * world.world_size if world.is_master else None
            dist.gather_object(part, parts, dst=0)
            eval_time = time.perf_counter() - start_time
            if parts is not None:
                score_choice_predictions(
                    choice_eval_fn,
                    log_metrics,
                    name=name,
                    path=eval_set.path,
                    step=step,
                    output_dir=choice_eval_step_dir(config.run_dir, name, step),
                    parts=parts,
                    choice_counts=eval_set.choice_counts,
                    metrics={
                        f"time/choice_eval/{name}": eval_time,
                        f"perf/choice_eval_peak_memory/{name}": peak_memory.item(),
                    },
                )
            dist.barrier()

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    # A broadcast must land at every step an online eval env is due. The schedule is
    # deterministic, so all ranks agree when to enter the transport collective.
    online_eval_intervals = sorted({source.interval for source in config.eval.source}) if config.eval else []

    def is_online_eval_step(step: int) -> bool:
        return any(step % interval == 0 for interval in online_eval_intervals)

    weight_sender = None
    if online_eval_intervals:
        assert config.weight_broadcast is not None
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        weight_sender = setup_weight_sender(
            config.run_dir,
            config.weight_broadcast,
            parallel_dims,
            config.model.lora,
        )
        # Startup broadcast of the incoming policy: fails fast on a broken
        # transport and lets the evals process re-trigger at the resume step
        # (older broadcasts may have been cleaned).
        startup_version = checkpoint_step or 0
        if world.is_master:
            prune_broadcasts_beyond(config.run_dir, startup_version)
        logger.info(f"Broadcasting startup policy weights (v{startup_version}) for online evals")
        weight_sender.broadcast(model, startup_version)

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3  # GiB
    is_first_step = True
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function  # noqa: F841 – captured by run_forward_loop closure
    max_peak_memory = 0.0
    while not resumed_past_end:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step >= config.max_steps

        memory_profiler = (
            MemoryProfiler(progress.step, config.memory_profiler_path) if config.memory_profiler_path else None
        )

        step_start_time = time.perf_counter()
        forward_backward_start_time = time.perf_counter()

        step_loss_sum = torch.tensor(0.0, device="cuda")
        nan_loss_count = torch.tensor(0, device="cuda")
        is_moe_model = is_tt_moe_model(model)
        moe_stats = (
            {
                "max_vio/mean": torch.tensor(0.0),
                "max_vio/max": torch.tensor(0.0),
                "routing_confidence/mean": torch.tensor(0.0),
            }
            if is_moe_model
            else {}
        )
        run_validation_this_step = config.val is not None and (
            (is_first_step and config.val.eval_on_start)
            or (not is_first_step and progress.step % config.val.interval == 0)
        )
        if gradient_manager is None:
            micro_batches = (next(dataiter) for _ in range(grad_accum_steps))
            step_local_token_count = torch.tensor(0, dtype=normalizer_dtype, device="cuda")
        else:
            micro_batches = [next(dataiter) for _ in range(grad_accum_steps)]
            local_token_count = sum(int(micro_batch["loss_mask"].sum()) for micro_batch in micro_batches)
            global_step_token_count = torch.tensor(local_token_count, dtype=torch.int64, device="cuda")
            dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
            global_token_count_val = global_step_token_count.item() // cp_size
            grad_scale = (
                parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
                if global_token_count_val > 0
                else 1.0
            )
            prepare_gradient_offload(
                gradient_manager,
                grad_scale,
                overlap_optimizer=not run_validation_this_step,
            )

        for micro_step, micro_batch in enumerate(micro_batches):
            if config.log.log_data:
                print_sample(
                    micro_batch["input_ids"].flatten().tolist(), micro_batch["loss_mask"].flatten().tolist(), tokenizer
                )

            with maybe_record_function("forward"):
                local_loss_sum, batch_token_count = compute_loss(micro_batch)

            if gradient_manager is None:
                step_local_token_count += batch_token_count

            if torch.isnan(local_loss_sum.detach()):
                nan_loss_count += 1
                logger.warning("Local loss is nan, excluding this micro step from backward")
                scaled_loss = torch.nan_to_num(local_loss_sum, nan=0.0) / grad_accum_steps
            else:
                step_loss_sum += local_loss_sum.detach()
                scaled_loss = local_loss_sum / grad_accum_steps

            with maybe_record_function("backward"):
                begin_backward(gradient_manager, final_backward=micro_step == grad_accum_steps - 1)
                scaled_loss.backward()
                finish_backward(gradient_manager)

            if is_moe_model:
                for name, value in get_global_moe_stats(model, ep_group, dp_cp_group).items():
                    moe_stats[f"{name}/mean"] += value / grad_accum_steps
                    if name == "max_vio":
                        moe_stats["max_vio/max"] = torch.maximum(moe_stats["max_vio/max"], value)

        forward_backward_time = time.perf_counter() - forward_backward_start_time

        if gradient_manager is None:
            global_step_token_count = step_local_token_count.clone()
            dist.all_reduce(global_step_token_count, op=dist.ReduceOp.SUM, group=dp_cp_group)
            global_token_count_val = global_step_token_count.item()
            if global_token_count_val > 0:
                grad_scale = parallel_dims.fsdp_gradient_divide_factor * grad_accum_steps / global_token_count_val
                scale_gradients_(None, model, grad_scale)

        # Run validation after forward-backward (so torch.compile sees training graph first) but before
        # optimizer step (so eval_on_start evaluates untrained weights)
        if run_validation_this_step:
            run_validation(progress.step)
        if is_first_step and config.choice_eval is not None:
            run_choice_evals(progress.step - 1, "start")

        # Compute the global mean loss for logging.
        dist.all_reduce(step_loss_sum, op=dist.ReduceOp.SUM, group=dp_cp_group)
        dist.all_reduce(nan_loss_count, op=dist.ReduceOp.SUM)
        if global_token_count_val > 0:
            batch_loss = (step_loss_sum / global_token_count_val).item()
        else:
            batch_loss = 0.0
        nan_loss_count = nan_loss_count.item()

        grad_norm: torch.Tensor | None = None
        if config.optim.max_norm is not None:
            logger.debug(f"Clipping gradients with max norm {config.optim.max_norm}")
            grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm, parallel_dims.ep_enabled)
        logger.debug("Optimizer step")
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Checkpoint the step we just finished. The last step's checkpoint is written once after
        # the loop, so skip it here to avoid a double-save. Weight broadcasts land at
        # online-eval steps — they are how the inference server picks up the new policy.
        save_ckpt_time = 0
        is_ckpt_step = bool(config.ckpt and config.ckpt.interval) and progress.step % config.ckpt.interval == 0
        if ckpt_manager is not None and is_ckpt_step and not is_last_step:
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
            save_ckpt_time += time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()

        broadcast_weights_time = 0
        if weight_sender is not None and not is_last_step and is_online_eval_step(progress.step):
            logger.info(f"Broadcasting weights at step {progress.step}")
            broadcast_start_time = time.perf_counter()
            weight_sender.broadcast(model, step=progress.step)
            broadcast_weights_time = time.perf_counter() - broadcast_start_time

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Compute step metrics. CP shards the same sequences across cp ranks
        # (sequence-sharded data parallelism on the seq dim), so the unique
        # training tokens per step is dp_size * (batch_per_dp_rank * seq).
        # The `dp` mesh excludes cp by construction (parallel_dims.py), mirroring
        # the RL trainer's accounting (rl/train.py).
        dp_size = parallel_dims.get_mesh("dp").size()
        num_local_tokens = config.data.seq_len * (config.data.batch_size // dp_size)
        num_tokens = dp_size * num_local_tokens
        progress.total_tokens += num_tokens
        dataset_progress = get_dataset_progress(dataloader)
        progress.total_samples = dataset_progress["step"]
        perf_counter = get_perf_counter(model, config.data.seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB
        max_peak_memory = max(max_peak_memory, peak_memory)

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | {format_time(step_time):>7} | Loss {batch_loss:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}% | Peak Mem. {peak_memory:.1f}/{max_memory:.1f} GiB ({peak_memory / max_memory * 100:.1f}%)"
        if is_moe_model:
            for name, label in (("max_vio/mean", "Max Vio"), ("routing_confidence/mean", "Routing Conf.")):
                value = moe_stats[name].item()
                if value > 0:
                    step_message += f" | {label} {value:.4f}"
        logger.success(step_message)

        # Log progress metrics
        samples_by_source = dataset_progress["num_samples"]
        tokens_by_source = dataset_progress["num_tokens"]
        total_samples = sum(samples_by_source.values())
        total_tokens = sum(tokens_by_source.values())
        progress_metrics = {
            "progress/epoch": dataset_progress["epoch"],
            "progress/num_samples": progress.total_samples,
            "progress/num_tokens": progress.total_tokens,
            "step": progress.step,
        }
        # At least two subsets/splits
        if len(samples_by_source) > 1:
            progress_metrics.update(
                **{
                    f"progress/{subset_or_split}/ratio_samples": num_samples / total_samples
                    for subset_or_split, num_samples in samples_by_source.items()
                },
                **{
                    f"progress/{subset_or_split}/ratio_tokens": num_tokens / total_tokens
                    for subset_or_split, num_tokens in tokens_by_source.items()
                },
            )
        asyncio.run(monitors.log(progress_metrics, step=progress.step))

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/peak_memory": peak_memory,
            "perf/mfu": mfu,
            "step": progress.step,
        }
        asyncio.run(monitors.log(perf_metrics, step=progress.step))

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "step": progress.step,
        }
        if grad_norm is not None:
            optim_metrics["optim/grad_norm"] = grad_norm.item()
        asyncio.run(monitors.log(optim_metrics, step=progress.step))

        loss_log_metrics = {
            "loss/mean": batch_loss,
            "loss/perplexity": math.exp(min(batch_loss, 20)),
            "loss/nan_count": nan_loss_count,
            "step": progress.step,
        }
        # Log tensor stats
        asyncio.run(monitors.log(loss_log_metrics, step=progress.step))

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/broadcast_weights": broadcast_weights_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        asyncio.run(monitors.log(time_metrics, step=progress.step))

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.run_dir)
        disk_metrics["step"] = progress.step
        asyncio.run(monitors.log(disk_metrics, step=progress.step))

        moe_log_metrics = {name: value.item() for name, value in moe_stats.items()}
        if moe_log_metrics:
            asyncio.run(monitors.log({**moe_log_metrics, "step": progress.step}, step=progress.step))

        is_first_step = False

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

        if config.choice_eval is not None:
            run_choice_evals(progress.step, "step")

        if is_last_step:
            break
        progress.step += 1

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Write final checkpoint
    if config.ckpt is not None and not resumed_past_end:
        logger.info(f"Saving final checkpoint at step {progress.step}")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    if config.choice_eval is not None:
        run_choice_evals(progress.step - 1 if resumed_past_end else progress.step, "final")

    # Broadcast the final weights so the evals process can run its forced final epoch; after a resume past
    # the end, the startup broadcast already carried them
    if weight_sender is not None and not resumed_past_end:
        logger.info("Broadcasting final weights")
        weight_sender.broadcast(model, step=progress.step)

    if gradient_manager is not None:
        gradient_manager.close()

    logger.info(f"Peak memory: {max_peak_memory:.1f} GiB")
    logger.success("SFT trainer finished")
    asyncio.run(monitors.finalize())


def main():
    set_proc_title("SFTTrainer")
    train(cli(SFTConfig))


if __name__ == "__main__":
    main()
