import prime_rl._compat  # noqa: F401

from contextlib import nullcontext
from functools import partial
import time
import asyncio
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.transports.weights import prune_broadcasts_beyond, setup_weight_sender
from prime_rl.utils.act_offloading import maybe_activation_offloading
import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader, prepare_micro_batch, validate_score_centering_data
from prime_rl.utils.cp import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    setup_context_parallel,
    setup_cp_params,
    shard_for_cp,
    validate_cp_seq_lens,
)
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    _mismatch_kl_from_log_ratio,
    selective_log_softmax,
    selective_log_softmax_with_sampling_mask,
    selective_topk_log_softmax,
    setup_rl_loss_fn,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.trainer.rl.annotations import AnnotationWriter
from prime_rl.trainer.model import (
    forward,
    get_full_offload_dtype_policy,
    get_expert_load_stats,
    get_global_moe_stats,
    is_tt_moe_model,
    setup_model,
    setup_processor,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    GarbageCollection,
    MemoryProfiler,
    TensorStats,
    begin_backward,
    broadcast_pause,
    clip_grad_norm_,
    filter_rl_trainer_tensor_stats_for_wandb,
    finish_backward,
    get_ckpt_disk_metrics,
    prepare_gradient_offload,
    scale_gradients_,
    setup_full_cpu_optimizer_offload,
    setup_torch_distributed,
)
from prime_rl.trainer.world import get_world
from prime_rl.trainer.lora import get_lora_state
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl import monitors
from prime_rl.utils.config import cli
from prime_rl.control.controller import Controller
from prime_rl.control.server import ControlServer
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.worker_pool import WorkerPool
from prime_rl.utils.pathing import resolve_latest_ckpt_step, write_resolved_config
from prime_rl.utils.utils import clean_exit


@clean_exit
def train(config: TrainerConfig):
    # Setup world and logger
    world = get_world()
    if world.is_master:
        write_resolved_config(config.output_dir, config, "trainer")
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting RL trainer in {world} (output_dir={config.output_dir})")

    # Setup the monitors
    asyncio.run(
        monitors.setup(
            producer="trainer",
            wandb=config.monitors.wandb,
            file=config.monitors.file,
            output_dir=config.output_dir,
            run_config=config,
        )
    )

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.is_master:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat)

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.offload == "full",
    )
    if config.model.offload == "full":
        setup_full_cpu_optimizer_offload()
    # Configurable to support ROCm/AMD GPUs where reduced precision
    # matmul corrupts softmax over large vocabularies. Override via config
    # (e.g. matmul_precision = "highest") on ROCm.
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Resolve ep="auto" to a concrete integer before creating parallel dims
    resolve_ep(config.model)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)

    # Check for checkpoint to resume from
    checkpoint_step = None
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt, resume=config.resume)

    if config.resume is not None:
        if config.resume.dir is not None:
            checkpoint_step = config.resume.dir_step
        else:
            checkpoint_step = config.resume.step
            if checkpoint_step is None:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)

    # Initialize the model
    logger.info(f"Initializing model ({config.model})")
    t0 = time.perf_counter()
    loading_from_ckpt_later = checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
    logger.debug(f"Initialized model in {format_time(time.perf_counter() - t0)}")

    processor = None
    mm_adapter = None
    if config.model.vlm is not None:
        processor = setup_processor(config.model)
        if processor is None:
            raise ValueError("Multimodal training requires a model image processor")
        mm_adapter = get_multimodal_adapter(model.config.model_type)
    prepare = partial(prepare_micro_batch, processor=processor, mm_adapter=mm_adapter)
    micro_batch_workers = WorkerPool(config.data.num_workers)

    if config.model.vlm is not None and not getattr(model, "supports_packed_multimodal_training", False):
        raise ValueError("Packed multimodal training requires model support")

    # Set up the loss function for the RL loss type (ce / ref_kl are fixed)
    logger.info(f"Initializing loss function ({config.loss})")
    rl_loss_fn = setup_rl_loss_fn(config.loss)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    t0 = time.perf_counter()
    optimizer, gradient_manager = setup_optimizer(
        config.optim,
        list(model.named_parameters()),
        parallel_dims,
        offload=config.model.offload,
        model=model,
        full_offload_dtype_policy=(get_full_offload_dtype_policy(model) if config.model.offload == "full" else None),
    )
    logger.debug(f"Initialized optimizer in {format_time(time.perf_counter() - t0)}")

    logger.info(f"Initializing scheduler ({config.scheduler})")
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)

    # Set up weight broadcast (skip when using fake data since there's no inference server)
    if config.data.fake:
        weight_sender = None
        logger.info("Skipping weight broadcast setup (fake data mode)")
    else:
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        t0 = time.perf_counter()
        weight_sender = setup_weight_sender(
            config.output_dir,
            config.weight_broadcast,
            parallel_dims,
            config.model.lora,
        )
        logger.debug(f"Initialized weight broadcast in {format_time(time.perf_counter() - t0)}")

    if parallel_dims.cp_enabled:
        setup_context_parallel(model, config.model, parallel_dims)

    is_moe_model = is_tt_moe_model(model)
    ep_group = parallel_dims.get_mesh("ep").get_group() if parallel_dims.ep_enabled else None

    # Fresh adapter init after FSDP materialization (the pretrained checkpoint
    # carries no adapter weights); a checkpoint resume below overwrites it.
    if config.model.lora is not None:
        get_lora_state().reset_adapter_parameters()

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if checkpoint_step is not None:
        resume_dir = config.resume.dir if config.resume else None
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer],
            scheduler,
            progress,
            path=resume_dir / "trainer" if resume_dir is not None else None,
        )
        # The checkpoint finished step ``checkpoint_step``; resume training at the next step.
        progress.step += 1
        logger.info(
            f"Resuming from step {checkpoint_step} "
            f"(total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
        )
    else:
        logger.info("Starting from scratch")

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    t0 = time.perf_counter()
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake, config.model.seq_len, parallel_dims.get_mesh("dp").size())
    else:
        dataloader = DataLoader(
            config.output_dir,
            progress.step,
            parallel_dims.get_mesh("dp").size(),
            config.rollout_transport,
        )
    logger.debug(f"Initialized data loader in {format_time(time.perf_counter() - t0)}")

    annotation_writer = AnnotationWriter(
        parallel_dims, world, config.monitors.file.float_decimals if config.monitors.file else None
    )

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    start_step = progress.step
    max_peak_memory = 0.0
    controller = Controller(requires_step=False)
    control_server = ControlServer(controller, config.control if world.is_master else None)
    control_server.start()
    paused = False
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step >= config.max_steps
        if config.control is not None:
            paused = broadcast_pause(controller, progress.step)
            is_last_step = is_last_step or paused

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # Broadcast the incoming policy (v{progress.step-1}) before waiting for its
        # rollouts so the trainer and inference pool join the same update lifecycle,
        # and so a broken broadcast path fails at startup instead of after the first
        # optimizer step.
        if progress.step == start_step and weight_sender is not None:
            startup_version = progress.step - 1
            if world.is_master:
                prune_broadcasts_beyond(config.output_dir, startup_version)
            logger.info(f"Broadcasting startup policy weights (v{startup_version}) to inference engines")
            t0 = time.perf_counter()
            weight_sender.broadcast(model, startup_version)
            logger.debug(
                f"Broadcast startup policy weights (v{startup_version}) in {format_time(time.perf_counter() - t0)}"
            )

        # Wait for the batch to be available
        logger.debug("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.perf_counter()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.perf_counter() - wait_for_batch_start_time
        logger.debug(f"Waited for batch for {format_time(wait_for_batch_time)}")

        # Load the training batch
        logger.debug("Loading batch")
        load_data_start_time = time.perf_counter()
        micro_batches = dataloader.get_batch()
        load_data_time = time.perf_counter() - load_data_start_time
        logger.debug(f"Loaded batch in {format_time(load_data_time)}")

        batch_size = len(micro_batches)
        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.perf_counter()
        perf_counter = get_perf_counter(model)

        # Normalize each loss component by its own global (dp_cp) denominator, so every rank
        # divides by the same value. With a per-rank denominator, ranks with fewer loss
        # tokens implicitly upweight their per-token gradient contribution after FSDP averaging.
        # FSDP's per-rank divide is undone after the microbatch loop via
        # fsdp_gradient_divide_factor. One batched collective keeps every rank issuing the same
        # op regardless of which components its samples carry. rl divides by the sum of its
        # weights (the token count for 0/1 weights; the group count under prompt-mean
        # aggregation); ce and ref_kl divide by their token counts, so a fractional ce weight
        # (echo's alpha) scales the loss instead of cancelling out.
        local_rl_scale = 0.0
        local_ce_scale = 0
        local_ref_kl_scale = 0
        for micro_batch in micro_batches:
            mask = micro_batch["loss_mask"]
            rl_w = micro_batch["rl_weights"]
            local_rl_scale += float(rl_w[mask].sum(dtype=torch.float64)) if rl_w is not None else int(mask.sum())
            if micro_batch["ce_weights"] is not None:
                local_ce_scale += int((micro_batch["ce_weights"] != 0).sum())
            if micro_batch["ref_kl_weights"] is not None:
                local_ref_kl_scale += int((micro_batch["ref_kl_weights"] != 0).sum())
        global_scales = torch.tensor(
            [local_rl_scale, local_ce_scale, local_ref_kl_scale], dtype=torch.float64, device="cuda"
        )
        dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
        dist.all_reduce(global_scales, op=dist.ReduceOp.SUM, group=dp_cp_group)
        rl_scale, ce_scale, ref_kl_scale = (scale if scale > 0 else 1 for scale in global_scales.tolist())
        prepare_gradient_offload(
            gradient_manager,
            parallel_dims.fsdp_gradient_divide_factor,
            overlap_optimizer=True,
        )

        logger.debug(f"Starting forward and backward pass ({batch_size=})")
        tensor_stats = TensorStats()  # Accumulates metrics across micro-batches and ranks for logging
        cp_enabled = parallel_dims.cp_enabled
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
        cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
        cp_size = parallel_dims.cp

        step_tokens_per_expert = 0
        step_local_num_tokens = 0
        step_local_flops = 0
        for micro_step, micro_batch in enumerate(micro_batch_workers(prepare, micro_batches)):
            step_local_num_tokens += micro_batch["input_ids"].shape[1]
            score_centering = config.loss.type == "icepop" and config.loss.score_centering
            if score_centering:
                vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
                validate_score_centering_data(micro_batch, vocab_size=vocab_size)
            step_local_flops += perf_counter.flop_counter(micro_batch["seq_lens"].tolist())
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            # Each token's env: the sequences' env names spread over their lengths
            env_names = sorted(set(micro_batch["env_names"]))
            env_ids = torch.from_numpy(
                np.repeat(
                    np.array([env_names.index(name) for name in micro_batch["env_names"]], dtype=np.int64),
                    micro_batch["sequence_lengths"],
                )
            ).to("cuda")
            inference_logprobs = micro_batch["inference_logprobs"].to("cuda")
            ref_logprobs = micro_batch["ref_logprobs"].to("cuda") if micro_batch["ref_logprobs"] is not None else None
            rl_weights = micro_batch["rl_weights"].to("cuda") if micro_batch["rl_weights"] is not None else None
            ce_weights = micro_batch["ce_weights"].to("cuda") if micro_batch["ce_weights"] is not None else None
            ref_kl_weights = (
                micro_batch["ref_kl_weights"].to("cuda") if micro_batch["ref_kl_weights"] is not None else None
            )
            # Stays on the host in its wire dtype (uint8/uint16) until after the CP shard below
            routed_experts = micro_batch["routed_experts"]

            if routed_experts is None and config.enable_router_replay:
                raise ValueError(
                    "You must set `enable_return_routed_experts=True` in the inference config or pass `--enable-return-routed-experts` to vLLM server to use router replay."
                )

            if routed_experts is not None and not config.enable_router_replay:
                # we could've gotten routed experts from the inference server, but we didn't enable router replay
                routed_experts = None

            sampling_mask = (
                micro_batch["sampling_mask"].to("cuda") if micro_batch["sampling_mask"] is not None else None
            )

            sampler_support_complete = (
                (sampling_mask >= 0).any(-1) if score_centering and sampling_mask is not None else None
            )
            topk_ids = micro_batch["top_logprobs_ids"] if score_centering else None
            sampler_topk = micro_batch["top_logprobs_values"].to("cuda") if topk_ids is not None else None
            topk_valid = (topk_ids >= 0).to("cuda") if topk_ids is not None else None
            if topk_ids is not None:
                topk_ids = topk_ids.to("cuda")

            mm_kwargs = micro_batch.pop("mm_kwargs")
            if mm_kwargs is not None:
                mm_kwargs = {key: value.to("cuda") for key, value in mm_kwargs.items()}
            mm_forward_policy = micro_batch.pop("mm_forward_policy")
            mm_token_type_ids = (
                micro_batch["mm_token_type_ids"].to("cuda")
                if micro_batch.get("mm_token_type_ids") is not None
                else None
            )

            seq_lens = micro_batch["seq_lens"].to("cuda")

            labels = shift_tensor_left(input_ids)
            if topk_ids is not None:
                topk_ids = shift_tensor_left(topk_ids, pad_value=-1)
            if sampling_mask is not None:
                # Sampling masks ride at the sampled token's own position (like inference
                # logprobs); shift to align with the label each position predicts.
                sampling_mask = shift_tensor_left(sampling_mask, pad_value=-1)

            seq_lens_are_pre_shard = False

            if cp_enabled:
                validate_cp_seq_lens(micro_batch["seq_lens"], input_ids.shape[-1])
                defer_vlm_cp_to_model = bool(
                    mm_forward_policy is not None and mm_forward_policy.defer_context_parallelism
                )
                if not defer_vlm_cp_to_model:
                    input_ids, position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size)
                seq_lens_are_pre_shard = True
                labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp_size)
                if topk_ids is not None:
                    topk_ids = shard_for_cp(topk_ids, cp_rank=cp_rank, cp_world_size=cp_size)
                if routed_experts is not None and not defer_vlm_cp_to_model:
                    routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)
                if sampling_mask is not None:
                    # The LM head consumes masks after any deferred VLM sharding, so
                    # they must follow the label shard rather than the input shard.
                    sampling_mask = shard_for_cp(sampling_mask, cp_rank=cp_rank, cp_world_size=cp_size)

            if routed_experts is not None:
                routed_experts = routed_experts.to("cuda").int()

            temperatures = micro_batch["temperatures"].to("cuda")

            # Shard temperatures for context parallelism if enabled
            if cp_enabled:
                temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=cp_size)

            if sampling_mask is not None:
                assert sampling_mask.shape[:2] == labels.shape, (
                    f"sampling_mask shape {tuple(sampling_mask.shape)} is not aligned with "
                    f"labels shape {tuple(labels.shape)}"
                )

            # Forward pass with per-token temperatures
            with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading, model):
                out = forward(
                    model,
                    input_ids,
                    position_ids,
                    labels=labels,
                    temperature=temperatures,
                    mm_kwargs=mm_kwargs,
                    mm_forward_policy=mm_forward_policy,
                    mm_token_type_ids=mm_token_type_ids,
                    seq_lens=seq_lens,
                    seq_lens_are_pre_shard=seq_lens_are_pre_shard,
                    routed_experts=routed_experts,
                    sampling_mask=sampling_mask,
                    topk_ids=topk_ids,
                )

            if out.get("logprobs") is None:
                # VanillaOutputLinear was used - need to compute logprobs externally with per-token temps
                assert out.get("logits") is not None, "Logits must be provided to compute logprobs"
                logits = out["logits"]
                # Per-token temperature scaling: temperatures is [batch, seq], logits is [batch, seq, vocab]
                scaled_logits = logits / temperatures.unsqueeze(-1)
                if sampling_mask is not None:
                    out["logprobs"] = selective_log_softmax_with_sampling_mask(scaled_logits, labels, sampling_mask)
                else:
                    out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                out["entropy"] = compute_entropy(scaled_logits)
                if topk_ids is not None:
                    out["topk_logprobs"] = selective_topk_log_softmax(scaled_logits, topk_ids, sampling_mask, labels)
            # else: FusedOutputLinear was used - logprobs already computed with per-token temperatures

            if cp_enabled:
                out["logprobs"] = gather_for_cp(out["logprobs"], cp_group)
                out["entropy"] = gather_for_cp_wo_grad(out["entropy"], cp_size, cp_group)
                if out.get("topk_logprobs") is not None:
                    out["topk_logprobs"] = gather_for_cp(out["topk_logprobs"], cp_group)

            vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
            # This is not really necessary as the first token should be masked out, but we do it anyway to be sure
            out["logprobs"] = shift_tensor_right(
                out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
            )
            out["entropy"] = shift_tensor_right(
                out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
            )

            trainer_topk = out.get("topk_logprobs")
            if trainer_topk is not None:
                trainer_topk = torch.cat([torch.zeros_like(trainer_topk[:, :1]), trainer_topk[:, :-1]], dim=1)

            # Compute loss
            sequence_lengths = micro_batch["sequence_lengths"]
            loss, loss_tensors = compute_loss(
                trainer_logprobs=out["logprobs"].squeeze().split(sequence_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(sequence_lengths),
                ref_logprobs=ref_logprobs.squeeze().split(sequence_lengths) if ref_logprobs is not None else None,
                advantages=advantages.squeeze().split(sequence_lengths),
                loss_mask=loss_mask.squeeze().split(sequence_lengths),
                rl_weights=rl_weights.squeeze().split(sequence_lengths) if rl_weights is not None else None,
                ce_weights=ce_weights.squeeze().split(sequence_lengths) if ce_weights is not None else None,
                ref_kl_weights=ref_kl_weights.squeeze().split(sequence_lengths) if ref_kl_weights is not None else None,
                rl_loss_fn=rl_loss_fn,
                rl_scale=rl_scale,
                ce_scale=ce_scale,
                ref_kl_scale=ref_kl_scale,
                trainer_topk_logprobs=trainer_topk.squeeze(0).split(sequence_lengths)
                if trainer_topk is not None
                else None,
                sampler_topk_logprobs=sampler_topk.squeeze(0).split(sequence_lengths)
                if sampler_topk is not None
                else None,
                topk_valid=topk_valid.squeeze(0).split(sequence_lengths) if topk_valid is not None else None,
                sampler_support_complete=sampler_support_complete.squeeze(0).split(sequence_lengths)
                if sampler_support_complete is not None
                else None,
            )

            # Backward pass
            with maybe_record_function("backward"):
                begin_backward(gradient_manager, final_backward=micro_step == len(micro_batches) - 1)
                loss.backward()
                finish_backward(gradient_manager)

            mm_kwargs = None

            # Every CP rank computes the whole sequence's loss, and the global normalizers count each
            # sequence once per CP rank (gradients still sum correctly), so undo that for the logged value.
            tensor_stats.add("loss", loss * cp_size)

            tensor_stats.add("entropy", out["entropy"], loss_mask, env_ids, env_names)

            # Mismatch KL is only meaningful where sampling logprobs exist —
            # keep rl/ref_kl member tokens (policy-sampled), exclude tokens
            # whose action component is ce (frozen-model tokens).
            mismatch_mask = loss_mask
            if rl_weights is not None or ref_kl_weights is not None:
                sampled_mask = (rl_weights != 0) if rl_weights is not None else loss_mask
                if ref_kl_weights is not None:
                    sampled_mask = sampled_mask | (ref_kl_weights != 0)
                mismatch_mask = loss_mask & sampled_mask
            with torch.no_grad():
                mismatch_kl = _mismatch_kl_from_log_ratio(out["logprobs"] - inference_logprobs)
            tensor_stats.add("mismatch_kl", mismatch_kl, mismatch_mask, env_ids, env_names)

            annotation_writer.export(micro_batch, out)

            if is_moe_model:
                moe_stats, tokens_per_expert = get_global_moe_stats(model, ep_group, dp_cp_group)
                for name, value in moe_stats.items():
                    tensor_stats.add(name, value)
                step_tokens_per_expert += tokens_per_expert

            for key, loss_tensor in loss_tensors.items():
                tensor_stats.add(key, loss_tensor)

            # Debug log with *local, micro step* stats, evaluated only when debug logging is on
            logger.opt(lazy=True).debug(
                "Micro Step {} | Loss {:.4f} | Entropy {:.4f} | Mismatch KL {:.4f}",
                lambda: f"{micro_step + 1}/{len(micro_batches)}",
                lambda: loss.item() * cp_size,
                lambda: out["entropy"][loss_mask].mean().item(),
                lambda: mismatch_kl[mismatch_mask].mean().item(),
            )

        annotation_writer.flush()
        # Free this batch before the next one is received and converted
        del micro_batches, micro_batch

        # compute_loss already divided by the global token count. Undo FSDP's per-rank averaging
        # across dp_cp so the final gradient is the true per-token mean over the global batch.
        if gradient_manager is None:
            scale_gradients_(None, model, parallel_dims.fsdp_gradient_divide_factor)

        # Optionally, clip the gradients
        grad_norm: torch.Tensor | None = None
        if config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm)

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Broadcast the model just produced (policy v{progress.step}). The final
        # broadcast keeps inference synchronized with the completed trainer.
        if weight_sender is None:
            broadcast_weights_time = 0
        else:
            broadcast_weights_start_time = time.perf_counter()
            # The per-layer gather + fp8 conversion peaks ~50 GiB above the
            # resident weights; release cached blocks (incl. offload-stream
            # pools) so the broadcast gets the full headroom. Drain all
            # pending work first: empty_cache returns blocks to the driver,
            # so a still-running kernel holding a cached block (e.g. the
            # optimizer step's tail) faults with an illegal memory access
            # once its block is freed under it.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            weight_sender.broadcast(model, step=progress.step)
            broadcast_weights_time = time.perf_counter() - broadcast_weights_start_time

        # Checkpoint the step we just finished (model = policy v{progress.step}).
        if (
            (config.ckpt and config.ckpt.interval)
            # the last step is written once after the loop (final ckpt), so skip it here
            and not is_last_step
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        tensor_metrics = tensor_stats.compute_stats()
        if is_moe_model:
            tensor_metrics.update(get_expert_load_stats(step_tokens_per_expert, dp_cp_group))

        # The dp mesh excludes cp, whose ranks hold the same rows.
        global_counts = torch.tensor([step_local_num_tokens, step_local_flops], dtype=torch.float64, device="cuda")
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp").get_group())
        num_tokens, num_flops = int(global_counts[0].item()), global_counts[1].item()
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, forward_backward_time)
        mfu = perf_counter.get_step_mfu(num_flops, forward_backward_time)
        mfu_6dn = perf_counter.get_step_mfu(
            perf_counter.flop_counter.flops_per_token_6dn * num_tokens, forward_backward_time
        )
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB
        max_peak_memory = max(max_peak_memory, peak_memory)

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        active_step_time = step_time - wait_for_batch_time
        if progress.step > start_step and wait_for_batch_time >= active_step_time:
            logger.warning(
                f"Trainer waited {format_time(wait_for_batch_time)} for a batch, at least as long as its "
                f"{format_time(active_step_time)} active step time. Train-inference compute is imbalanced; "
                "add more inference nodes."
            )
        step_message = f"Step {progress.step} | {format_time(step_time):>7} | Loss {tensor_metrics['loss/mean']:.4f} | Entropy {tensor_metrics.get('entropy/all/mean', float('nan')):.4f}"
        if "mismatch_kl/all/mean" in tensor_metrics:
            step_message += f" | Mismatch KL {tensor_metrics['mismatch_kl/all/mean']:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}% | MFU 6DN {mfu_6dn:.1f}% | Peak Mem. {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_metrics:
            step_message += f" | Max Vio {tensor_metrics['max_vio/mean']:.4f}"
        if "routing_confidence/mean" in tensor_metrics:
            step_message += f" | Routing Conf. {tensor_metrics['routing_confidence/mean']:.4f}"
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/mfu_6dn": mfu_6dn,
            "perf/peak_memory": peak_memory,
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

        # Compute derived metrics
        entropy_mean = tensor_metrics.get("entropy/all/mean", 0.0)
        mismatch_kl_mean = tensor_metrics.get("mismatch_kl/all/mean")
        if mismatch_kl_mean is not None and entropy_mean > 0:
            tensor_metrics["kl_ent_ratio/mean"] = mismatch_kl_mean / entropy_mean

        tensor_metrics["step"] = progress.step
        asyncio.run(monitors.log(filter_rl_trainer_tensor_stats_for_wandb(tensor_metrics), step=progress.step))

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_batch": wait_for_batch_time,
            "time/load_data": load_data_time,
            "time/broadcast_weights": broadcast_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        asyncio.run(monitors.log(time_metrics, step=progress.step))

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.output_dir)
        disk_metrics["step"] = progress.step
        asyncio.run(monitors.log(disk_metrics, step=progress.step))

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

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
    if config.ckpt is not None or paused:
        logger.info(f"Saving final checkpoint at step {progress.step}")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        if paused:
            controller.mark_saved(progress.step)
        else:
            ckpt_manager.maybe_clean()

    if gradient_manager is not None:
        gradient_manager.close()
    micro_batch_workers.close()

    logger.info(f"Peak memory: {max_peak_memory:.1f} GiB")
    logger.success(f"RL trainer {'paused' if paused else 'finished'} at step {progress.step}")
    asyncio.run(monitors.finalize(paused=paused))
    if paused and world.is_master:
        # The run's owner reads the committed pause from the control API, then stops the run.
        logger.info("Paused, serving the control API until the run is stopped")
        control_server.wait()


def main():
    """Main entry-point for RL trainer. Run using `uv run trainer`"""
    set_proc_title("Trainer")
    train(cli(TrainerConfig))


if __name__ == "__main__":
    main()
