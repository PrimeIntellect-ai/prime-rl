import prime_rl._compat  # noqa: F401

from contextlib import nullcontext
from functools import partial
import time
import asyncio
from datetime import timedelta
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.transports.weights import latest_finished_broadcast, prune_broadcasts_beyond, setup_weight_sender
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import Progress, ckpt_output_dir, setup_ckpt_manager, setup_model_files
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader, drop_micro_batches
from prime_rl.utils.cp import setup_context_parallel
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.trainer.rl.loss import setup_rl_loss_fn
from prime_rl.trainer.rl.annotations import AnnotationWriter
from prime_rl.trainer.rl.step import ForwardBackward
from prime_rl.trainer.rl.warmup import warmup
from prime_rl.trainer.model import (
    get_full_offload_dtype_policy,
    get_expert_load_stats,
    get_global_moe_stats,
    is_tt_moe_model,
    setup_model,
    check_ckpt_dir,
    setup_tokenizer,
    use_resumed_model_files,
)
from prime_rl.trainer.moe_runtime import setup_moe_comm_env
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.vlm import setup_processor
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    GarbageCollection,
    MemoryProfiler,
    broadcast_step_action,
    clip_grad_norm_,
    filter_rl_trainer_tensor_stats_for_wandb,
    get_ckpt_disk_metrics,
    get_peak_memory_across_ranks,
    scale_gradients_,
    setup_cpu_offload,
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
from prime_rl.utils.pathing import (
    get_ckpt_dir,
    get_step_path,
    is_checkpoint_step_dir,
    resolve_latest_ckpt_step,
    resolve_latest_trainer_ckpt_step,
    write_resolved_config,
)
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
            prime=config.monitors.prime,
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

    # Check for checkpoint to resume from
    ckpt_dir = get_ckpt_dir(ckpt_output_dir(config.output_dir, config.ckpt))
    checkpoint_step = None
    if config.resume is not None:
        if config.resume.dir is not None:
            checkpoint_step = config.resume.dir_step
        else:
            checkpoint_step = config.resume.step
            if checkpoint_step is None and config.rollout_transport.retain:
                # A trainer restarted on its own: the orchestrator may be steps ahead.
                checkpoint_step = resolve_latest_trainer_ckpt_step(ckpt_dir)
            elif checkpoint_step is None:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_dir)
    if checkpoint_step is not None:
        resume_step_dir = config.resume.dir or get_step_path(ckpt_dir, checkpoint_step)
        use_resumed_model_files(config.model, config.tokenizer, resume_step_dir)

    # Set precision
    setup_moe_comm_env(config.model)
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.offload == "full",
    )
    setup_cpu_offload(config.model)
    # Configurable to support ROCm/AMD GPUs where reduced precision
    # matmul corrupts softmax over large vocabularies. Override via config
    # (e.g. matmul_precision = "highest") on ROCm.
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Resolve ep="auto" to a concrete integer before creating parallel dims
    resolve_ep(config.model)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)

    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    check_ckpt_dir(config.model)
    tokenizer = setup_tokenizer(config.tokenizer, config.model.name)
    processor = None
    if config.model.vlm is not None:
        processor = setup_processor(config.model)
        if processor is None:
            raise ValueError("Multimodal training requires a model image processor")
    model_files = setup_model_files(config.model, tokenizer, processor)
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt, model_files, resume=config.resume)

    # Initialize the model
    logger.info(f"Initializing model ({config.model})")
    t0 = time.perf_counter()
    from_checkpoint = config.model.init.type == "pretrained" and is_checkpoint_step_dir(config.model.name)
    loading_from_ckpt_later = checkpoint_step is not None or from_checkpoint
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
    logger.debug(f"Initialized model in {format_time(time.perf_counter() - t0)}")

    materialize_images = None
    if config.model.vlm is not None:
        if not getattr(model, "supports_packed_multimodal_training", False):
            raise ValueError("Packed multimodal training requires model support")
        materialize_images = partial(model.materialize_images, processor=processor)
    micro_batch_workers = WorkerPool(config.data.num_workers)

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
            model_name=config.model.name,
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
    elif from_checkpoint:
        ckpt_manager.load_weights(Path(config.model.name) / "trainer", model, [optimizer])
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

    # Only the file monitor stores annotations, so without it there is nothing to write.
    annotation_writer = (
        AnnotationWriter(parallel_dims, world, config.monitors.file.float_decimals)
        if config.annotations and config.monitors.file
        else None
    )

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    start_step = progress.step
    # One rank per data rank releases its micro batches' arrays.
    is_data_master = world.rank % (world.world_size // parallel_dims.get_mesh("dp").size()) == 0
    # A trainer restarted on its own trains the retained steps again, and inference already
    # serves the policies they produced: publish only what it has not finished publishing.
    published = -1
    if config.rollout_transport.retain and weight_sender is not None:
        published = latest_finished_broadcast(config.output_dir)
        logger.info(f"Policies up to v{published} are published; not broadcasting them again")
    max_peak_memory = 0.0
    controller = Controller(requires_step=False)
    control_server = ControlServer(controller, config.control if world.is_master else None, ckpt_manager.ckpt_dir)
    control_server.start()
    dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
    forward_backward = ForwardBackward(
        model,
        config,
        parallel_dims,
        rl_loss_fn,
        gradient_manager,
        micro_batch_workers,
        materialize_images,
        annotation_writer,
        maybe_record_function,
    )

    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step >= config.max_steps
        step_action = broadcast_step_action(controller, progress.step) if config.control is not None else None
        paused = step_action == "pause"
        is_last_step = is_last_step or paused

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # Broadcast the incoming policy (v{progress.step-1}) before waiting for its
        # rollouts so the trainer and inference pool join the same update lifecycle,
        # and so a broken broadcast path fails at startup instead of after the first
        # optimizer step.
        if progress.step == start_step and weight_sender is not None and progress.step - 1 > published:
            startup_version = progress.step - 1
            if world.is_master:
                prune_broadcasts_beyond(config.output_dir, startup_version)
            logger.info(f"Broadcasting startup policy weights (v{startup_version}) to inference engines")
            t0 = time.perf_counter()
            weight_sender.broadcast(model, startup_version)
            logger.debug(
                f"Broadcast startup policy weights (v{startup_version}) in {format_time(time.perf_counter() - t0)}"
            )

        # Warm up while the first batch is still being generated
        if progress.step == start_step and config.warmup:
            with maybe_record_function("warmup"):
                warmup_time = warmup(model, optimizer, config, forward_backward)
            asyncio.run(monitors.log({"time/warmup": warmup_time, "step": progress.step}, step=progress.step))

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
        dropped_micro_batches = 0
        if not config.data.fake:
            # Ranks that share a data rank train the same micro batches, so they must drop
            # the same ones even when only one of them failed to fetch.
            failed = torch.zeros(dataloader.dp_world_size, len(micro_batches), device="cuda")
            failed[dataloader.dp_rank, dataloader.failed_micro_batches] = 1
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            # An NCCL all-reduce returns once it is enqueued, not once every rank joined it.
            # Reading the result on the host waits for every rank's contribution, which each
            # rank makes only after its fetch: without this, a data master that finished
            # fetching first deleted the arrays while its CP peers were still fetching them.
            failed = failed.cpu()
            # Every rank has fetched its arrays now, so one rank per data rank frees them
            # (dummy micro batches may reuse another data rank's, hence after the world
            # all-reduce).
            if is_data_master:
                dataloader.release_payloads()
            # A step where no micro batch's arrays could be fetched anywhere is not a
            # transient failure (unreachable or mismatched env worker pools): stop instead of
            # training on nothing.
            if failed.numel() and bool(failed.all()):
                raise RuntimeError(
                    "Could not fetch the training arrays of any micro batch this step; check that "
                    "trainer ranks reach the env worker pools' payload URLs (see the trainer and env-worker-pool logs)"
                )
            dropped = failed[dataloader.dp_rank].nonzero().flatten().tolist()
            drop_micro_batches(micro_batches, dropped)
            dropped_micro_batches = int(failed.sum().item())
            if dropped:
                logger.warning(f"Dropped {len(dropped)} micro batches whose payloads could not be fetched")
        load_data_time = time.perf_counter() - load_data_start_time
        logger.debug(f"Loaded batch in {format_time(load_data_time)}")

        batch_size = len(micro_batches)
        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.perf_counter()
        perf_counter = get_perf_counter(model)

        tensor_stats, routing_stats, step_local_num_tokens, step_local_flops, grad_scale = forward_backward(
            micro_batches
        )
        # Free this batch before the next one is received and converted
        del micro_batches

        # Divide by max_scale and undo FSDP's per-rank averaging across dp_cp so the final
        # gradient is the true per-token mean over the global batch.
        if gradient_manager is None:
            scale_gradients_(None, model, grad_scale)

        # Optionally, clip the gradients
        grad_norm: torch.Tensor | None = None
        if config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(gradient_manager, model, config.optim.max_norm)

        # An async checkpoint must finish copying the state before the optimizer step mutates it
        ckpt_manager.wait_for_staging()

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Broadcast the model just produced (policy v{progress.step}). The final
        # broadcast keeps inference synchronized with the completed trainer.
        if weight_sender is None or progress.step <= published:
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

        # Count the step before checkpointing it, so a resume continues the counters from here.
        # The dp mesh excludes cp, whose ranks hold the same rows.
        global_counts = torch.tensor([step_local_num_tokens, step_local_flops], dtype=torch.float64, device="cuda")
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=parallel_dims.get_mesh("dp").get_group())
        num_tokens, num_flops = int(global_counts[0].item()), global_counts[1].item()
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size

        # Checkpoint the step we just finished (model = policy v{progress.step}).
        is_ckpt_step = bool(config.ckpt and config.ckpt.interval) and progress.step % config.ckpt.interval == 0
        # The last step is written once after the loop (final ckpt), so skip it here.
        if (is_ckpt_step or step_action == "checkpoint") and not is_last_step:
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
            if step_action == "checkpoint" or config.rollout_transport.retain:
                # The action and the release of retained rollouts both need the checkpoint on storage
                ckpt_manager.wait()
            dataloader.commit(progress.step, release=is_data_master)
            if step_action == "checkpoint":
                controller.mark_saved(progress.step)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        if is_moe_model:
            moe_stats, step_tokens_per_expert = get_global_moe_stats(model, routing_stats, ep_group, dp_cp_group)
            for name, values in moe_stats.items():
                tensor_stats.add(name, values)

        # Synchronize the tensor metrics across all steps and ranks
        tensor_metrics = tensor_stats.compute_stats()
        if is_moe_model:
            tensor_metrics.update(get_expert_load_stats(step_tokens_per_expert, dp_cp_group))

        throughput = perf_counter.get_step_tokens_per_second(num_tokens, forward_backward_time)
        mfu = perf_counter.get_step_mfu(num_flops, forward_backward_time)
        mfu_6dn = perf_counter.get_step_mfu(
            perf_counter.flop_counter.flops_per_token_6dn * num_tokens, forward_backward_time
        )
        peak_memory, non_torch_memory = get_peak_memory_across_ranks()
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
        if "ep_imbalance/rank/mean" in tensor_metrics:
            step_message += f" | EP Imbalance {tensor_metrics['ep_imbalance/rank/mean']:.4f}"
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
            "perf/non_torch_memory": non_torch_memory,
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
            "payload/dropped_micro_batches": dropped_micro_batches,
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
    if config.ckpt is not None or step_action is not None:
        logger.info(f"Saving final checkpoint at step {progress.step}")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        ckpt_manager.wait()
        dataloader.commit(progress.step, release=is_data_master)
        if step_action is not None:
            controller.mark_saved(progress.step)
        if not paused:
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
