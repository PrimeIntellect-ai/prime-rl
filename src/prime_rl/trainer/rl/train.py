from contextlib import nullcontext
import time
from copy import deepcopy
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from loguru import logger
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.weights import setup_weight_ckpt_manager
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.rl.loss import (
    shift_logits,
    selective_log_softmax,
    compute_entropy,
    compute_loss,
)
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    reshard_module,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    MemoryProfiler,
    OffloadedTensor,
    Tensors,
    copy_model_to_cpu,
    setup_torch_distributed,
    offload_model_to_cpu,
    wake_up_model_from_cpu,
    print_benchmark,
    get_response_lengths,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format


@clean_exit
@logger.catch(reraise=True)
def train(config: RLTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting RL trainer in {world}")

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Set precision
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)
    if config.model.cp > 1:
        raise ValueError(
            "CP is not supported for RL. No reason it shouldn't, we just didn't test it. If you need it, please open an issue."
        )

    # Initialize the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model})")
    model = setup_model(config.model, parallel_dims)
    tokenizer = setup_tokenizer(config.model)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    logger.info(f"Using `{config.loss.type}` loss ({config.loss})")

    optimizer = setup_optimizer(config.optim, model, parallel_dims.world_mesh["dp_shard_cp"])

    # Set up the learning rate scheduler
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Set up weight checkpoint manager
    logger.info(f"Initializing weight checkpoint manager ({config.weights})")
    weight_ckpt_manager = setup_weight_ckpt_manager(config.output_dir, config.weights, config.ckpt, config.async_level)
    assert weight_ckpt_manager is not None, "Weight checkpoint manager must be set on RL trainer"

    # Set up checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if config.ckpt and ckpt_manager is not None and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step {config.ckpt.resume_step}")
        ckpt_manager.load(model, [optimizer], scheduler, progress, step=config.ckpt.resume_step)
    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Optionally, initialize a model to compute logprobs
    logprob_model, tensor_offloaded_repository = None, {}
    if config.recompute_logprobs:
        # Initialize the logprob model
        tensor_offloaded_repository: dict[int, OffloadedTensor] = {}
        logger.info(f"Initializing logprob model ({config.model})")
        logprob_model = setup_model(config.model, parallel_dims)

        # Load async models from weights checkpoint if resuming from checkpoint
        if config.ckpt and config.ckpt.resume_step:
            for step in range(max(progress.step - config.async_level, 0), progress.step):
                logger.info(f"Initializing logprob model ({config.model}) for step {step}")
                model_name_or_path = (
                    config.model.name
                    if not (config.ckpt and config.ckpt.resume_step)
                    else weight_ckpt_manager._get_step_path(step).as_posix()
                )
                model_config = deepcopy(config.model)
                model_config.name = model_name_or_path
                logprob_model = setup_model(model_config, parallel_dims)
                tensor_offloaded_repository[step] = offload_model_to_cpu(logprob_model)

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    dataloader = DataLoader(config.output_dir, progress.step)
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake)

    logger.info(f"Starting training loop ({config.max_steps=})")
    is_first_step = True
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Save the weight checkpoint (if we are not at the first step, because no updates to the model have been made yet)
        save_weights_time = 0
        if progress.step > 0:
            save_weights_start_time = time.time()
            weight_ckpt_manager.save(model, tokenizer, step=progress.step)
            save_weights_time = time.time() - save_weights_start_time

        # Save the full checkpoint (if we are at an interval step and not at the first or last step)
        is_last_step = config.max_steps is not None and progress.step == config.max_steps
        save_ckpt_time = 0
        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

            # Maybe clean up old trainer checkpoints
            ckpt_manager.maybe_clean()

        # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        logger.info(f"Starting training step {progress.step}")
        step_start_time = time.time()

        # Offload the current model to CPU for logprob computation
        if logprob_model is not None:
            logger.debug(f"Offloading model for step {progress.step} to CPU for future logprob calculation")
            reshard_module(logprob_model)
            tensor_offloaded_repository[progress.step] = copy_model_to_cpu(model)

        # Wait for the batch to be available
        logger.info("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.time()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.time() - wait_for_batch_start_time
        logger.debug(f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds")

        # Load the training batch
        logger.debug("Loading batch")
        load_data_start_time = time.time()
        micro_batches = dataloader.get_batch()
        load_data_time = time.time() - load_data_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        # Optionally, compute the logprobs for the training batch
        compute_logprobs_time = 0
        num_micro_batches = len(micro_batches)
        recomputed_logprob_errors = [torch.ones_like(mb["logprobs"], device="cuda") for mb in micro_batches]
        if logprob_model is not None:
            compute_logprobs_start_time = time.time()
            og_infer_step = progress.step - config.async_level
            infer_step = max(og_infer_step, 0)
            logger.info(f"Recomputing logprobs with model weight checkpoint {infer_step}")

            # Wake up the logprob model from CPU
            wake_up_model_from_cpu(logprob_model, tensor_offloaded_repository[infer_step])
            if og_infer_step == infer_step:
                del tensor_offloaded_repository[infer_step]

            with torch.no_grad():
                for micro_step, micro_batch in enumerate(micro_batches):
                    input_ids = micro_batch["input_ids"].to("cuda")
                    position_ids = micro_batch["position_ids"].to("cuda")
                    loss_mask = micro_batch["loss_mask"].to("cuda")
                    logprobs = micro_batch["logprobs"].to("cuda")
                    temperature = micro_batch["temperature"]

                    # Compute the logprobs
                    logits = forward(logprob_model, input_ids, position_ids).float().contiguous()
                    shifted_logits = shift_logits(logits)
                    shifted_logits = shifted_logits / temperature
                    recomputed_logprobs = selective_log_softmax(shifted_logits, input_ids)

                    # Compute the recomputed logprob error
                    recomputed_logprob_error = torch.exp(recomputed_logprobs - logprobs)

                    micro_batch["logprobs"] = recomputed_logprobs.cpu()
                    recomputed_logprob_errors[micro_step] = recomputed_logprob_error

            # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
            # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
            reshard_module(logprob_model)
            offload_model_to_cpu(logprob_model)

            compute_logprobs_time = time.time() - compute_logprobs_start_time
            logger.debug(f"Recomputed logprobs in {compute_logprobs_time:.2f} seconds")

        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.time()
        micro_batch_size, seq_len = micro_batches[0]["input_ids"].shape
        batch_size = micro_batch_size * num_micro_batches

        # Normalize by the local number of unmasked tokens in the batch (per-batch length normalization)
        if config.loss.norm_type == "token":
            loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
        elif config.loss.norm_type == "sequence":
            loss_scale = batch_size

        logger.info(f"Starting forward and backward pass ({num_micro_batches=})")
        tensors = Tensors()  # Used to accumulate tensor statistics across micro-batches and ranks for logging
        for micro_step, micro_batch in enumerate(micro_batches):
            # we only all reduce at the last grad acc step
            model.set_requires_all_reduce(micro_step == len(micro_batches) - 1)

            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            old_logprobs = micro_batch["logprobs"].to("cuda")
            temperature = micro_batch["temperature"]
            micro_batch_size, seq_len = input_ids.shape

            # Forward pass
            with maybe_record_function("forward"):
                logits = forward(model, input_ids, position_ids).float().contiguous()
            shifted_logits = shift_logits(logits)
            shifted_logits = shifted_logits / temperature
            logprobs = selective_log_softmax(shifted_logits, input_ids)

            # Compute loss
            response_lengths = get_response_lengths(position_ids)
            loss, loss_tensors = compute_loss(
                logprobs=logprobs.squeeze().split(response_lengths),
                old_logprobs=old_logprobs.squeeze().split(response_lengths),
                advantages=advantages.squeeze().split(response_lengths),
                loss_mask=loss_mask.squeeze().split(response_lengths),
                loss_config=config.loss,
                loss_scale=loss_scale,
            )

            # Compute entropy
            with torch.no_grad():
                entropy = compute_entropy(shifted_logits)

            # Delete logits and shifted_logits before backward pass to avoid memory spike
            del logits, shifted_logits

            # Backward pass
            with maybe_record_function("backward"):
                loss.backward()

            # Add relevant tensors to tensor dict for logging purposes
            tensors["probs"].append(torch.exp(logprobs)[loss_mask].detach().to("cpu"))
            tensors["old_probs"].append(torch.exp(old_logprobs)[loss_mask].detach().to("cpu"))
            tensors["entropy"].append(entropy[loss_mask].detach().to("cpu"))
            tensors["recomputed_logprob_error"].append(
                recomputed_logprob_errors[micro_step][loss_mask].detach().to("cpu")
            )
            tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))

            if is_tt_moe_model(model):
                load_balance_stats = get_load_balance_stats(model)
                for k, v in load_balance_stats.items():
                    if v is not None:
                        tensors[k].append(v)

            # Add loss tensors to tensor dict for logging purposes
            for key, loss_tensor in loss_tensors.items():
                loss_tensor = loss_tensor.detach()[loss_mask.squeeze()].detach().to("cpu")
                tensors[key].append(loss_tensor)

            # Debug log with *local, micro step* stats
            micro_step_message = f"Micro Step {micro_step}/{len(micro_batches)} | Loss: {tensors['loss'][-1].mean().item():.4f} | Entropy: {tensors['entropy'][-1].mean().item():.4f} | Importance Ratio: {tensors['importance_ratio'][-1].mean().item():.4f}"
            if "max_vio" in tensors:
                micro_step_message += f" | Max Vio: {tensors['max_vio'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

        # Optionally, clip the gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm).full_tensor()

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        forward_backward_time = time.time() - forward_backward_start_time

        # TODO: Broadcast weight checkpoint via shardcast

        # Maybe clean up weight checkpoint
        weight_ckpt_manager.maybe_clean(progress.step)

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        tensor_stats = tensors.compute_stats()

        # Compute step metrics
        num_local_tokens = micro_batch_size * seq_len * num_micro_batches
        num_tokens = world.world_size * num_local_tokens
        batch_size = micro_batch_size * num_micro_batches
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        perf_counter = get_perf_counter(model, seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GiB

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {tensor_stats['loss/mean']:.4f} | Entropy: {tensor_stats['entropy/mean']:.4f} | Importance Ratio: {tensor_stats['importance_ratio/mean']:.4f} | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_stats:
            step_message += f" | Max Vio: {tensor_stats['max_vio/mean']:.4f}"
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/peak_memory": peak_memory,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "step": progress.step,
        }
        monitor.log(optim_metrics)

        # Log tensor stats
        tensor_stats["step"] = progress.step
        monitor.log(tensor_stats)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_batch": wait_for_batch_time,
            "time/load_data": load_data_time,
            "time/save_weights": save_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/compute_logprobs": compute_logprobs_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Log distributions to W&B table if enabled
        assert all(len(tensors) == 1 for tensors in tensors.values()), "Tensors must be lists of length 1"
        distributions = {key: tensors[key][0] for key in tensors.keys()}
        monitor.log_distributions(
            distributions=distributions,
            step=progress.step,
        )

        progress.step += 1
        is_first_step = False

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Log final (immutable) distributions to W&B table
    monitor.log_final_distributions()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(model, [optimizer], scheduler, progress, step=progress.step)
        ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("RL trainer finished!")

    # Optionally, print benchmark table
    if config.bench and world.is_master:
        print_benchmark(to_col_format(monitor.history))


def main():
    train(parse_argv(RLTrainerConfig))


if __name__ == "__main__":
    main()
