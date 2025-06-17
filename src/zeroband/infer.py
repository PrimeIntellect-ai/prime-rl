import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
import uuid

# Import environment before any other imports
# ruff: noqa: I001
from zeroband.inference import envs

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from datasets import load_dataset
from toploc.utils import sha256sum
from vllm import LLM, SamplingParams, TokensPrompt
from huggingface_hub import snapshot_download

from zeroband.utils.pydantic_config import parse_argv
from zeroband.inference.config import Config as InferenceConfig
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import all_reduce, patch_model_load, setup_comm, setup_hooks
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.toploc2 import Toploc2Sampler
from zeroband.utils.monitor import setup_monitor
from zeroband.inference.utils import (
    filter_data_by_prompt_length,
    reload_model_weights,
    compute_max_batch_size,
    get_inference_input_output_flops,
    generate_target_lengths,
    format_prompts,
)
from zeroband.training.mp import EnvWrapper
from zeroband.utils.utils import clean_exit
from zeroband.inference.logger import setup_logger


@clean_exit
def inference(config: InferenceConfig):
    # Initialize the logger
    dp_rank = int(os.environ.get("DP_RANK", 0))
    logger = setup_logger(config.log, parallel_config=config.parallel, dp_rank=dp_rank)
    logger.info("Starting inference")

    # Optionally, clean the rollout path
    if config.clean_rollout_path and config.rollout_path is not None:
        logger.info(f"Cleaning rollout path ({config.rollout_path})")
        shutil.rmtree(config.rollout_path, ignore_errors=True)

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initialize metrics
    monitor = setup_monitor(config.monitor, config.task_id, config)

    # Patch vLLM's model loading to load model shard
    patch_model_load(config=config.parallel.pp)

    # Initialize model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm = LLM(
        model=config.model.name,
        dtype=config.model.dtype,
        kv_cache_dtype=config.model.kv_cache_dtype,
        max_seq_len_to_capture=config.model.max_model_len,
        max_model_len=config.model.max_model_len,
        quantization=config.model.quantization,
        enforce_eager=config.model.enforce_eager,
        device=config.model.device,
        tensor_parallel_size=config.parallel.tp,
        disable_async_output_proc=True,  # We have an off by 1 error in toploc without this flag when cuda graph padding is enabled.
        enable_chunked_prefill=False,  # This is required for toploc2 because chunked prefill seems to allow len(seq_groups) != len(selected_token_indices) which is unexpected
        seed=config.seed,
    )
    if config.toploc.enable_toploc2:
        llm.llm_engine.model_executor.driver_worker.model_runner.sampler = Toploc2Sampler()
    tokenizer = llm.get_tokenizer()
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Initialize dataset
    logger.info(f"Initializing dataset (name={config.data.name}, split={config.data.split})")
    start_time = time.time()
    dataset = load_dataset(config.data.name, split=config.data.split)

    if not config.rewards.compute_reward:
        logger.info("Reward computation is disabled, setting task_type to null_reward")
        dataset = dataset.map(lambda _: {"task_type": "null_reward"})

    logger.success(f"Initialized dataset with {len(dataset):,} problems in {time.time() - start_time:.2f}s")

    # Optionally shuffle dataset
    if config.group_id is not None:
        # We dont shuffle here because we shuffle reproducibly in the sampling loop.
        assert config.seed is None, "Seed is not supported when group ID is set"
        assert config.parallel.dp == 1, "DP is not supported when group ID is set"
        node_address_int = int(config.group_id, 16)
        logger.info(f"Seeding with {node_address_int} ({config.group_id})")
    else:
        # Seed the dataset with a random number
        seed = config.seed + int(os.environ.get("DP_RANK", 0)) if config.seed is not None else None
        generator = np.random.default_rng(seed)
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(generator=generator)
        node_address_int = None

    # Optionally, filter out prompts that are too long
    if config.data.max_prompt_len:
        logger.info(f"Filtering out prompts with more than {config.data.max_prompt_len} tokens")
        start_time = time.time()
        dataset = filter_data_by_prompt_length(dataset, config.data.max_prompt_len, tokenizer)
        logger.success(f"Filtered long prompts in {time.time() - start_time:.2f}s - {len(dataset)} samples remaining")

    # Optionally, filter dataset for samples within difficulty range
    if config.data.difficulty_filtering:
        logger.info(
            f"Filtering dataset for difficulty in [{config.data.difficulty_filtering.min_solve_rate}, {config.data.difficulty_filtering.max_solve_rate}]"
        )
        dataset = dataset.filter(
            lambda x: x[config.data.difficulty_filtering.solve_rate_field] >= config.data.difficulty_filtering.min_solve_rate
            and x[config.data.difficulty_filtering.solve_rate_field] <= config.data.difficulty_filtering.max_solve_rate
        )

    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({config.sampling} seed={config.seed})")
    sampling_params = SamplingParams(**config.sampling.model_dump(), seed=config.seed)

    # Setup pipeline parallel communication and hook
    node = setup_comm(config.parallel.pp)
    setup_hooks(llm, config.parallel.pp, node)

    # Compute the maximum batch size
    max_batch_size = config.max_batch_size
    if max_batch_size == "auto":
        # Automatically compute the maximum batch size
        logger.info("Auto-computing maximum batch size")
        local_max_batch_size = compute_max_batch_size(llm)
        max_batch_size = all_reduce(node, torch.tensor(local_max_batch_size), config=config.parallel.pp, op=torch.min).item()

    logger.info(f"Using maximum batch size: {max_batch_size}")

    # Throw an error if the batch cannot fit number of samples to generate per problem.
    # TODO(Mika): Circumvent this assertion by distribtuting parallel samples across multiple batches
    if config.sampling.n > max_batch_size:
        raise ValueError(f"Sampling.n ({config.sampling.n}) must be less than or equal to max_batch_size ({max_batch_size})")

    # Compute the true batch size
    problems_per_batch = max_batch_size // config.sampling.n
    batch_size = problems_per_batch * config.sampling.n
    logger.info(
        f"Problems per batch: {max_batch_size} // {config.sampling.n} = {problems_per_batch}, batch size: {problems_per_batch} * {config.sampling.n} = {batch_size} (missing: {max_batch_size % config.sampling.n})"
    )

    # Setup TOPLOC
    hidden_size = llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size
    toploc_cache, _ = setup_toploc_cache(
        llm,
        pp_config=config.parallel.pp,
        disable=not config.toploc.enable_toploc1,
        max_seqs=batch_size,
        hidden_size=hidden_size,
        topk=config.toploc.topk,
    )

    ckpt_step = 0
    real_step = config.start_step
    if config.rl and config.rl.ckpt_start_path is not None:
        logger.info(f"Resuming from checkpoint {config.rl.ckpt_start_path}")
        path = Path(config.rl.ckpt_start_path)
        path_file = path / "model.safetensors"
        if not path_file.exists():
            raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
        ckpt_step = int(path.name.split("_")[-1])
        logger.info(f"Resuming from step {ckpt_step} at {path_file}")
        llm = reload_model_weights(llm, path_file)
        real_step = ckpt_step

    # Check if we should resume from step_path file
    if config.step_path is not None and config.step_path.exists():
        try:
            saved_step = int(config.step_path.read_text().strip())
            logger.info(f"Found existing step file at {config.step_path} with step {saved_step}")
            real_step = saved_step
            logger.info(f"Resuming from step {real_step} (loaded from {config.step_path})")
        except (ValueError, IOError) as e:
            logger.warning(f"Failed to read step from {config.step_path}: {e}")

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_samples = 0
    total_tokens = 0

    dataset_offset = 0
    while True:
        if config.rl and config.rl.step_endpoint is not None:
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_real_step = requests.get(config.rl.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.rl.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_real_step != real_step:
                real_step = new_real_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        logger.info(f"Inference step {real_step} (Checkpoint step: {ckpt_step})")
        if config.rl and real_step - ckpt_step > config.rl.max_async:
            logger.info(f"Required to reload model weights for step {ckpt_step} from {config.rl.ckpt_path}")
            ckpt_step = real_step - config.rl.max_async
            attempt_count = 0
            while True:
                stable_file = Path(config.rl.ckpt_path) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights for step {ckpt_step} from {stable_file}")
                    llm = reload_model_weights(llm, Path(config.rl.ckpt_path) / f"step_{ckpt_step}/model.safetensors")
                    total_problems = 0
                    total_tokens = 0
                    logger.success(f"Reloaded model weights for step {ckpt_step} from {stable_file}")
                    break
                if attempt_count % 30 == 0:
                    logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint")
                time.sleep(1)
                attempt_count += 1

        if config.step_path is not None:
            logger.info(f"Writing current inference step ({real_step}) to {config.step_path}")
            if not config.step_path.exists():
                config.step_path.parent.mkdir(parents=True, exist_ok=True)
            config.step_path.write_text(str(real_step))

        # Get batch
        if node_address_int is not None:
            # TODO: What if we have multiple sample per real step?
            # Its impossible right now but we need to fix this if accept counter is used.

            # We reseed the generator here to make the sampling reproducible at each step.
            # This would work even if the node restarts and resumes from the current step.
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + real_step)
            indices = generator.integers(0, len(dataset), problems_per_batch)
            sampling_params.seed = int(generator.integers(2**32))
        else:
            # Use modulo to cycle through the dataset instead of terminating
            indices = [(dataset_offset + j) % len(dataset) for j in range(problems_per_batch)]

        logger.debug(f"Sampling batch with indices [{' '.join(map(str, indices[:3]))}...{' '.join(map(str, indices[-3:]))}]")
        problems = dataset.select(indices)

        verification_infos = [json.loads(item["verification_info"]) for item in problems]
        task_types = [item["task_type"] for item in problems]
        prompts = [item["prompt"] for item in problems]

        target_lengths = generate_target_lengths(config.rewards.len_reward, len(prompts))
        for target_length, verification_info in zip(target_lengths, verification_infos):
            verification_info["target_length"] = target_length

        # Get tokenized prompts as BatchEncoding
        tokenized_prompts = format_prompts(
            prompts,
            target_lengths,
            config.rewards.len_reward,
            tokenizer=tokenizer,
            enable_thinking=config.model.enable_thinking,
            tokenize=True,
        )

        # Convert BatchEncoding to TokensPrompt objects
        token_prompts = [TokensPrompt(prompt_token_ids=input_ids) for input_ids in tokenized_prompts]

        logger.info(f"Generating {len(token_prompts)} samples for {len(problems)} problems")
        start_time = time.time()
        request_outputs = llm.generate(token_prompts, sampling_params, use_tqdm=False)
        end_time = time.time()

        # Dropping like this isn't ideal. But in practice, we shouldn't have any prompts that are too long.
        request_outputs = [req for req in request_outputs if len(req.outputs[0].token_ids) > 0]
        if len(request_outputs) != len(prompts):
            logger.warning(f"{len(prompts) - len(request_outputs)} prompts were filtered out because they were too long")

        # This generates proofs for the remaining sequences that haven't reached max_len.
        # We call here to give time for the proofs to be generated non-blocking in the background.
        toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

        # Compute progress metrics
        batch_problems = len(problems)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.success(f"Generated {batch_samples} samples for {batch_problems} problems in {end_time - start_time:.2f}s")

        # Print example
        first_prompt = tokenizer.decode(request_outputs[0].prompt_token_ids)
        first_completion = tokenizer.decode(request_outputs[0].outputs[0].token_ids)
        logger.debug(f"Showing example (first completion):\n{first_prompt}{first_completion}")

        # Log progress metrics
        progress_metrics = {
            "progress/batch_problems": batch_problems,
            "progress/batch_samples": batch_samples,
            "progress/batch_tokens": batch_tokens,
            "step": real_step,
        }
        monitor.log(progress_metrics)

        # Compute performance metrics
        batch_tokens_per_second = batch_tokens / (end_time - start_time)
        batch_samples_per_minute = batch_samples / (end_time - start_time) * 60
        batch_avg_seq_length = batch_tokens / batch_size
        logger.info(
            f"Batch throughput: {batch_tokens_per_second:.2f} tokens/sec, {batch_samples_per_minute:.2f} samples/min ({batch_tokens} tokens in {end_time - start_time:.2f}s, avg seq len: {batch_avg_seq_length:.1f})"
        )

        # Log performance metrics
        perf_metrics = {
            "performance/batch_tokens_per_second": batch_tokens_per_second,
            "performance/batch_samples_per_minute": batch_samples_per_minute,
            "performance/batch_avg_seq_length": batch_avg_seq_length,
            "step": real_step,
        }
        monitor.log(perf_metrics)

        # Compute proofs
        # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
        # Generate always adds requests to the engine in the order of the prompts.
        # And returns them in the sequence they were added.
        toploc_cache.wait_for_proofs()
        proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
        toploc_cache.reset_cache()

        # Compute and log rewards and advantages
        logger.info("Computing rewards and advantages")
        request_rewards = compute_vllm_rewards(request_outputs, verification_infos, task_types, config.rewards)
        batch_rewards = sum(sum(r.reward for r in req.rewards) for req in request_rewards) / batch_samples
        logger.info(f"Average reward of the batch: {batch_rewards:.2f}")
        monitor.log({"rewards/batch_rewards": batch_rewards, "step": real_step})

        if sampling_params.seed is not None:
            sampling_seeds = [sampling_params.seed + i for i in range(sampling_params.n)] * problems_per_batch
        else:
            sampling_seeds = [None] * batch_samples

        # Get parquet table
        table = get_parquet_table(
            request_outputs,
            request_rewards,
            prompts,
            proofs,
            ckpt_step,
            target_lengths,
            problems,
            enable_logprobs=config.sampling.logprobs is not None,
            seeds=sampling_seeds,
            temperature=sampling_params.temperature,
        )

        # Save outputs to parquet file
        step_path = Path(config.rollout_path) / f"step_{real_step}"
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"{uuid.uuid4()}.parquet"
        logger.info(f"Saving batch outputs to {save_path}")
        pq.write_table(table, save_path)

        # Log file metadata
        sha256 = sha256sum(save_path)
        flop_counts = [
            get_inference_input_output_flops(config.model.name, len(input_tokens), len(output_tokens))
            for input_tokens, output_tokens in zip(table.column("input_tokens").to_pylist(), table.column("output_tokens").to_pylist())
        ]

        outputs_metrics = {
            "output/output_flops": sum(output_flops for _, output_flops in flop_counts) // config.parallel.pp.world_size,
            "output/input_flops": sum(input_flops for input_flops, _ in flop_counts) // config.parallel.pp.world_size,
            "step": real_step,
        }
        monitor.log(outputs_metrics)
        monitor.log({"output/save_path": save_path.as_posix(), "output/sha256": sha256, "step": real_step}, exclude=["wandb"])

        real_step += 1

        if config.max_steps is not None and real_step > config.max_steps:
            logger.info(f"Reached max steps {config.max_steps}, stopping inference")
            break

        dataset_offset += problems_per_batch

    logger.success(f"Inference finished! Generated {total_samples} samples for {total_problems} problems")


def main(config: InferenceConfig) -> list[mp.Process]:
    processes = []

    if config.parallel.dp > 1:
        if config.parallel.tp == "auto":
            assert torch.cuda.device_count() % config.parallel.dp == 0, "Number of GPUs must be divisible by DP"
            config.parallel.tp = torch.cuda.device_count() // config.parallel.dp
        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.parallel.tp] for i in range(0, len(gpu_ids), config.parallel.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, env), args=(config,))
            processes.append(process)
    else:
        if config.parallel.tp == "auto":
            config.parallel.tp = torch.cuda.device_count()
        inference(config)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")

    config = parse_argv(InferenceConfig)

    if config.rl and config.rl.step_endpoint is not None:
        current_step = requests.get(config.rl.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        assert config.rl is not None, "RL config is required when SHARDCAST_SERVERS is set"
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rl.ckpt_path,
            config.rl.max_async + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.rl.max_async, 1),
        )
    else:
        shardcast_process = None

    try:
        main(config)

    finally:
        if shardcast_process is not None:
            import os
            import signal

            # SIGTERM is not working, so we use SIGKILL
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()
