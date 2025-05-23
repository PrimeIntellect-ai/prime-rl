import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

# Setup logger (and environment variables before any other imports)
# ruff: noqa: I001
from zeroband.inference.logger import setup_logger

logger = setup_logger(tag="Inference")

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.distributed as dist
from datasets import load_dataset
from pydantic_config import parse_argv
from toploc.utils import sha256sum
from vllm import LLM, SamplingParams


from zeroband.inference import envs
from zeroband.inference.config import Config
from zeroband.inference.logger import get_logger
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import setup_pipeline
from zeroband.inference.rewards import compute_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.utils import fake_chat_template, generate_target_length_prompts, reload_model_weights
from zeroband.training.mp import EnvWrapper
from zeroband.utils.metrics import PrimeMetric


def inference(config: Config):
    # Get inference logger
    logger = get_logger()
    logger.info("Starting inference")

    # Log configuration
    for field, value in config.model_dump().items():
        logger.info(f"{field}: {value}")

    if config.io.cleanup and config.io.data_dir is not None:
        logger.info(f"Cleaning data directory {config.io.data_dir}")
        shutil.rmtree(config.io.data_dir, ignore_errors=True)

    # Initialize prime metrics
    prime_metric = PrimeMetric(disable=config.prime_log_freq is None, period=config.prime_log_freq)

    # Initialize vLLM and get tokenizer
    logger.info(f"Initializing vLLM v{os.environ.get('VLLM_USE_V1')} for model {config.model.name}")
    llm = LLM(
        model=config.model.name,
        tensor_parallel_size=config.parallel.tp,
        max_seq_len_to_capture=config.model.max_model_len,
        max_model_len=config.model.max_model_len,
        quantization=config.model.quant,
        enforce_eager=config.model.enforce_eager,
        disable_async_output_proc=True,  # We have an off by 1 error in toploc without this flag when cuda graph padding is enabled.
        download_dir=config.io.cache_dir,
        dtype="bfloat16" if config.model.dtype == "bf16" else torch.float32,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(**config.sampling.model_dump())

    # Create communication for pipeline
    if config.parallel.pp.world_size > 1:
        logger.info(f"Setting up pipeline rank {config.parallel.pp.rank} in world size {config.parallel.pp.world_size}")
        setup_pipeline(
            llm=llm,
            rank=config.parallel.pp.rank,
            world_size=config.parallel.pp.world_size,
            iroh_seed=config.parallel.pp.iroh_seed,
            iroh_peer_id=config.parallel.pp.iroh_peer_id,
        )

    # Load  dataset
    logger.info(f"Loading dataset {config.data.name}")
    dataset = load_dataset(config.data.name, split="train")
    logger.info(f"Loaded dataset {config.data.name} with {len(dataset)} samples")

    # Optionally shuffle dataset
    if envs.NODE_ADDRESS is not None:
        # We dont shuffle here because we shuffle reproducibly in the sampling loop.
        assert config.seed is None, "Seed is not supported when NODE_ADDRESS is set"
        assert envs.RANK == 0, "DP is not supported when NODE_ADDRESS is set"
        node_address_int = int(envs.NODE_ADDRESS, 16)
        logger.info(f"Seeding with {node_address_int} ({envs.NODE_ADDRESS})")
    else:
        # Seed the dataset with a random number
        seed = config.seed + envs.RANK if config.seed is not None else None
        logger.info(f"Seeding with {seed}")
        generator = np.random.default_rng(seed)
        node_address_int = None

    logger.info(f"Shuffling dataset {config.data.name}")
    dataset = dataset.shuffle(generator=generator)

    # Optionally filter dataset
    if config.data.filtering:
        logger.info(f"Filtering dataset {config.data.name} with {config.data.filtering.solve_rate_field}")
        dataset = dataset.filter(
            lambda x: x[config.data.filtering.solve_rate_field] >= config.data.filtering.min_solve_rate
            and x[config.data.filtering.solve_rate_field] <= config.data.filtering.max_solve_rate
        )
        logger.info(f"Filtered dataset {config.data.name}, {len(dataset)} samples remaining")

    # Setup TOPLOC cache and register hook to add hidden states to it
    num_batch_samples = config.batch_size * config.sampling.n
    hidden_size = llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size
    toploc_cache, _ = setup_toploc_cache(
        llm,
        config.toploc,
        max_seqs=num_batch_samples,
        hidden_size=hidden_size,
    )

    step = 0
    ckpt_step = 0
    if config.io.checkpoint_start_dir is not None:
        checkpoint_dir = Path(config.io.checkpoint_start_dir)
        checkpoint_file = checkpoint_dir / "model.safetensors"
        logger.info(f"Resuming from step {ckpt_step} at {checkpoint_file}")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist")
        llm = reload_model_weights(llm, checkpoint_file)
        logger.info(f"Reloaded model weights from {checkpoint_file}")
        ckpt_step = int(checkpoint_file.name.split("_")[-1])
        step = ckpt_step

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_tokens = 0
    max_samples = config.max_samples or len(dataset)

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        logger.info("---")
        logger.info(f"Step {step}")
        logger.info(f"Generating: {config.sampling.n} samples for {config.batch_size} problems")

        # Get current step from endpoint
        if config.step_endpoint is not None:
            logger.info(f"Getting current step from endpoint {config.step_endpoint}")
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_step = requests.get(config.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_step != step:
                step = new_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        # Reload model weights if we are past the async level
        if config.io.checkpoint_dir is not None and step - ckpt_step > config.async_level:
            ckpt_step = step - config.async_level
            attempt_count = 0
            while True:
                stable_file = Path(config.io.checkpoint_dir) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights for step {ckpt_step} from {config.io.checkpoint_dir} ckpt {ckpt_step}")
                    ckpt_file = Path(config.io.checkpoint_dir) / f"step_{ckpt_step}/model.safetensors"
                    llm = reload_model_weights(llm, ckpt_file)
                    total_problems = 0
                    total_tokens = 0
                    logger.info(f"Reloaded model weights for step {ckpt_step} from {ckpt_file}")
                    break
                if attempt_count % 30 == 0:
                    logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint")
                time.sleep(1)
                attempt_count += 1

        # Randomize sampling indices in production
        if node_address_int is not None:
            # TODO: What if we have multiple sample per real step?
            # Its impossible right now but we need to fix this if accept counter is used.

            # We reseed the generator here to make the sampling reproducible at each step.
            # This would work even if the node restarts and resumes from the current step.
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + step)
            indices = generator.integers(0, len(dataset), config.batch_size)
        else:
            indices = range(i, min(i + config.batch_size, len(dataset)))

        # Sample batch from dataset
        logger.debug(f"Sampling batch with indices {indices[:5]}")
        batch = dataset.select(indices)

        # Prepare prompts
        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]
        length_prompt_additions, target_lengths = generate_target_length_prompts(config.len_reward, len(batch))

        # Assume verification_info is stored as a JSON string in the dataset.
        verification_infos = [json.loads(item["verification_info"]) for item in batch]
        for target_length, verification_info in zip(target_lengths, verification_infos):
            verification_info["target_length"] = target_length
        task_types = [item["task_type"] for item in batch]

        if config.len_reward:
            if config.len_reward.length_prompt_location == "system_prompt":
                messages = [
                    [
                        {"role": "system", "content": length_prompt},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": "<think>\n"},
                    ]
                    for item, length_prompt in zip(batch, length_prompt_additions)
                ]
            else:
                messages = [
                    [{"role": "user", "content": item["prompt"] + length_prompt}, {"role": "assistant", "content": "<think>\n"}]
                    for item, length_prompt in zip(batch, length_prompt_additions)
                ]
        else:
            messages = [
                [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}]
                for item, length_prompt in zip(batch, length_prompt_additions)
            ]

        # Apply chat template
        if tokenizer.chat_template:
            prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

            # Remove <think> from prompts
            if config.model.name != "Qwen/QwQ-32B":
                for i, p in enumerate(prompts):
                    prompts[i] = p.replace("<｜begin▁of▁sentence｜>", "")
        else:
            prompts = fake_chat_template(messages)

        # Generating
        batch_start = time.time()
        request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_end = time.time()

        # Dropping like this isnt ideal. But in practice, we shouldnt have any prompts that are too long.
        request_outputs = [req for req in request_outputs if len(req.outputs[0].token_ids) > 0]
        if len(request_outputs) != len(prompts):
            logger.warning(f"{len(prompts) - len(request_outputs)} prompts were filtered out because they were too long")

        # This generates proofs for the remaining sequences that haven't reached max_len.
        # We call here to give time for the proofs to be generated non-blocking in the background.
        toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

        # Calculate batch throughput and average sequence length
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        batch_throughput = batch_tokens / (batch_end - batch_start)
        avg_sequence_length = batch_tokens / num_batch_samples

        # Calculate overall tokens
        total_tokens += batch_tokens

        logger.info(f"Generated {len(request_outputs)} samples ({batch_tokens} tokens) in {batch_end - batch_start:.2f}s")
        logger.info(f"Batch throughput: {batch_throughput:.2f} tok/sec")
        logger.info(f"Average sequence length  {avg_sequence_length:.1f}")

        # Compute proofs
        # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
        # Generate always adds requests to the engine in the order of the prompts.
        # And returns them in the sequence they were added.
        toploc_cache.wait_for_proofs()
        proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
        toploc_cache.reset_cache()

        # Compute rewards and advantages
        start = time.time()
        request_rewards = compute_rewards(request_outputs, verification_infos, task_types, config.len_reward)
        logger.info(f"Computed rewards and advantages in in {time.time() - start:.2f}s")

        table = get_parquet_table(
            request_outputs,
            request_rewards,
            proofs,
            ckpt_step,
            target_lengths,
        )

        # Write step outputs to parquet file
        step_dir = Path(config.io.data_dir) / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        step_file = f"{step_dir}/{uuid.uuid4()}.parquet"
        pq.write_table(table, step_file)
        logger.info(f"Saved {num_batch_samples} output samples to {step_file}")

        # Compute and log file sha
        file_sha = sha256sum(step_file)
        logger.info(f"Logged output file with SHA256 {file_sha or 'NA'}")
        prime_metric.log_prime({"file_sha": file_sha, "file_name": step_file})

        # Log metrics to dashboard
        total_problems += len(prompts)
        metric = {"dashbord-progress/total": total_problems, f"dashbord-progress/{config.data.name}": total_tokens}
        prime_metric.log_prime(metric)

        # Increment step counter
        step += 1

        if config.max_steps is not None and step > config.max_steps:
            logger.info(f"Reached max step {config.max_steps}, stopping inference")
            break

    # Manually destroy vLLM process group to avoid warnings
    dist.destroy_process_group()


def main(config: Config) -> list[mp.Process]:
    processes = []
    from zeroband.inference import envs as inference_envs

    if config.parallel.dp > 1:
        if config.parallel.tp == "auto":
            assert torch.cuda.device_count() % config.parallel.dp == 0, "Number of GPUs must be divisible by DP"
            config.parallel.tp = torch.cuda.device_count() // config.parallel.dp
        gpu_ids = inference_envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.parallel.tp] for i in range(0, len(gpu_ids), config.parallel.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            envs = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "RANK": str(rank), "LOCAL_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, envs), args=(config,))
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
    # mp.set_start_method("spawn")
    config = Config(**parse_argv())  # type: ignore

    if config.step_endpoint is not None:
        current_step = requests.get(config.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.io.checkpoint_dir,
            config.async_level + 1,
            # TODO: maybe +1 because we most likely wont download the current step in time?
            # We could deadlock though.
            max(current_step - config.async_level, 1),
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
