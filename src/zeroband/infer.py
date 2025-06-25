# (Jack): This is an umerged patch to fix a bug in vllm https://github.com/vllm-project/vllm/pull/19940
# This can be removed once the patch is merged and vllm is updated.
import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from toploc.utils import sha256sum
from vllm import LLM, SamplingParams, TokensPrompt

from zeroband.inference import envs
from zeroband.inference.config import Config as InferenceConfig
from zeroband.inference.logger import setup_logger
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import (
    all_reduce,
    patch_model_load,
    setup_comm,
    setup_hooks,
)
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.toploc2 import Toploc2Sampler
from zeroband.inference.utils import (
    compute_max_batch_size,
    filter_data_by_prompt_length,
    format_prompts,
    generate_target_lengths,
    get_inference_input_output_flops,
    reload_model_weights,
)
from zeroband.training.mp import EnvWrapper
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


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
        logger.info("Using toploc2 sampler")
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
    logger.info(f"Initializing sampling parameters ({config.sampling})")
    sampling_params = SamplingParams(**config.sampling.model_dump())

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
        logger.info(f"Auto-computed maximum batch size: {max_batch_size}")
        max_batch_size = int(max_batch_size * config.scale_factor)
        logger.info(f"Scaled maximum batch size by {config.scale_factor} to {max_batch_size}")

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
        if config.rl and real_step - ckpt_step > config.rl.async_level:
            logger.info(f"Required to reload model weights for step {ckpt_step} from {config.rl.ckpt_path}")
            ckpt_step = real_step - config.rl.async_level
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
            if seed is not None:
                sampling_params.seed = seed + real_step * 1_000_000  # 1M is needed to avoid collision from sampling.n

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

        generate_start_time = time.time()
        if config.contexts:
            # Convert tokenized prompts to TokensPrompt objects (prompt_id -> TokensPrompt)
            # TODO(jack): These can probably be lists and then we dont need the list dict sort thing going on at the end?
            unordered_token_prompts: dict[int, TokensPrompt] = {
                i: TokensPrompt(prompt_token_ids=prompt_token_ids) for i, prompt_token_ids in enumerate(tokenized_prompts)
            }
            # Save chunked request outputs (prompt_id -> RequestOutput)
            unordered_output_token_ids: dict[int, list[int]] = {i: [] for i in unordered_token_prompts.keys()}
            unordered_proofs: dict[int, bytes] = {i: b"" for i in unordered_token_prompts.keys()}

            max_model_len = llm.llm_engine.model_config.max_model_len
            assert max(config.contexts) <= max_model_len, "The final context should be less than the max model length"
            assert sorted(config.contexts) == config.contexts, "Contexts must be sorted"
            num_finished_sequences = 0
            for context in config.contexts:
                # Auto-compute the max batch size for the current context
                logger.info(f"Running at context {context}")
                local_max_batch_size = compute_max_batch_size(llm, max_model_len=context)
                if node:
                    max_batch_size = int(
                        all_reduce(node, torch.tensor(local_max_batch_size), config=config.parallel.pp, op=torch.min).item()
                    )
                else:
                    max_batch_size = local_max_batch_size
                logger.info(f"Auto-computed maximum batch size for context {context}: {max_batch_size}")

                # Micro-batch the sequences to not trigger cache eviction
                prompt_ids = list(unordered_token_prompts.keys())
                micro_batch_prompt_ids = [prompt_ids[i : i + max_batch_size] for i in range(0, len(prompt_ids), max_batch_size)]
                micro_batches = [
                    {prompt_id: unordered_token_prompts[prompt_id] for prompt_id in micro_batch_prompt_ids}
                    for micro_batch_prompt_ids in micro_batch_prompt_ids
                ]
                finish_sequence = context == max(config.contexts)  # Last context
                for mb_i, mb_unordered_tokens_prompts in enumerate(micro_batches):
                    # 1. Figure out the max_tokens to set
                    input_token_count = sum(len(token_prompt["prompt_token_ids"]) for token_prompt in mb_unordered_tokens_prompts.values())
                    cacheable_area = max_batch_size * context
                    max_tokens = (cacheable_area - input_token_count) // len(mb_unordered_tokens_prompts)

                    # TopLoc1 proofs do chunks of 32 tokens
                    # So we must have a multiple of 32 tokens
                    max_tokens = int(max_tokens / 32) * 32
                    assert max_tokens > 0, "Context must be larger than the max input tokens"
                    sampling_params.max_tokens = max_tokens

                    # 2. Generate
                    logger.info(
                        f"Generating {len(mb_unordered_tokens_prompts)} samples with context {context} for micro batch {mb_i + 1}/{len(micro_batches)} ({max_tokens} tokens)"
                    )
                    logger.info(f"Token prompt IDs: {list(mb_unordered_tokens_prompts.keys())}")
                    start_time = time.time()
                    request_outputs = llm.generate(list(mb_unordered_tokens_prompts.values()), sampling_params, use_tqdm=config.use_tqdm)
                    generation_time = time.time() - start_time
                    logger.info(f"Generated {len(request_outputs)} samples in {generation_time:.2f}s")

                    # Force generation of proofs for the remaining sequences that haven't reached max_len.
                    toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

                    # Save the TOPLOC proofs for the current chunk
                    toploc_cache.wait_for_proofs()
                    chunked_proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
                    for prompt_id, chunked_proof in zip(mb_unordered_tokens_prompts.keys(), chunked_proofs):
                        unordered_proofs[prompt_id] += chunked_proof
                    toploc_cache.reset_cache()

                    # Only keep unfinished sequences for the next iteration
                    finished_prompt_ids = []
                    for prompt_id, request_output in zip(mb_unordered_tokens_prompts.keys(), request_outputs):
                        # Note: This assumes that sampling.n == 1, else it might break
                        assert len(request_output.outputs) == 1, "Sampling.n must be 1"
                        output = request_output.outputs[0]
                        unordered_output_token_ids[prompt_id].extend(output.token_ids)
                        if (
                            output.finish_reason == "stop"
                            or finish_sequence
                            or len(request_output.prompt_token_ids) + len(output.token_ids) >= max_model_len
                        ):
                            # We remember which sequences finished, so we can pop remove them from the prompt pool
                            finished_prompt_ids.append(prompt_id)
                        else:
                            # Overwrite the token prompt so that we prefill the next chunk
                            unordered_token_prompts[prompt_id] = TokensPrompt(
                                prompt_token_ids=[*request_output.prompt_token_ids, *output.token_ids]
                            )
                    for prompt_id in finished_prompt_ids:
                        unordered_token_prompts.pop(prompt_id)

                    num_finished_sequences += len(finished_prompt_ids)
                    logger.info(f"Finished {len(finished_prompt_ids)} sequences at micro batch {mb_i} at context {context}")
                logger.info(f"Finished {num_finished_sequences} sequences at context {context}")

            # Order the request outputs by prompt_id to pretend like we generated all samples in one go
            # request_outputs = list(dict(sorted(unordered_request_outputs.items(), key=lambda x: x[0])).values())
            from vllm.outputs import CompletionOutput, RequestOutput

            request_outputs = []
            for prompt_id in unordered_output_token_ids.keys():
                request_outputs.append(
                    RequestOutput(
                        request_id="shouldnt matter",
                        prompt=None,
                        prompt_logprobs=None,
                        finished=True,
                        prompt_token_ids=tokenized_prompts[prompt_id],
                        outputs=[
                            CompletionOutput(
                                index=0,
                                token_ids=unordered_output_token_ids[prompt_id],
                                text=tokenizer.decode(unordered_output_token_ids[prompt_id], skip_special_tokens=True),
                                cumulative_logprob=None,
                                logprobs=None,
                            )
                        ],
                    )
                )
            proofs = list(dict(sorted(unordered_proofs.items(), key=lambda x: x[0])).values())
            assert len(request_outputs) == batch_size, "Number of request outputs must match batch size"
            assert len(proofs) == batch_size, "Number of proofs must match batch size"
        else:
            token_prompts: list[TokensPrompt] = [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in tokenized_prompts]
            request_outputs = llm.generate(token_prompts, sampling_params, use_tqdm=config.use_tqdm)

            # This generates proofs for the remaining sequences that haven't reached max_len.
            # We call here to give time for the proofs to be generated non-blocking in the background.
            toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

            # Compute proofs
            # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
            # Generate always adds requests to the engine in the order of the prompts.
            # And returns them in the sequence they were added.
            toploc_cache.wait_for_proofs()
            proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
            toploc_cache.reset_cache()

        generation_time = time.time() - generate_start_time

        # Compute progress metrics
        batch_problems = len(problems)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.success(f"Generated {batch_samples} samples for {batch_problems} problems in {generation_time:.2f}s")

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
        batch_tokens_per_second = batch_tokens / generation_time
        batch_samples_per_minute = batch_samples / generation_time * 60
        batch_avg_seq_length = batch_tokens / batch_size
        logger.info(
            f"Batch throughput: {batch_tokens_per_second:.2f} tokens/sec, {batch_samples_per_minute:.2f} samples/min ({batch_tokens} tokens in {generation_time:.2f}s, avg seq len: {batch_avg_seq_length:.1f})"
        )

        # Log performance metrics
        perf_metrics = {
            "performance/batch_tokens_per_second": batch_tokens_per_second,
            "performance/batch_samples_per_minute": batch_samples_per_minute,
            "performance/batch_avg_seq_length": batch_avg_seq_length,
            "step": real_step,
        }
        monitor.log(perf_metrics)

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

        work_submission = {
            "output/output_flops": sum(output_flops for _, output_flops in flop_counts) // config.parallel.pp.world_size,
            "output/input_flops": sum(input_flops for input_flops, _ in flop_counts) // config.parallel.pp.world_size,
            "output/save_path": save_path.as_posix(),
            "output/sha256": sha256,
            "output/step": real_step,
        }
        monitor.log(work_submission, exclude=["wandb"])

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
            config.rl.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.rl.async_level, 1),
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
