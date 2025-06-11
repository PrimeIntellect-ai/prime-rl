import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001
from zeroband.inference import envs

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.distributed as dist
import verifiers as vf
from datasets import load_dataset
from pydantic_config import parse_argv
from toploc.utils import sha256sum
from vllm import LLM
from openai import OpenAI

from zeroband.inference.config import Config
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import all_reduce, patch_model_load, setup_comm, setup_hooks
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.utils.monitor import setup_monitor
from zeroband.inference.utils import (
    filter_data_by_prompt_length,
    reload_model_weights,
    compute_max_batch_size,
    get_inference_input_output_flops,
)
from zeroband.inference.openai_wrapper import MockOpenAIServer
from zeroband.training.mp import EnvWrapper
from zeroband.utils.logger import get_logger

# Global logger
logger = get_logger("INFER")


def inference(config: Config):
    # Initialize the logger
    logger.info("Starting inference")

    # Log relevant configuration
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Parallelism: TP={config.tp}, DP={config.dp}, PP={config.pp.world_size}")

    if config.clean_output_path and config.output_path is not None:
        logger.info(f"Cleaning output path {config.output_path}")
        shutil.rmtree(config.output_path, ignore_errors=True)

    # Initialize metrics
    monitor = setup_monitor(config.monitor)

    # Patch vLLM's model loading to load model shard
    patch_model_load(config=config.pp)

    # Initialize vLLM and get tokenizer
    logger.info(
        f"Initializing vLLM for {config.model_name} (max_model_len={config.max_model_len}, enforce_eager={config.enforce_eager}, dtype={config.dtype}, quant={config.quant})"
    )
    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tp,  # type: ignore
        max_seq_len_to_capture=config.max_model_len,  # type: ignore
        max_model_len=config.max_model_len,
        quantization=config.quant,
        enforce_eager=config.enforce_eager,
        disable_async_output_proc=True,  # We have an off by 1 error in toploc without this flag when cuda graph padding is enabled.
        download_dir=config.download_dir,
        dtype="bfloat16" if config.dtype == "bf16" else torch.float32,  # type: ignore
    )
    tokenizer = llm.get_tokenizer()
    # sampling_params = SamplingParams(**config.sampling.model_dump())

    # Setup pipeline parallel communication
    node = setup_comm(config.pp)

    # Setup pipeline parallel hooks
    setup_hooks(llm, config.pp, node)

    # Compute the maximum batch size
    batch_size = config.batch_size
    if batch_size == "auto":
        # Automatically compute the maximum batch size
        local_batch_size = compute_max_batch_size(llm)
        batch_size = all_reduce(node, torch.tensor(local_batch_size), config=config.pp, op=torch.min).item()
        logger.info(f"Auto-computed batch size: {batch_size}")

    # Start MockOpenAI server
    base_port = 8000
    dp_rank = int(os.environ.get("DP_RANK", 0))
    port = base_port + dp_rank

    logger.info(f"Starting MockOpenAI server on port {port}")
    # Create mock config for MockOpenAI server with required fields
    from types import SimpleNamespace

    mock_config = SimpleNamespace(batch_size=batch_size, max_wait_time=1.0, model_name=config.model_name)
    oai_server = MockOpenAIServer(llm, tokenizer, mock_config)
    oai_server.start(port=port)

    # Create OpenAI client
    import httpx

    client_config = oai_server.get_client_config(port=port)
    openai_client = OpenAI(
        base_url=client_config["base_url"],
        api_key=client_config["api_key"],
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=int(batch_size)),
            timeout=httpx.Timeout(300.0),  # 300 second timeout
        ),
    )

    # Throw an error if the batch size is too small for the number of samples to generate per problem
    if config.sampling.n > batch_size:
        raise ValueError(f"Sampling.n ({config.sampling.n}) must be less than or equal to batch_size ({batch_size})")

    # Initialize environment
    train_dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {"question": x["text"], "answer": x["text"][::-1]}
    )
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"""Reverse the given text.

    Respond in the following format:
    {parser.get_format_str()}"""

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        """

        def lcs_ratio(x: str, y: str) -> float:
            """
            Return the longest common subsequence ratio of x and y.
            """
            from difflib import SequenceMatcher

            return SequenceMatcher(None, x, y).ratio()

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        client=openai_client,
        model=config.model_name,
    )

    # Load dataset
    dataset = env.get_dataset()
    logger.info(f"Loaded dataset {config.dataset} with {len(dataset):,} problems")

    # Optionally shuffle dataset
    if envs.PRIME_GROUP_ID is not None:
        # We dont shuffle here because we shuffle reproducibly in the sampling loop.
        assert config.seed is None, "Seed is not supported when PRIME_GROUP_ID is set"
        assert os.environ.get("DP_RANK") is None, "DP is not supported when PRIME_GROUP_ID is set"
        node_address_int = int(envs.PRIME_GROUP_ID, 16)
        logger.info(f"Seeding with {node_address_int} ({envs.PRIME_GROUP_ID})")
    else:
        # Seed the dataset with a random number
        seed = config.seed + int(os.environ.get("DP_RANK", 0)) if config.seed is not None else None
        generator = np.random.default_rng(seed)
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(generator=generator)
        node_address_int = None

    if config.max_prompt_len:
        dataset = filter_data_by_prompt_length(dataset, config.max_prompt_len, tokenizer)
        logger.info(f"✨ Removed long prompts - {len(dataset)} samples remaining")

    # Optionally filter dataset
    if config.difficulty_filtering:
        logger.info(
            f"Filtering dataset for difficulty in [{config.difficulty_filtering.min_solve_rate}, {config.difficulty_filtering.max_solve_rate}]"
        )
        dataset = dataset.filter(
            lambda x: x[config.difficulty_filtering.solve_rate_field] >= config.difficulty_filtering.min_solve_rate
            and x[config.difficulty_filtering.solve_rate_field] <= config.difficulty_filtering.max_solve_rate
        )

    # Setup TOPLOC
    hidden_size = llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size
    toploc_cache, _ = setup_toploc_cache(
        llm,
        pipeline_config=config.pp,
        disable=not config.toploc,
        max_seqs=batch_size,
        hidden_size=hidden_size,
    )

    ckpt_step = 0
    real_step = config.start_step or 0
    if config.ckpt_start_path is not None:
        logger.info(f"Resuming from checkpoint {config.ckpt_start_path}")
        path = Path(config.ckpt_start_path)
        path_file = path / "model.safetensors"
        if not path_file.exists():
            raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
        ckpt_step = int(path.name.split("_")[-1])
        logger.info(f"Resuming from step {ckpt_step} at {path_file}")
        llm = reload_model_weights(llm, path_file)
        real_step = ckpt_step

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_samples = 0
    total_tokens = 0

    # Compute the maximum number of problems and problems per batch
    problems_per_batch = batch_size // config.sampling.n
    logger.info(
        f"Problems per batch: {batch_size} // {config.sampling.n} = {problems_per_batch} (missing: {batch_size % config.sampling.n})"
    )

    dataset_offset = 0
    while True:
        if config.step_endpoint is not None:
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_real_step = requests.get(config.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_real_step != real_step:
                real_step = new_real_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        # Reload model weights if needed
        logger.info(f"Inference step {real_step} (Checkpoint step: {ckpt_step})")
        if config.rollout_path is not None and real_step - ckpt_step > config.async_level:
            logger.info(f"Required to reload model weights for step {ckpt_step} from {config.rollout_path}")
            ckpt_step = real_step - config.async_level
            attempt_count = 0
            while True:
                stable_file = Path(config.rollout_path) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights for step {ckpt_step} from {stable_file}")
                    llm = reload_model_weights(llm, Path(config.rollout_path) / f"step_{ckpt_step}/model.safetensors")
                    total_problems = 0
                    total_tokens = 0
                    logger.info(f"Reloaded model weights for step {ckpt_step} from {stable_file}")
                    break
                if attempt_count % 30 == 0:
                    logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint")
                time.sleep(1)
                attempt_count += 1

        # Get batch
        if node_address_int is not None:
            # TODO: What if we have multiple sample per real step?
            # Its impossible right now but we need to fix this if accept counter is used.

            # We reseed the generator here to make the sampling reproducible at each step.
            # This would work even if the node restarts and resumes from the current step.
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + real_step)
            indices = generator.integers(0, len(dataset), problems_per_batch)
        else:
            # Use modulo to cycle through the dataset instead of terminating
            indices = [(dataset_offset + j) % len(dataset) for j in range(problems_per_batch)]

        logger.debug(f"Sampling batch with indices [{' '.join(map(str, indices[:3]))}...{' '.join(map(str, indices[-3:]))}]")
        inputs = dataset.select(indices)

        # Prepare expanded batch for environment
        # Prime-RL uses vLLM's `n` parameter for multiple samples per prompt
        # But verifiers environments expect expanded batch format (prompt repeated for each generation)
        expanded_batch = {}
        for key in inputs.column_names:
            expanded_values = []
            for value in inputs[key]:
                for _ in range(config.sampling.n):
                    expanded_values.append(value)
            expanded_batch[key] = expanded_values

        start_time = time.time()

        # Use environment's generate method
        env_results = env.generate(
            inputs=expanded_batch,
            client=openai_client,
            model=config.model_name,
            score_rollouts=True,
            max_concurrent=batch_size,
            sampling_args={
                "max_tokens": config.sampling.max_tokens or 4096,
                "temperature": config.sampling.temperature,
            },
        )

        end_time = time.time()

        # Extract prompts and completions from env results
        prompts = env_results["prompt"]
        completions = env_results["completion"]
        rewards = env_results["reward"]
        states = env_results.get("state", [{}] * len(prompts))

        # Use process_env_results to handle tokenization
        processed_results = env.process_env_results(
            prompts=prompts,
            completions=completions,
            states=states,
            rewards=rewards,
            processing_class=tokenizer,
            max_completion_length=-1,
            mask_truncated_completions=False,
            mask_env_responses=False,
        )

        # Convert to vLLM-like format for compatibility with rest of pipeline
        # Create mock request_outputs structure
        from types import SimpleNamespace

        request_outputs = []
        req_id = 0
        for i in range(0, len(prompts), config.sampling.n):
            # Group outputs by original prompt
            req = SimpleNamespace()
            req.request_id = str(req_id)
            req.prompt = prompts[i]  # Take first prompt from group
            req.prompt_token_ids = processed_results["prompt_ids"][i]
            req.outputs = []

            for j in range(config.sampling.n):
                idx = i + j
                if idx < len(completions):
                    output = SimpleNamespace()
                    output.index = j
                    # Convert completion to string if it's a list of message dicts
                    if isinstance(completions[idx], list):
                        # Extract assistant content from message dicts
                        completion_text = ""
                        for msg in completions[idx]:
                            if msg.get("role") == "assistant":
                                completion_text += msg.get("content", "")
                        output.text = completion_text
                    else:
                        output.text = completions[idx]
                    output.token_ids = processed_results["completion_ids"][idx]
                    output.finish_reason = "stop"
                    req.outputs.append(output)

            request_outputs.append(req)
            req_id += 1

        # Dropping like this isn't ideal. But in practice, we shouldn't have any prompts that are too long.
        request_outputs = [req for req in request_outputs if len(req.outputs[0].token_ids) > 0]
        if len(request_outputs) != problems_per_batch:
            logger.warning(f"{problems_per_batch - len(request_outputs)} prompts were filtered out because they were too long")

        # This generates proofs for the remaining sequences that haven't reached max_len.
        # We call here to give time for the proofs to be generated non-blocking in the background.
        toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

        # Compute progress metrics
        batch_problems = len(request_outputs)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.info(f"Generated {batch_samples} samples for {batch_problems} problems for step {real_step} in {end_time - start_time:.2f}s")

        # Print example
        # first_prompt = tokenizer.decode(request_outputs[0].prompt_token_ids)
        # first_completion = tokenizer.decode(request_outputs[0].outputs[0].token_ids)
        logger.debug(f"Example prompt tokens: {len(request_outputs[0].prompt_token_ids)}")
        logger.debug(f"Example completion tokens: {len(request_outputs[0].outputs[0].token_ids)}")

        # Log progress metrics
        progress_metrics = {
            "progress/batch_problems": batch_problems,
            "progress/batch_samples": batch_samples,
            "progress/batch_tokens": batch_tokens,
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
        }
        monitor.log(perf_metrics)

        # Compute proofs
        # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
        # Generate always adds requests to the engine in the order of the prompts.
        # And returns them in the sequence they were added.
        toploc_cache.wait_for_proofs()

        # For environments, we don't have toploc proofs, so create empty proofs for each completion
        if config.toploc and toploc_cache.proofs:
            proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
        else:
            # Create empty proofs for each completion
            proofs = []
            for req in request_outputs:
                for _ in req.outputs:
                    proofs.append(b"")

        toploc_cache.reset_cache()

        # Compute rewards and advantages
        start = time.time()
        # Create mock reward structure compatible with existing pipeline
        request_rewards = []
        reward_idx = 0
        for req in request_outputs:
            req_reward = SimpleNamespace()
            req_reward.request_id = getattr(req, "request_id", str(reward_idx))
            req_reward.task_type = "default"  # Could extract from env results if available
            req_reward.rewards = []
            for i, output in enumerate(req.outputs):
                reward_obj = SimpleNamespace()
                reward_obj.completion_id = i
                reward_obj.reward = rewards[reward_idx] if reward_idx < len(rewards) else 0.0
                reward_obj.advantage = 0.0  # Will be computed later in pipeline
                reward_obj.task_reward = reward_obj.reward
                reward_obj.length_penalty = 0.0
                req_reward.rewards.append(reward_obj)
                reward_idx += 1
            request_rewards.append(req_reward)
        logger.info(f"Computed rewards and advantages in {time.time() - start:.2f}s")

        batch_rewards = sum(sum(r.reward for r in req.rewards) for req in request_rewards) / batch_samples

        monitor.log({"rewards/batch_rewards": batch_rewards})
        logger.info(f"Average reward of the batch: {batch_rewards}")

        # Get parquet table
        # Extract prompts list for parquet table
        prompts_for_table = []
        for req in request_outputs:
            for _ in req.outputs:
                # Convert message dicts to string if needed
                if isinstance(req.prompt, list):
                    # It's a list of message dicts, use apply_chat_template
                    prompt_str = tokenizer.apply_chat_template(req.prompt, tokenize=False, add_generation_prompt=True)
                    prompts_for_table.append(prompt_str)
                else:
                    prompts_for_table.append(req.prompt)

        table = get_parquet_table(request_outputs, request_rewards, prompts_for_table, proofs, ckpt_step)

        # Save outputs to parquet file
        step_path = Path(config.output_path) / f"step_{real_step}"
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"{uuid.uuid4()}.parquet"
        pq.write_table(table, save_path)
        logger.info(f"Saved batch outputs to {save_path}")

        # Log file metadata
        sha256 = sha256sum(save_path)
        flop_counts = [
            get_inference_input_output_flops(config.model_name, len(input_tokens), len(output_tokens))
            for input_tokens, output_tokens in zip(table.column("input_tokens").to_pylist(), table.column("output_tokens").to_pylist())
        ]

        monitor.log(
            {
                "output/save_path": save_path.as_posix(),
                "output/sha256": sha256,
                "output/output_flops": sum(output_flops for _, output_flops in flop_counts) // config.pp.world_size,
                "output/input_flops": sum(input_flops for input_flops, _ in flop_counts) // config.pp.world_size,
            }
        )

        real_step += 1

        if config.total_step is not None and real_step > config.total_step:
            logger.info(f"Reached total step {config.total_step}, stopping inference")
            break

        dataset_offset += batch_size

    logger.info(f"Inference finished! Generated {total_samples} samples for {total_problems} problems")

    # Shutdown MockOpenAI server
    oai_server.shutdown()

    # Manually destroy vLLM process group to avoid warnings
    dist.destroy_process_group()


def main(config: Config) -> list[mp.Process]:
    processes = []
    import zeroband.inference.envs as envs

    if config.dp > 1:
        if config.tp == "auto":
            assert torch.cuda.device_count() % config.dp == 0, "Number of GPUs must be divisible by DP"
            config.tp = torch.cuda.device_count() // config.dp
        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.tp] for i in range(0, len(gpu_ids), config.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            envs = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, envs), args=(config,))
            processes.append(process)
    else:
        if config.tp == "auto":
            config.tp = torch.cuda.device_count()
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
            config.rollout_path,
            config.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
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
