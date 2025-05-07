from functools import lru_cache
import requests
import os
import asyncio
import json
from pathlib import Path
from typing import Literal
import uuid
import numpy as np
import torch
from vllm import LLM, SamplingParams
from pydantic import model_validator
from pydantic_config import BaseConfig, parse_argv
import vllm
import concurrent.futures
import time
from toploc.utils import sha256sum
from safetensors import safe_open
import torch.distributed as dist

# from vllm.model_executor.model_loader
from vllm.model_executor.model_loader.loader import _process_weights_after_loading
from vllm.sequence import SampleLogprobs

from zeroband.utils.logger import get_logger

from zeroband.utils.models import ModelName
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.pipeline import PipelineConfig, setup_pipeline
from zeroband.rewards.registry import REWARD_FUNCTIONS


from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp

from zeroband.training.mp import EnvWrapper
from zeroband.utils.metrics import PrimeMetric
from zeroband.inference.schema import pa_schema

from zeroband.inference import envs

# Global logger
logger = get_logger("INFER")


class SamplingParamConfig(BaseConfig):
    temperature: float = 0.6
    max_tokens: int | None = None
    ignore_eos: bool = False
    top_p: float = 1
    n: int = 8
    logprobs: int = 0  # 0 mean 1 logprob here
    top_k: int = -1


class LenRewardConfig(BaseConfig):
    reward_type: Literal["exact", "max", "clip"] = "max"
    target_length_sampling: Literal["discrete", "range"] = "discrete"
    length_prompt_location: Literal["system_prompt", "instruction"] = "system_prompt"

    # applicable if target_length_sampling == "range"
    min_length: int = 1000
    max_length: int = 24000

    # applicable if target_length_sampling == "discrete"
    target_lengths: list[float] = [500, 1000, 2000, 3000]

    # applicable for reward_type max and exact
    reward_coef: float = 0.0003

    # only applicable for reward_type == "max"
    max_reward_delta: float = 0.5


class DifficultyFilteringConfig(BaseConfig):
    solve_rate_field: str = "solve_rate_qwen_r1_distill_7b"
    min_solve_rate: float = 0.0
    max_solve_rate: float = 0.5


class Config(BaseConfig):
    model_name: ModelName = "150M"
    dataset: str = "justus27/rl-code-and-math"
    batch_size: int = 32
    max_samples: int | None = None
    output_path: str = "outputs"
    total_step: int | None = None
    rollout_path: str | None = None
    step_endpoint: str | None = None
    download_dir: str | None = None

    quant: Literal["fp8"] | None = None

    sampling: SamplingParamConfig = SamplingParamConfig()
    enforce_eager: bool = False
    max_model_len: int | None = None

    async_level: int = 2  # the amount of step for which we can be in advance

    # Parallelism
    tp: int | Literal["auto"] = 1
    dp: int = 1
    pp: PipelineConfig = PipelineConfig()

    gpus_ids: list[int] | None = None
    prime_log_freq: int | None = None

    seed: int | None = None  # THIS ARG FOR TESTING PURPOSES ONLY

    dtype: Literal["fp32", "bf16"] = "bf16"

    ckpt_start_path: str | None = None

    toploc: bool = False

    len_reward: LenRewardConfig | None = None
    difficulty_filtering: DifficultyFilteringConfig | None = None

    @model_validator(mode="after")
    def disable_toploc_for_fp32(self):
        if self.dtype == "fp32":
            self.toploc = False
        return self

    @model_validator(mode="after")
    def enforce_eager_for_tp(self):
        if self.pp.world_size > 1:
            self.enforce_eager = True
        return self

    @model_validator(mode="after")
    def assert_valid_parallelism(self):
        assert not (self.dp > 1 and self.pp.world_size > 1), "Cannot use PP and DP together"
        return self


def fake_chat_template(messages):
    formatted_prompts = []

    for conversation in messages:
        prompt = ""
        for message in conversation:
            if message["role"] == "user":
                prompt += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        formatted_prompts.append(prompt.strip())

    return formatted_prompts


@lru_cache(maxsize=1)
def get_process_executor():
    return concurrent.futures.ProcessPoolExecutor(max_workers=32)


def get_own_logprobs(sample_logprobs: SampleLogprobs) -> float:
    logprobs = []

    for logprob in sample_logprobs:
        assert isinstance(logprob, dict), "Logprobs should be a dict"
        assert len(logprob) == 1, "Logprobs should be a dict with 1 key"

        _token_id, logprob_p = list(logprob.items())[0]
        logprobs.append(logprob_p.logprob)

    return logprobs


def get_parquet_table(
    generated_tokens: list[vllm.RequestOutput],
    grouped_advantages: dict[int, list[float]],
    grouped_rewards: dict[int, torch.FloatTensor],
    grouped_task_rewards: dict[int, torch.FloatTensor],
    grouped_length_penalties: dict[int, torch.FloatTensor],
    proofs: list[bytes],
    step: int,
    target_lengths: list[int],
) -> pa.Table:
    input_tokens_list = []
    output_tokens_list = []
    input_logprobs_list = []
    output_logprobs_list = []
    advantages_list = []
    rewards_list = []
    task_rewards_list = []
    length_penalty_list = []
    proofs_list = []
    steps_list = []
    target_lengths_list = []

    proof_iter = iter(proofs)

    for i, (request, target_len) in enumerate(zip(generated_tokens, target_lengths)):
        advantages = grouped_advantages[i]
        rewards = grouped_rewards[i].tolist()
        task_rewards = grouped_task_rewards[i].tolist()
        length_penalties = grouped_length_penalties[i].tolist()
        for adv, reward, task_reward, length_penalty, output in zip(advantages, rewards, task_rewards, length_penalties, request.outputs):
            input_tokens_list.append(request.prompt_token_ids)
            output_tokens_list.append(output.token_ids)
            input_logprobs_list.append([0] * len(request.prompt_token_ids))  # putting 0 for now as not needed in the grpo loss
            output_logprobs_list.append(get_own_logprobs(output.logprobs))
            advantages_list.append(adv)
            rewards_list.append(reward)
            task_rewards_list.append(task_reward)
            length_penalty_list.append(length_penalty)
            proofs_list.append(next(proof_iter) if len(output.token_ids) > 1 else b"")
            steps_list.append(step)
            target_lengths_list.append(target_len)

    arrays = [
        pa.array(input_tokens_list, type=pa.list_(pa.int32())),
        pa.array(output_tokens_list, type=pa.list_(pa.int32())),
        pa.array(input_logprobs_list, type=pa.list_(pa.float32())),
        pa.array(output_logprobs_list, type=pa.list_(pa.float32())),
        pa.array(advantages_list, type=pa.float32()),
        pa.array(rewards_list, type=pa.float32()),
        pa.array(task_rewards_list, type=pa.float32()),
        pa.array(length_penalty_list, type=pa.float32()),
        pa.array(proofs_list, type=pa.binary()),
        pa.array(steps_list, type=pa.int32()),
        pa.array(target_lengths_list, type=pa.int32()),
    ]
    return pa.Table.from_arrays(arrays, schema=pa_schema)


def reload_model_weights(llm: LLM, ckpt_path: str):
    # Access the internal model from vLLM
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # Load state dict
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        # Create a better weight iterator that filters out empty keys and handles prefixes
        def weights_iterator():
            for key in f.keys():
                # Skip empty keys
                if not key:
                    continue
                yield key, f.get_tensor(key)

        # Load weights
        model.load_weights(weights_iterator())

    # Process weights after loading (important for some models)
    model_config = llm.llm_engine.model_config
    device = next(model.parameters()).device
    _process_weights_after_loading(model, model_config, device)

    return llm


def generate_target_length_prompts(config: Config, batch_size: int):
    if config.len_reward is None:
        return [""] * batch_size, [-1] * batch_size

    if config.len_reward.target_length_sampling == "discrete":
        indices = torch.randint(low=0, high=len(config.len_reward.target_lengths), size=(batch_size,), device="cpu")
        target_lengths = [int(config.len_reward.target_lengths[i]) for i in indices]

    elif config.len_reward.target_length_sampling == "range":
        target_lengths = torch.randint(
            low=config.len_reward.min_length, high=config.len_reward.max_length + 1, size=(batch_size,), device="cpu"
        ).tolist()

    else:
        raise ValueError("'length_target_sampling' has to be 'discrete' or 'range'")

    prompt_prefix = " " if config.len_reward.length_prompt_location == "instruction" else " "
    max_word = " maximally " if config.len_reward.reward_type == "clip" else ""

    return [f"{prompt_prefix}Think for{max_word}{target} tokens before giving a response." for target in target_lengths], target_lengths


async def compute_reward_for_output(output, verification_info, len_reward_config, task_type):
    loop = asyncio.get_running_loop()
    reward_fn = REWARD_FUNCTIONS[task_type]
    task_reward = await loop.run_in_executor(get_process_executor(), reward_fn, output.text, verification_info)

    total_reward = task_reward
    length_penalty = 0
    if verification_info["target_length"] > 0:
        output_length = len(output.token_ids)
        target_length = verification_info["target_length"]

        if len_reward_config.reward_type == "exact":
            length_penalty = abs(output_length - target_length)
            length_penalty = length_penalty * len_reward_config.reward_coef  # Scale factor to balance with math reward
            total_reward -= length_penalty

        elif len_reward_config.reward_type == "max":
            diff = target_length - output_length
            length_penalty = torch.clip(
                torch.tensor(len_reward_config.reward_coef * diff + len_reward_config.max_reward_delta), 0, 1
            ).item()
            total_reward *= length_penalty

        elif len_reward_config.reward_type == "clip":
            length_penalty = int(output_length > target_length)

            if length_penalty == 1:
                total_reward = 0

    return dict(total_reward=total_reward, task_reward=task_reward, length_penalty=length_penalty)


async def compute_rewards_async(
    generated_tokens: list[vllm.RequestOutput],
    verification_infos: list[str],
    target_lengths: list[int],
    task_types: list[str],
    config: Config,
) -> tuple[dict[int, torch.FloatTensor], dict[int, torch.FloatTensor], dict[int, torch.FloatTensor]]:
    parsed_infos = [json.loads(ver) for ver in verification_infos]

    for info, target_len in zip(parsed_infos, target_lengths):
        info["target_length"] = target_len

    tasks = []
    mapping = []

    for req_idx, (request, verification_info, task_type) in enumerate(zip(generated_tokens, parsed_infos, task_types)):
        for output in request.outputs:
            tasks.append(asyncio.create_task(compute_reward_for_output(output, verification_info, config.len_reward, task_type)))
            mapping.append(req_idx)

    all_results = await asyncio.gather(*tasks)

    grouped_total_rewards = {}
    grouped_task_rewards = {}
    grouped_length_penalties = {}

    for req_idx in set(mapping):
        grouped_total_rewards[req_idx] = []
        grouped_task_rewards[req_idx] = []
        grouped_length_penalties[req_idx] = []

    for req_idx, result in zip(mapping, all_results):
        grouped_total_rewards[req_idx].append(result["total_reward"])
        grouped_task_rewards[req_idx].append(result["task_reward"])
        grouped_length_penalties[req_idx].append(result["length_penalty"])

    for req_idx in grouped_total_rewards:
        grouped_total_rewards[req_idx] = torch.FloatTensor(grouped_total_rewards[req_idx])
        grouped_task_rewards[req_idx] = torch.FloatTensor(grouped_task_rewards[req_idx])
        grouped_length_penalties[req_idx] = torch.FloatTensor(grouped_length_penalties[req_idx])

    return grouped_total_rewards, grouped_task_rewards, grouped_length_penalties


def compute_advantages_grpo(grouped_rewards: dict[int, dict[str, torch.FloatTensor]], epsilon: float = 1e-6) -> dict[int, list[float]]:
    advantages = {}
    for req_idx, rewards_tensor in grouped_rewards.items():
        mean = torch.mean(rewards_tensor).item()
        std_dev = torch.std(rewards_tensor).item()
        normalized = ((rewards_tensor - mean) / (std_dev + epsilon)).tolist()
        advantages[req_idx] = normalized
    return advantages


def inference(config: Config):
    # Initialize the logger
    logger.info("Starting inference")
    logger.info(f"TP={config.tp}, DP={config.dp}, PP={config.pp.world_size}")

    # Initialize prime metrics
    prime_metric = PrimeMetric(disable=config.prime_log_freq is None, period=config.prime_log_freq)

    # Initialize vLLM and get tokenizer
    logger.info("Initializing vLLM")
    llm = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tp,
        max_seq_len_to_capture=config.max_model_len,
        max_model_len=config.max_model_len,
        quantization=config.quant,
        enforce_eager=config.enforce_eager,
        disable_async_output_proc=True,  # We have an off by 1 error in toploc without this flag when cuda graph padding is enabled.
        download_dir=config.download_dir,
        dtype="bfloat16" if config.dtype == "bf16" else torch.float32,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(**config.sampling.model_dump())

    # Load  dataset
    logger.info(f"Loading dataset {config.dataset}")
    dataset = load_dataset(config.dataset, split="train")

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
        generator = np.random.default_rng(seed)
        dataset = load_dataset(config.dataset, split="train").shuffle(generator=generator)
        node_address_int = None

    # Optionally filter dataset
    if config.difficulty_filtering:
        dataset = dataset.filter(
            lambda x: x[config.difficulty_filtering.solve_rate_field] >= config.difficulty_filtering.min_solve_rate
            and x[config.difficulty_filtering.solve_rate_field] <= config.difficulty_filtering.max_solve_rate
        )

    # Setup TOPLOC
    toploc_cache, _ = setup_toploc_cache(
        llm,
        disable=not config.toploc,
        max_seqs=config.batch_size * config.sampling.n,
        hidden_size=llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size,
    )

    if config.ckpt_start_path is not None:
        path = Path(config.ckpt_start_path)
        path_file = path / "model.safetensors"
        if not path_file.exists():
            raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
        ckpt_step = int(path.name.split("_")[-1])
        logger.info(f"Resuming from step {ckpt_step} at {path_file}")
        llm = reload_model_weights(llm, path_file)
        real_step = ckpt_step
    else:
        ckpt_step = 0
        real_step = 0

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_tokens = 0
    max_samples = config.max_samples or len(dataset)

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
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

        logger.info(
            f"real_step: {real_step}, ckpt_step: {ckpt_step}, real_step - ckpt_step: {real_step - ckpt_step}, config.async_level: {config.async_level}"
        )
        if config.rollout_path is not None and real_step - ckpt_step > config.async_level:
            ckpt_step = real_step - config.async_level
            attempt_count = 0
            while True:
                stable_file = Path(config.rollout_path) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights from {config.rollout_path} ckpt {ckpt_step}")
                    llm = reload_model_weights(llm, Path(config.rollout_path) / f"step_{ckpt_step}/model.safetensors")
                    total_problems = 0
                    total_tokens = 0
                    logger.info(f"Reloaded model weights from {config.rollout_path} ckpt {ckpt_step}")
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
            indexes = generator.integers(0, len(dataset), config.batch_size)
            batch = dataset.select(indexes)
        else:
            batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))
        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]

        length_prompt_additions, target_lengths = generate_target_length_prompts(config, len(batch))

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

        # Assume verification_info is stored as a JSON string in the dataset.
        verification_infos = [item["verification_info"] for item in batch]
        task_types = [item["task_type"] for item in batch]

        if tokenizer.chat_template:
            prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
            if config.model_name != "Qwen/QwQ-32B":
                for i, p in enumerate(prompts):
                    prompts[i] = p.replace("<｜begin▁of▁sentence｜>", "")
        else:
            prompts = fake_chat_template(messages)

        # Create communication for pipeline
        if config.pp.world_size > 1:
            setup_pipeline(
                llm=llm,
                rank=config.pp.rank,
                world_size=config.pp.world_size,
                iroh_seed=config.pp.iroh_seed,
                iroh_peer_id=config.pp.iroh_peer_id,
            )

        start_time = time.time()
        generated_tokens = llm.generate(prompts, sampling_params, use_tqdm=False)
        end_time = time.time()

        # Dropping like this isnt ideal. But in practice, we shouldnt have any prompts that are too long.
        generated_tokens = [req for req in generated_tokens if len(req.outputs[0].token_ids) > 0]
        if len(generated_tokens) != len(prompts):
            logger.warning(f"{len(prompts) - len(generated_tokens)} prompts were filtered out because they were too long")

        # This generates proofs for the remaining sequences that haven't reached max_len.
        # We call here to give time for the proofs to be generated non-blocking in the background.
        toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

        # Calculate tokens and throughput
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in generated_tokens)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in generated_tokens)
        batch_total_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_total_tokens

        avg_seq_length = batch_total_tokens / (len(generated_tokens) * config.sampling.n) if generated_tokens else 0

        elapsed_time = end_time - start_time
        tokens_per_second = batch_total_tokens / elapsed_time if elapsed_time > 0 else 0

        logger.info(
            f"Batch throughput: {tokens_per_second:.2f} tok/sec ({batch_total_tokens} tokens in {elapsed_time:.2f}s, avg seq len: {avg_seq_length:.1f})"
        )

        # Compute proofs
        # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
        # Generate always adds requests to the engine in the order of the prompts.
        # And returns them in the sequence they were added.
        toploc_cache.wait_for_proofs()
        proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
        toploc_cache.reset_cache()

        start_reward_advantages = time.time()
        # Compute rewards asynchronously, grouped as a dictionary.
        grouped_rewards, grouped_task_rewards, grouped_length_penalties = asyncio.run(
            compute_rewards_async(generated_tokens, verification_infos, target_lengths, task_types, config)
        )
        # Compute normalized advantages per prompt.
        grouped_advantages = compute_advantages_grpo(grouped_rewards)
        end_reward_advantages = time.time()

        elapsed_time = end_reward_advantages - start_reward_advantages
        logger.info(f"Computed rewards and advantages in in {elapsed_time:.2f}s")

        table = get_parquet_table(
            generated_tokens,
            grouped_advantages,
            grouped_rewards,
            grouped_task_rewards,
            grouped_length_penalties,
            proofs,
            ckpt_step,
            target_lengths,
        )

        step_path = Path(config.output_path) / f"step_{real_step}"
        os.makedirs(step_path, exist_ok=True)
        pq_save_path = f"{step_path}/{uuid.uuid4()}.parquet"
        pq.write_table(table, pq_save_path)
        file_sha = sha256sum(pq_save_path)
        prime_metric.log_prime({"file_sha": file_sha, "file_name": pq_save_path})
        logger.info(f"✨ Saved {len(proofs)} samples to {pq_save_path} with sha {file_sha or 'NA'}")

        total_problems += len(prompts)
        metric = {"dashbord-progress/total": total_problems, f"dashbord-progress/{config.dataset}": total_tokens}
        prime_metric.log_prime(metric)

        logger.info(f"Generated {total_problems} problems for step {real_step}")
        real_step += 1

        if config.total_step is not None and real_step > config.total_step:
            logger.info(f"Reached total step {config.total_step}, stopping inference")
            break

    # Manually destroy vLLM process group to avoid warnings
    dist.destroy_process_group()

    get_process_executor().shutdown(wait=True)


def main(config: Config) -> list[mp.Process]:
    processes = []
    from zeroband.inference import envs as inference_envs

    if config.dp > 1:
        if config.tp == "auto":
            assert torch.cuda.device_count() % config.dp == 0, "Number of GPUs must be divisible by DP"
            config.tp = torch.cuda.device_count() // config.dp
        gpu_ids = inference_envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.tp] for i in range(0, len(gpu_ids), config.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            envs = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "RANK": str(rank), "LOCAL_RANK": str(rank)}
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
