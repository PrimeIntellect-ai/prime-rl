from functools import lru_cache
import os
import asyncio
import json
from pathlib import Path
from typing import Literal
import uuid
import numpy as np
import torch
from vllm import LLM, SamplingParams
from pydantic_config import BaseConfig, parse_argv
import vllm
import concurrent.futures
import time
from toploc.utils import sha256sum
from safetensors import safe_open

# from vllm.model_executor.model_loader
from vllm.model_executor.model_loader.loader import _process_weights_after_loading
from vllm.sequence import SampleLogprobs
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import _prune_hidden_states

from zeroband.logger import get_logger
from zeroband.models import ModelName
from zeroband.rewards.math import compute_math_reward

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp

from zeroband.inferencing.toploc import TopLocCache
from zeroband.training.mp import EnvWrapper, cuda_available_devices
from zeroband.prime_metrics import PrimeMetric

from pydantic import model_validator


class SamplingParamConfig(BaseConfig):
    temperature: float = 0.6
    max_tokens: int | None = None
    ignore_eos: bool = False
    top_p: float = 1
    n: int = 8
    logprobs: int = 0  # 0 mean 1 logprob here
    top_k: int = -1


class LenRewardConfig(BaseConfig):
    min_len: int = 1000
    max_len: int = 24000
    reward_coef: int = 0.0003


class Config(BaseConfig):
    name_model: ModelName = "150M"
    dataset: str = "justus27/deepscaler-math-genesys-format"
    batch_size: int = 32
    max_samples: int | None = None
    output_path: str = "outputs"
    total_step: int | None = None
    rollout_path: str | None = None

    quant: Literal["fp8"] | None = None

    sampling: SamplingParamConfig = SamplingParamConfig()
    enforce_eager: bool = False
    max_model_len: int | None = None

    async_level: int = 2  # the amount of step for which we can be in advance

    # mutli gpu
    tp: int = 1
    dp: int = 1
    gpus_ids: list[int] | None = None
    prime_log_freq: int | None = None

    seed: int | None = None  # THIS ARG FOR TESTING PURPOSES ONLY

    dtype: Literal["fp32", "bf16"] = "bf16"

    ckpt_start_path: str | None = None

    len_reward: LenRewardConfig | None = None
    toploc: bool = True

    @model_validator(mode="after")
    def validate_step_batch_size(self):
        assert (self.len_reward.min_len is None) == (self.len_reward.max_len is None), (
            "len_reward.min_length and len_reward.max_length must be either both defined or both None"
        )
        return self


pa_schema = pa.schema(
    [
        ("input_tokens", pa.list_(pa.int32())),
        ("output_tokens", pa.list_(pa.int32())),
        ("input_logprobs", pa.list_(pa.float32())),
        ("output_logprobs", pa.list_(pa.float32())),
        ("advantages", pa.float32()),
        ("rewards", pa.float32()),
        ("task_rewards", pa.float32()),
        ("length_penalties", pa.float32()),
        ("proofs", pa.binary()),
        ("step", pa.int32()),
        ("target_lengths", pa.int32()),
    ]
)


@lru_cache(maxsize=1)
def get_process_executor():
    return concurrent.futures.ProcessPoolExecutor(max_workers=8)


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
    if config.len_reward.min_len and config.len_reward.max_len:
        target_lengths = torch.randint(
            low=config.len_reward.min_len, high=config.len_reward.max_len + 1, size=(batch_size,), device="cpu"
        ).tolist()

        return [f"\n\nThink for {target} tokens." for target in target_lengths], target_lengths

    return [""] * batch_size, [-1] * batch_size


async def compute_reward_for_output(output, verification_info, len_reward_coeff):
    loop = asyncio.get_running_loop()
    # Run compute_math_reward in a separate process via our ProcessPoolExecutor.
    math_reward = await loop.run_in_executor(get_process_executor(), compute_math_reward, output.text, verification_info)

    total_reward = math_reward
    length_penalty = 0
    if verification_info["target_length"] > 0 and math_reward == 1:
        # Calculate length reward - this could be a separate function
        output_length = len(output.token_ids)
        target_length = verification_info["target_length"]

        length_penalty = abs(output_length - target_length)
        length_penalty = length_penalty * len_reward_coeff  # Scale factor to balance with math reward
        length_penalty = min(1, length_penalty)

        total_reward -= length_penalty

    return dict(total_reward=total_reward, task_reward=math_reward, length_penalty=length_penalty)


async def compute_rewards_async(
    generated_tokens: list[vllm.RequestOutput], verification_infos: list[str], target_lengths: list[int], config: Config
) -> tuple[dict[int, torch.FloatTensor], dict[int, torch.FloatTensor], dict[int, torch.FloatTensor]]:
    parsed_infos = [json.loads(ver) for ver in verification_infos]

    for info, target_len in zip(parsed_infos, target_lengths):
        info["target_length"] = target_len

    tasks = []
    mapping = []

    for req_idx, (request, verification_info) in enumerate(zip(generated_tokens, parsed_infos)):
        for output in request.outputs:
            tasks.append(asyncio.create_task(compute_reward_for_output(output, verification_info, config.len_reward.reward_coef)))
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
    prime_metric = PrimeMetric(disable=config.prime_log_freq is None, period=config.prime_log_freq)
    llm = LLM(
        model=config.name_model,
        tensor_parallel_size=config.tp,
        max_seq_len_to_capture=config.max_model_len,
        max_model_len=config.max_model_len,
        quantization=config.quant,
        enforce_eager=config.enforce_eager,
        dtype="bfloat16" if config.dtype == "bf16" else torch.float32,
    )
    tokenizer = llm.get_tokenizer()
    rank = int(os.environ.get("RANK", "0"))
    logger = get_logger(f"INFERENCE {rank}")
    sampling_params = SamplingParams(**config.sampling.model_dump())

    generator = np.random.default_rng(config.seed + rank) if config.seed is not None else np.random.default_rng()
    # not sure what is the default seed for np.random.default_rng so doing this to make sure we use the default value

    dataset = load_dataset(config.dataset, split="train").shuffle(generator=generator)
    max_samples = config.max_samples or len(dataset)

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    if config.dtype == "fp32":
        config.toploc = False

    toploc_cache = TopLocCache(
        max_seqs=config.batch_size * config.sampling.n, max_len=32, hidden_size=model.config.hidden_size, disable=not config.toploc
    )

    def logits_processor_hook(module, input):
        hidden_states, sampling_metadata = input[1], input[2]
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(sampling_metadata, SamplingMetadata)
        # This check is true only for prefills
        if max(sampling_metadata.selected_token_indices) > len(sampling_metadata.seq_groups):
            return
        # This pruning is required when cuda graph padding is enabled.
        hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)
        if len(sampling_metadata.seq_groups) != hidden_states.shape[0]:
            raise ValueError(f"Lengths dont match: {len(sampling_metadata.seq_groups)} {hidden_states.shape}")

        # TODO: Maybe just use sampling_metadata.selected_token_indices?
        index = [i.seq_ids[0] for i in sampling_metadata.seq_groups]
        toploc_cache.add(index, hidden_states)

    if not toploc_cache.disable:
        model.logits_processor.register_forward_pre_hook(logits_processor_hook)

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

    total_problems = 0
    total_tokens = 0

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        logger.info(
            f"real_step: {real_step}, ckpt_step: {ckpt_step}, real_step - ckpt_step: {real_step - ckpt_step}, config.async_level: {config.async_level}"
        )
        if config.rollout_path is not None and real_step - ckpt_step > config.async_level:
            ckpt_step += 1
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
        batch = dataset.select(range(i, min(i + config.batch_size, len(dataset))))

        length_prompt_additions, target_lengths = generate_target_length_prompts(config, len(batch))

        messages = [
            [{"role": "user", "content": item["prompt"] + length_prompt}, {"role": "assistant", "content": "<think>\n"}]
            for item, length_prompt in zip(batch, length_prompt_additions)
        ]
        # Assume verification_info is stored as a JSON string in the dataset.
        verification_infos = [item["verification_info"] for item in batch]

        assert tokenizer.chat_template is not None, "Selected Tokenizer does not have a chat template"
        prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        for i, p in enumerate(prompts):
            prompts[i] = p.replace("<｜begin▁of▁sentence｜>", "")

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

        # Compute rewards asynchronously, grouped as a dictionary.
        grouped_rewards, grouped_task_rewards, grouped_length_penalties = asyncio.run(
            compute_rewards_async(generated_tokens, verification_infos, target_lengths, config)
        )
        # Compute normalized advantages per prompt.
        grouped_advantages = compute_advantages_grpo(grouped_rewards)

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

    get_process_executor().shutdown(wait=True)


def inference_sub_process(config: Config, gpus_ids: list[int], rank: int) -> list[mp.Process]:
    """
    This function is used to run inference by creating a sub process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_available_devices(gpus_ids)
    # this is a hack that work with spawn, basically allow the env var to be overridden when spawn read the env var a second time

    envs = {"CUDA_VISIBLE_DEVICES": cuda_available_devices(gpus_ids), "RANK": str(rank)}
    print(f"start inference on {gpus_ids} with rank {rank}")
    fn_env = EnvWrapper(inference, envs)
    process = mp.Process(target=fn_env, args=(config,))
    process.start()

    return process


def inference_run(config: Config) -> list[mp.Process]:
    if config.dp > 1:
        processes = []

        gpus_ids = config.gpus_ids if config.gpus_ids is not None else list(range(torch.cuda.device_count()))

        assert len(gpus_ids) % (config.dp * config.tp) == 0, "Number of GPUs must be divisible by dp * tp"

        num_process = len(gpus_ids) // config.tp
        sub_process_ids = [gpus_ids[i * config.tp : (i + 1) * config.tp] for i in range(num_process)]

        for rank, sub_process_id in enumerate(sub_process_ids):
            processes.append(inference_sub_process(config, sub_process_id, rank))

        return processes

    else:
        inference(config)
        return []


def main(config: Config) -> list[mp.Process]:
    processes = inference_run(config)
    for process in processes:
        process.join()


if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")
    config = Config(**parse_argv())  # type: ignore

    # Maybe start shardcast downloader
    from zeroband.inferencing import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        from zeroband.inferencing.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rollout_path,
            config.async_level + 1,
            inference_envs.SHARDCAST_BACKLOG_VERSION,
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
