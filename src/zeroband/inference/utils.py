import torch
from datasets import Dataset
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm import LLM
from vllm.model_executor.model_loader.loader import _process_weights_after_loading

from zeroband.inference.rewards import LenRewardsConfig


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


def generate_target_length_prompts(config: LenRewardsConfig, batch_size: int):
    if config is None:
        return [""] * batch_size, [-1] * batch_size

    if config.target_length_sampling == "discrete":
        indices = torch.randint(low=0, high=len(config.len_reward.target_lengths), size=(batch_size,), device="cpu")
        target_lengths = [int(config.len_reward.target_lengths[i]) for i in indices]

    elif config.target_length_sampling == "range":
        target_lengths = torch.randint(
            low=config.len_reward.min_length, high=config.len_reward.max_length + 1, size=(batch_size,), device="cpu"
        ).tolist()

    else:
        raise ValueError("'length_target_sampling' has to be 'discrete' or 'range'")

    prompt_prefix = " " if config.len_reward.length_prompt_location == "instruction" else " "
    max_word = " maximally " if config.len_reward.reward_type == "clip" else ""

    return [f"{prompt_prefix}Think for{max_word}{target} tokens before giving a response." for target in target_lengths], target_lengths


def filter_data_by_prompt_length(data: Dataset, max_length: int, tokenizer: AutoTokenizer, tokenize_batch_size: int = 10000):
    def _add_token_lengths_batched(examples):
        prompts = examples["prompt"]
        tokenized = tokenizer(prompts, padding=False, truncation=False)
        token_lengths = [len(ids) for ids in tokenized.input_ids]
        return {"token_length": token_lengths}

    data = data.map(
        _add_token_lengths_batched,
        batched=True,
        batch_size=tokenize_batch_size,
        desc=f"Calculating prompt lengths to filter out lengths > {max_length}",
    )

    data = data.filter(lambda x: x["token_length"] <= max_length)

    return data
