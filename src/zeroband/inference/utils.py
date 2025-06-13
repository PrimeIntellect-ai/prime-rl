from typing import Any

import torch
from datasets import Dataset
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from vllm import LLM
from vllm.model_executor.model_loader.loader import _process_weights_after_loading
from vllm.transformers_utils.tokenizer import AnyTokenizer

from zeroband.inference.config import LenRewardsConfig
from zeroband.inference.work_counting import get_inference_input_output_flops  # noqa: F401


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


def generate_target_lengths(len_reward_config: LenRewardsConfig | None, batch_size: int) -> list[int]:
    """
    Generate target lengths for all prompts in the batch.

    Args:
        len_reward_config: The length reward configuration.
        batch_size: The number of prompts to generate target lengths for.

    Returns:
        A list of target lengths.
    """
    if len_reward_config is None:
        target_lengths = [-1] * batch_size
    elif len_reward_config.target_length_sampling == "discrete":
        indices = torch.randint(low=0, high=len(len_reward_config.target_lengths), size=(batch_size,), device="cpu")
        target_lengths = [int(len_reward_config.target_lengths[i]) for i in indices]
    elif len_reward_config.target_length_sampling == "range":
        target_lengths = torch.randint(
            low=len_reward_config.min_length, high=len_reward_config.max_length + 1, size=(batch_size,), device="cpu"
        ).tolist()
    else:
        raise ValueError("'length_target_sampling' has to be 'discrete' or 'range'")

    return target_lengths


def format_prompts(
    prompts: list[str],
    target_lengths: list[int],
    len_rewards_config: LenRewardsConfig | None,
    tokenizer: AnyTokenizer,
    enable_thinking: bool = True,
    tokenize: bool = False,
) -> list[str] | BatchEncoding:
    """
    Formats a batch of raw prompts. Relies on the default chat template of the
    LLM's tokenizer to call `apply_chat_template`. We call with
    `add_generation_prompt=True` to add the generation prompt to the beginning
    of the prompt. We also call with `enable_thinking=True` to enable thinking
    for models that support it. For example, for `Qwen/QwQ-32B` this will add an
    unclosed `</think>` tag to the beginning of the system response.

    Args:
        prompts: A list of raw prompts.
        target_lengths: A list of target lengths (will be [-1, -1, ...] if no length rewards are configured).
        len_rewards_config: A configuration for length rewards. If `None`, no length rewards are configured.
        tokenizer: Any HF tokenizer instance
        enable_thinking: Whether to enable thinking for the model. Used by the `apply_chat_template` to prepend a thinking prompt (for some models)
        tokenize: Whether to tokenize the formatted prompts. If True, returns BatchEncoding; if False (default), returns list[str].

    Returns:
        A list of formatted prompts if tokenize=False, or a BatchEncoding if tokenize=True.
    """
    # Apply length prompt additions
    if len_rewards_config:
        max_word = "maximally" if len_rewards_config.reward_type == "clip" else ""
        if len_rewards_config.length_prompt_location == "system_prompt":  # Add length prompt to system prompt
            messages = [
                [
                    {"role": "system", "content": f"Think for {max_word}{target_length} tokens before giving a response."},
                    {"role": "user", "content": prompt},
                ]
                for prompt, target_length in zip(prompts, target_lengths)
            ]
        else:  # Add length prompt to user prompt
            messages = [
                [{"role": "user", "content": prompt + f" Think for {max_word}{target_length} tokens before giving a response."}]
                for prompt, target_length in zip(prompts, target_lengths)
            ]
    else:
        # No length prompt additions, just use the prompts as is
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    # Apply chat template
    formatted_prompts = tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=enable_thinking
    )

    if not tokenize:
        for i, _formatted_prompt in enumerate(formatted_prompts):
            if tokenizer.bos_token and _formatted_prompt.startswith(tokenizer.bos_token):
                formatted_prompts[i] = _formatted_prompt[len(tokenizer.bos_token) :]

    return formatted_prompts


def compute_max_batch_size(llm: LLM) -> int:
    """
    Automatically computes the maximum batch size (number of sequences decoded in
    parallel) without exceeding the GPU memory to prevent cache eviction. We use vLLM's
    cache config which gets populated when first initializing the LLM class. The two
    important keys are `num_gpu_blocks` and `block_size`. The `block_size` is the number
    of tokens that can be cached per GPU block. The `num_gpu_blocks` is the number of
    GPU blocks that vLLM allocates in total. Hence, we can compute the total number of
    tokens that can be cached by multiplying the  two. The maximum batch size is then
    computed by dividing the total number of tokens that can be cached by the maximum
    model length. This is a conservative estimate of the maximum batch size as it assumes
    that all cached tokens are distinct (no cache hit) and all sequences reach the
    maximum sequence length possible, here `max_model_len`.

    Args:
        llm (LLM): The vLLM LLM instance.

    Returns:
        int: The maximum batch size.
    """
    num_gpu_blocks = llm.llm_engine.model_executor.cache_config.num_gpu_blocks
    block_size = llm.llm_engine.model_executor.cache_config.block_size
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_cache_tokens = num_gpu_blocks * block_size
    max_batch_size = max_cache_tokens // max_model_len

    return max_batch_size


def rgetattr(obj: Any, attr_path: str) -> Any:
    """
    Tries to get a (nested) attribute from an object. For example:

    ```python
    class Foo:
        bar = "baz"

    class Bar:
        foo = Foo()

    foo = Foo()
    bar = Bar()
    ```

    Here, the following holds:
    - `getattr(foo, "bar")` will return `"baz"`.
    - `getattr(bar, "foo)` will return an object of type `Foo`.
    - `getattr(bar, "foo.bar")` will error

    This function solves this. `rgetattr(bar, "foo.bar")` will return `"baz"`.

    Args:
        obj: The object to get the attribute from.
        attr_path: The path to the attribute, nested using `.` as separator.

    Returns:
        The attribute
    """
    attrs = attr_path.split(".")
    current = obj

    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)

    return current
