from functools import partial

import torch.nn as nn
from vllm import LLM
from vllm.model_executor.layers.sampler import SamplerOutput, SamplingMetadata

from zeroband.inference.config import SamplingConfig
from zeroband.utils.logger import get_logger


def swap_think_token(_, __, kwargs, outputs, config: SamplingConfig, stop_think_token_id: int) -> SamplerOutput | None:
    """
    A post-hook that swaps the last sampled token with the stop_think_token_id
    if the thinking budget is exceeded.

    Args:
        _: The module that is being hooked
        __: The arguments to the module
        kwargs: The named arguments to the module
        outputs: The outputs of the module
        config: The sampling configuration.
        stop_think_token_id: The token ID of the `</think>` token.

    Returns:
        None
    """
    sampling_metadata: SamplingMetadata = kwargs.get("sampling_metadata", None)
    assert sampling_metadata is not None, "Sampling metadata is required for the `swap_think_token` post-hook"
    sampling_output: SamplerOutput | None = outputs

    if sampling_metadata:
        for seq_group, seq_outputs in zip(sampling_metadata.seq_groups, sampling_output.outputs):
            for seq_id, seq_data, seq_output in zip(seq_group.seq_data.keys(), seq_group.seq_data.values(), seq_outputs.samples):
                num_output_tokens = len(seq_data.output_token_ids) + 1  # Increment by 1 because the current token is not yet in the output
                if num_output_tokens == config.max_think_tokens:
                    get_logger("INFER").debug(
                        f"Thinking budget reached for sequence {seq_id} after {num_output_tokens} tokens. Replacing sampled token {seq_output.output_token} with {stop_think_token_id}"
                    )
                    seq_output.logprobs = {stop_think_token_id: seq_output.logprobs[seq_output.output_token]}
                    seq_output.output_token = stop_think_token_id

    return sampling_output


def setup_elastic_reasoning(config: SamplingConfig, llm: LLM) -> None:
    """
    Sets up elastic reasoning by registering a post-hook on the sampler to swap
    the last sampled token with the stop_think_token_id if the thinking budget
    is exceeded.

    Args:
        config: The sampling configuration.
        llm: The LLM model.
    """
    assert config.max_think_tokens is not None, "`max_think_tokens` must be set for elastic reasoning"
    get_logger("INFER").info(f"Enabling elastic reasoning with max_think_tokens={config.max_think_tokens} (max_tokens={config.max_tokens})")

    # Dynamically get the token ID of the `</think>` token from model's tokenizer
    tokenizer = llm.get_tokenizer()
    stop_think_token_id = tokenizer.convert_tokens_to_ids("</think>")
    assert type(stop_think_token_id) == int, "`</think>` token must be a single token"

    # Register the post-hook on the sampler
    sampler: nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.sampler
    sampler.register_forward_hook(partial(swap_think_token, config=config, stop_think_token_id=stop_think_token_id), with_kwargs=True)

    get_logger("INFER").debug("Set up post-hook swap_think_token on sampler")
