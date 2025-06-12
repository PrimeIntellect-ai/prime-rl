from functools import partial
from typing import Annotated

import torch.nn as nn
from pydantic import Field
from vllm import LLM
from vllm.model_executor.layers.sampler import SamplerOutput, SamplingMetadata

from zeroband.utils.config import BaseConfig
from zeroband.utils.logger import get_logger


class ElasticReasoningConfig(BaseConfig):
    """Configures elastic reasoning from `Scalable Chain of Thoughts via Elastic Reasoning` (https://arxiv.org/abs/2505.05315)."""

    enable: Annotated[bool, Field(default=False, description="Whether to enable elastic reasoning.")]
    think_budget: Annotated[
        int,
        Field(
            default=1024,
            description="Number of tokens to allow for thinking. If thinking is not done after this amount of tokens, overwrites the last sampled token with the stop_think_token_id to force the model to produce a solution.",
        ),
    ]
    solution_budget: Annotated[int, Field(default=1024, description="Number of tokens to allow for solution.")]
    stop_think_token_id: Annotated[
        int,
        Field(
            default=151668, description="The token ID of the `</think>` token which is swapped in to force the model to produce a solution."
        ),
    ]  # </think


def swap_think_token(_, __, kwargs, outputs, config: ElasticReasoningConfig) -> SamplerOutput | None:
    """
    A post-hook that swaps the last sampled token with the stop_think_token_id if the thinking budget is exceeded.

    Args:
        _: The module that is being hooked
        inputs: The arguments to the module
        outputs: The outputs of the module

    Returns:
        None
    """
    print(f"sampling_output: {outputs}")

    sampling_metadata: SamplingMetadata = kwargs.get("sampling_metadata", None)
    assert sampling_metadata is not None, "Sampling metadata is required for the `swap_think_token` post-hook"
    sampling_output: SamplerOutput | None = outputs

    if sampling_metadata:
        for seq_group, seq_outputs in zip(sampling_metadata.seq_groups, sampling_output.outputs):
            for seq_id, seq_data, seq_output in zip(seq_group.seq_data.keys(), seq_group.seq_data.values(), seq_outputs.samples):
                num_output_tokens = len(seq_data.output_token_ids) + 1  # Increment by 1 because the current token is not yet in the output
                if num_output_tokens == config.think_budget:
                    get_logger("INFER").debug(
                        f"Thinking budget reached for sequence {seq_id}. Replacing sampled token {seq_output.output_token} with {config.stop_think_token_id}"
                    )
                    seq_output.logprobs = {config.stop_think_token_id: seq_output.logprobs[seq_output.output_token]}
                    seq_output.output_token = config.stop_think_token_id

    print(f"sampling_output: {sampling_output}")

    return sampling_output


def setup_elastic_reasoning(config: ElasticReasoningConfig, llm: LLM):
    # Skip if disabled
    if not config.enable:
        return

    get_logger("INFER").info(
        f"Enabling elastic reasoning with think_budget={config.think_budget} and solution_budget={config.solution_budget}"
    )
    sampler: nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.sampler
    sampler.register_forward_hook(partial(swap_think_token, config=config), with_kwargs=True)
    get_logger("INFER").debug("Set up post-hook swap_think_token on sampler")
