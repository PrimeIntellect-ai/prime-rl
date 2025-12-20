"""Teacher model utilities for computing logprobs on generated rollouts via prefill inference."""

import asyncio
from itertools import cycle

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import PreTrainedTokenizerFast
from verifiers.utils.token_utils import prepare_sampling_args_for_token_prompts

from prime_rl.orchestrator.utils import get_semaphore
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger

_DUMMY_MESSAGES = [{"role": "user", "content": ""}]

_BASE_TEACHER_SAMPLING_ARGS = {
    "max_tokens": 1,
    "temperature": 1.0,
    "top_p": 1.0,
}


async def compute_teacher_logprobs_for_sample(
    client: AsyncOpenAI,
    model_name: str,
    sample: TrainingSample,
) -> list[float]:
    """Compute teacher logprobs for a single training sample via prefill."""
    semaphore = await get_semaphore()
    sampling_args = prepare_sampling_args_for_token_prompts(_BASE_TEACHER_SAMPLING_ARGS.copy())
    extra_body = sampling_args.pop("extra_body", {})
    tokens = sample.prompt_ids + sample.completion_ids
    body = {
        "model": model_name,
        "messages": _DUMMY_MESSAGES,
        "tokens": tokens,
        **sampling_args,
        **extra_body,
    }

    async with semaphore:
        response = await client.post("/chat/completions/tokens", body=body, cast_to=ChatCompletion)
    return [0.0 if lp is None else next(iter(lp.values())) for lp in response.prompt_logprobs]


async def compute_teacher_logprobs_for_batch(
    clients: list[AsyncOpenAI],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:
    """Compute teacher logprobs for a batch of training samples."""
    tasks = [
        compute_teacher_logprobs_for_sample(client, model_name, sample)
        for client, sample in zip(cycle(clients), samples)
    ]
    return await asyncio.gather(*tasks)


def validate_tokenizer_compatibility(
    main_tokenizer: PreTrainedTokenizerFast,
    teacher_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Validate that the main and teacher tokenizers are compatible."""
    logger = get_logger()

    if main_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size mismatch: main={main_tokenizer.vocab_size}, teacher={teacher_tokenizer.vocab_size}"
        )

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        main_val = getattr(main_tokenizer, attr)
        teacher_val = getattr(teacher_tokenizer, attr)
        if main_val != teacher_val:
            raise ValueError(f"Special token mismatch for {attr}: main={main_val}, teacher={teacher_val}")

    logger.info(f"Tokenizer compatibility validated: vocab_size={main_tokenizer.vocab_size}")
