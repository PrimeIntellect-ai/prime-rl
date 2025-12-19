"""Teacher model utilities for computing logprobs on generated rollouts via prefill inference."""

import asyncio
from itertools import cycle

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from prime_rl.orchestrator.utils import get_semaphore
from prime_rl.transport import TrainingSample
from prime_rl.utils.logger import get_logger


# Sampling args for teacher prefill: generate 1 token to get prompt logprobs
TEACHER_SAMPLING_ARGS = {
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs": True,
    "top_p": 1.0,
    "extra_body": {
        "return_token_ids": True,
        "prompt_logprobs": True,
        "top_k": -1,
        "min_p": 0.0,
    },
}


def extract_logprobs_from_response(response: ChatCompletion) -> list[float]:
    """Extract logprobs from a ChatCompletion response."""
    if not response.choices or response.choices[0].logprobs is None:
        return []
    
    content = response.choices[0].logprobs.content
    if content is None:
        return []
    
    return [entry.logprob if entry.logprob is not None else 0.0 for entry in content]


async def compute_teacher_logprobs_for_sample(
    client: AsyncOpenAI,
    model_name: str,
    tokenizer: PreTrainedTokenizerFast,
    sample: TrainingSample,
) -> list[float]:
    """Compute teacher logprobs for a single training sample via prefill."""
    logger = get_logger()
    semaphore = await get_semaphore()
    completion_len = len(sample.completion_ids)
    
    # Decode full sequence and send to teacher
    full_text = tokenizer.decode(sample.prompt_ids + sample.completion_ids, skip_special_tokens=False)
    messages = [{"role": "user", "content": full_text}]
    
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                **TEACHER_SAMPLING_ARGS,
            )
        
        logprobs = extract_logprobs_from_response(response)
        
        # Pad or truncate to match completion length
        if len(logprobs) >= completion_len:
            return logprobs[:completion_len]
        return logprobs + [0.0] * (completion_len - len(logprobs))
            
    except Exception as e:
        logger.error(f"Error computing teacher logprobs: {e}")
        return [0.0] * completion_len


async def compute_teacher_logprobs_for_batch(
    clients: list[AsyncOpenAI],
    model_name: str,
    tokenizer: PreTrainedTokenizerFast,
    samples: list[TrainingSample],
    pbar_description: str = "Computing teacher logprobs",
) -> list[list[float]]:
    """Compute teacher logprobs for a batch of training samples."""
    pbar = tqdm(total=len(samples), desc=pbar_description)
    
    async def compute_with_progress(client: AsyncOpenAI, sample: TrainingSample) -> list[float]:
        result = await compute_teacher_logprobs_for_sample(client, model_name, tokenizer, sample)
        pbar.update(1)
        return result
    
    try:
        return await asyncio.gather(*[
            compute_with_progress(client, sample)
            for client, sample in zip(cycle(clients), samples)
        ])
    finally:
        pbar.close()


def validate_tokenizer_compatibility(
    main_tokenizer: PreTrainedTokenizerFast,
    teacher_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Validate that the main and teacher tokenizers are compatible."""
    logger = get_logger()
    
    # Check vocab size
    if main_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size mismatch: main={main_tokenizer.vocab_size}, "
            f"teacher={teacher_tokenizer.vocab_size}"
        )
    
    # Check special tokens
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        main_val = getattr(main_tokenizer, attr)
        teacher_val = getattr(teacher_tokenizer, attr)
        if main_val != teacher_val:
            raise ValueError(f"Special token mismatch for {attr}: main={main_val}, teacher={teacher_val}")
    
    logger.info(f"Tokenizer compatibility validated: vocab_size={main_tokenizer.vocab_size}")
