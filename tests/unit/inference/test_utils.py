import pytest
from transformers import AutoTokenizer

from zeroband.inference.utils import format_prompts


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "deepseek-ai/DeepSeek-R1-0528",
        "PrimeIntellect/INTELLECT-1-Instruct",
    ],
)
def test_format_prompts_single_bos_token(tokenizer_name: str) -> None:
    """Test that format_prompts results in only one BOS token per sequence when tokenized."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompts = [
        "Prove that 1 + 1 = 2",
        "Prove that 1 + 1 = 3",
    ]

    # Format prompts with thinking enabled
    formatted_prompts = format_prompts(
        prompts=prompts, target_lengths=[100] * len(prompts), len_rewards_config=None, tokenizer=tokenizer, enable_thinking=True
    )

    # Tokenize the formatted prompts
    tokenized = tokenizer(formatted_prompts)
    assert tokenizer.bos_token_id is not None, "BOS token id is not set"
    bos_token_id = tokenizer.bos_token_id

    # Check each sequence has exactly one BOS token
    for i, input_ids in enumerate(tokenized["input_ids"]):
        bos_count = input_ids.count(bos_token_id)
        bos_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == bos_token_id]
        assert bos_count == 1, f"Prompt {i} has {bos_count} BOS tokens at positions {bos_positions}, expected 1"


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "deepseek-ai/DeepSeek-R1-0528",
        "PrimeIntellect/INTELLECT-1-Instruct",
    ],
)
def test_format_prompts_tokenization_consistency(tokenizer_name: str) -> None:
    """Test that tokenized outputs are identical whether tokenization is done inside or outside format_prompts."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompts = [
        "Prove that 1 + 1 = 2",
        "What is the capital of France?",
    ]

    # Test without length rewards
    formatted_prompts = format_prompts(
        prompts=prompts,
        target_lengths=[-1] * len(prompts),
        len_rewards_config=None,
        tokenizer=tokenizer,
        enable_thinking=True,
    )

    # Tokenize externally
    external_tokenized = tokenizer(formatted_prompts)["input_ids"]

    # Get tokenized output directly from format_prompts
    internal_tokenized = format_prompts(
        prompts=prompts,
        target_lengths=[-1] * len(prompts),
        len_rewards_config=None,
        tokenizer=tokenizer,
        enable_thinking=True,
        tokenize=True,  # Get tokenized output
    )

    # Compare tokenized outputs
    for _external_tokenized, _internal_tokenized in zip(external_tokenized, internal_tokenized):
        assert _external_tokenized == _internal_tokenized, "Tokenized outputs should be identical"
