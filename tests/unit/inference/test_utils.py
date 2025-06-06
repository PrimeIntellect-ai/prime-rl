import pytest
from transformers import AutoTokenizer

from zeroband.inference.utils import format_prompts


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "deepseek-ai/DeepSeek-R1-0528",
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
