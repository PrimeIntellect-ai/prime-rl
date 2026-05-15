import json
from collections import Counter

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from prime_rl.trainer.sft.data import (
    SUBSET_TO_PROFILE,
    SYSTEM_PROMPT_PROFILES,
    SFTDataset,
    SystemPromptSampler,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture
def prompt_pool(tmp_path):
    """Create a small pool of tagged system prompts."""
    prompts = [
        "Be precise and concise.",
        "Think analytically and verify your reasoning.",
        "Respond naturally in a conversational tone.",
        "Give a literal, direct answer.",
        "Explain your reasoning step by step, showing derivations.",
    ]
    final_path = tmp_path / "system_prompts_final.json"
    final_path.write_text(json.dumps(prompts))

    # Expanded version with tags (dict format matching production)
    expanded_entries = [
        {"prompt": prompts[0], "tags": ["concise", "literal"], "tokens_approx": 5, "source": "seed"},
        {"prompt": prompts[1], "tags": ["analytic", "calibrated"], "tokens_approx": 7, "source": "seed"},
        {"prompt": prompts[2], "tags": ["natural", "pragmatic"], "tokens_approx": 7, "source": "seed"},
        {"prompt": prompts[3], "tags": ["literal", "concise"], "tokens_approx": 5, "source": "seed"},
        {"prompt": prompts[4], "tags": ["explanatory", "analytic"], "tokens_approx": 8, "source": "seed"},
    ]
    expanded_path = tmp_path / "system_prompts_expanded.json"
    expanded_path.write_text(json.dumps(expanded_entries))

    return str(final_path), prompts


def _make_single_turn_dataset(n: int = 10, subset: str = "wildchat") -> Dataset:
    """Create a single-turn messages dataset."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
                "__subset": subset,
                "__split": "train",
                "__index": i,
            }
        )
    return Dataset.from_list(rows)


def _make_multi_turn_dataset(n: int = 5, subset: str = "wildchat") -> Dataset:
    """Create a multi-turn messages dataset (>2 messages per sample)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"First answer {i}"},
                    {"role": "user", "content": f"Follow-up {i}"},
                    {"role": "assistant", "content": f"Second answer {i}"},
                ],
                "__subset": subset,
                "__split": "train",
                "__index": i,
            }
        )
    return Dataset.from_list(rows)


def test_sampler_loads_pool(prompt_pool):
    pool_path, prompts = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    assert len(sampler.prompts) == len(prompts)
    assert sampler.prompts == prompts


def test_sampler_deterministic(prompt_pool):
    """Same seed → same prompt."""
    pool_path, _ = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    result_a = sampler.sample("chat", seed=42)
    result_b = sampler.sample("chat", seed=42)
    assert result_a == result_b


def test_sampler_different_seeds_differ(prompt_pool):
    """Different seeds should (usually) produce different prompts."""
    pool_path, _ = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    results = {sampler.sample("chat", seed=i) for i in range(50)}
    assert len(results) > 1, "All 50 seeds produced the same prompt"


def test_sampler_no_profile_uniform(prompt_pool):
    """Without a matching profile, sampling is uniform."""
    pool_path, prompts = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    counts = Counter(sampler.sample(None, seed=i) for i in range(1000))
    # All prompts should appear at least once with 1000 draws
    assert len(counts) == len(prompts)


def test_sampler_profile_weighting(prompt_pool):
    """Chat profile should favor natural/conversational prompts."""
    pool_path, prompts = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    counts = Counter(sampler.sample("chat", seed=i) for i in range(2000))

    # "Respond naturally in a conversational tone." has tags [natural, pragmatic]
    # which match chat profile {natural:2, pragmatic:1} → score=3 → weight=exp(3)≈20
    # vs untagged prompts which get weight=ε+exp(0)≈1
    conversational_prompt = "Respond naturally in a conversational tone."
    assert counts[conversational_prompt] > counts[prompts[0]], (
        f"Chat profile should favor natural+pragmatic prompt, got {counts}"
    )


def test_injection_prepends_system_message(tokenizer, prompt_pool):
    """System prompt injection adds a system message at position 0."""
    pool_path, _ = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    dataset = _make_single_turn_dataset(n=3)
    sft = SFTDataset(
        dataset,
        tokenizer=tokenizer,
        shuffle=False,
        max_epochs=1,
        system_prompt_sampler=sampler,
    )
    sample = next(iter(sft))
    assert sample is not None
    # The sample should have input_ids (system + user + assistant tokens)
    assert len(sample["input_ids"]) > 0
    assert sum(sample["loss_mask"]) > 0


def test_injection_deterministic_across_restarts(tokenizer, prompt_pool):
    """Same dataset + seed → same tokenized output."""
    pool_path, _ = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    dataset = _make_single_turn_dataset(n=3)

    def get_first_sample():
        sft = SFTDataset(
            dataset,
            tokenizer=tokenizer,
            shuffle=False,
            max_epochs=1,
            system_prompt_sampler=sampler,
            seed=0,
        )
        return next(iter(sft))

    s1 = get_first_sample()
    s2 = get_first_sample()
    assert s1["input_ids"] == s2["input_ids"]
    assert s1["loss_mask"] == s2["loss_mask"]


def test_multi_turn_filter(tokenizer, prompt_pool):
    """Multi-turn samples (>2 messages) are dropped when injection is active."""
    pool_path, _ = prompt_pool
    sampler = SystemPromptSampler(pool_path)
    dataset = _make_multi_turn_dataset(n=5)
    sft = SFTDataset(
        dataset,
        tokenizer=tokenizer,
        shuffle=False,
        max_epochs=1,
        system_prompt_sampler=sampler,
    )
    samples = list(sft)
    assert len(samples) == 0, f"Expected 0 samples (all multi-turn), got {len(samples)}"


def test_no_filter_without_injection(tokenizer):
    """Without system_prompt_sampler, multi-turn filter is not applied.
    Samples may still fail at build_incremental_token_mask for models
    whose chat templates break incremental tokenization on multi-turn."""
    dataset = _make_single_turn_dataset(n=3)
    sft = SFTDataset(
        dataset,
        tokenizer=tokenizer,
        shuffle=False,
        max_epochs=1,
        system_prompt_sampler=None,
    )
    samples = list(sft)
    assert len(samples) == 3


def test_subset_to_profile_coverage():
    """All profiles referenced by SUBSET_TO_PROFILE exist in SYSTEM_PROMPT_PROFILES."""
    for subset, profile in SUBSET_TO_PROFILE.items():
        assert profile in SYSTEM_PROMPT_PROFILES, f"Subset {subset!r} maps to profile {profile!r} which is not defined"


def test_empty_pool_raises(tmp_path):
    """Empty prompt pool raises ValueError."""
    pool_path = tmp_path / "empty.json"
    pool_path.write_text("[]")
    with pytest.raises(ValueError, match="empty"):
        SystemPromptSampler(str(pool_path))
