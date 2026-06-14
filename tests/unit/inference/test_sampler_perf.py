"""Equivalence tests for the vLLM 0.22 sampler hot-path patches.

Both patches must be bit-identical to the upstream implementations they
replace; these tests drive randomized inputs through old and new paths and
compare results exactly.
"""

import random

import numpy as np
import pytest
import torch

from prime_rl.inference.vllm.sampler_perf import (
    _PinnedStaging,
    build_output_tokens_fast,
    find_last_in_window,
)

VOCAB = 1000


class FakeInputBatch:
    def __init__(self, prompts: list[list[int]], outputs: list[list[int]], max_model_len: int = 256):
        n = len(prompts)
        self.max_num_reqs = n
        self.max_model_len = max_model_len
        self.token_ids_cpu = np.full((n, max_model_len), -7, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(n, dtype=np.int32)
        self.req_output_token_ids = outputs
        for i, (p, o) in enumerate(zip(prompts, outputs)):
            self.num_prompt_tokens[i] = len(p)
            self.token_ids_cpu[i, : len(p)] = p
            self.token_ids_cpu[i, len(p) : len(p) + len(o)] = o


def _upstream_convert(output_token_ids: list[list[int]], vocab_size: int) -> torch.Tensor:
    from vllm.v1.sample.ops.penalties import _convert_to_tensors

    t = _convert_to_tensors(output_token_ids, vocab_size, torch.device("cpu"))
    t.masked_fill_(t == -1, vocab_size)
    return t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the pinned-staging GPU copy path")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fast_output_tokens_matches_upstream(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 24)
    prompts = [[rng.randrange(VOCAB) for _ in range(rng.randint(1, 40))] for _ in range(n)]
    outputs = []
    for _ in range(n):
        out = [rng.randrange(VOCAB) for _ in range(rng.randint(0, 60))]
        # async placeholders: trailing -1s, occasionally an unrepaired row
        if out and rng.random() < 0.5:
            out.append(-1)
        if out and rng.random() < 0.1:
            out = [-1] * len(out)
        outputs.append(out)
    ib = FakeInputBatch(prompts, outputs)
    staging = _PinnedStaging(ib.max_num_reqs, ib.max_model_len)

    fast = build_output_tokens_fast(ib, staging, outputs, VOCAB, torch.device("cpu"))
    ref = _upstream_convert(outputs, VOCAB)
    assert fast is not None
    assert fast.shape == ref.shape
    assert torch.equal(fast, ref)


def test_fast_output_tokens_falls_back_on_foreign_rows():
    prompts = [[1, 2, 3]]
    outputs = [[4, 5]]
    ib = FakeInputBatch(prompts, outputs)
    staging = _PinnedStaging(ib.max_num_reqs, ib.max_model_len)
    # Spec-decode combine creates new list objects -> identity check must fail
    combined = [[4, 5, 6]]
    assert build_output_tokens_fast(ib, staging, combined, VOCAB, torch.device("cpu")) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises the pinned-staging GPU copy path")
def test_staging_double_buffer_reuse():
    ib = FakeInputBatch([[1]], [[2, 3]])
    staging = _PinnedStaging(4, 32)
    a = build_output_tokens_fast(ib, staging, ib.req_output_token_ids, VOCAB, torch.device("cpu"))
    b = build_output_tokens_fast(ib, staging, ib.req_output_token_ids, VOCAB, torch.device("cpu"))
    assert torch.equal(a, b)


@pytest.mark.parametrize("pattern_len", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_find_last_in_window_matches_full_scan(pattern_len, seed):
    rng = random.Random(seed * 17 + pattern_len)
    pattern = [rng.randrange(5) for _ in range(pattern_len)]
    for _ in range(200):
        lst = [rng.randrange(5) for _ in range(rng.randint(0, 50))]
        full = find_last_in_window(lst, pattern, 0, len(lst))
        # reference: brute force
        ref = -1
        for i in range(len(lst) - pattern_len + 1):
            if lst[i : i + pattern_len] == pattern:
                ref = i
        assert full == ref
        # windows tile the list with (m-1) overlap: max over windows == full scan
        if lst:
            cut = rng.randint(0, len(lst))
            lo2 = max(0, cut - (pattern_len - 1))
            w1 = find_last_in_window(lst, pattern, 0, cut)
            w2 = find_last_in_window(lst, pattern, lo2, len(lst))
            assert max(w1, w2) == ref


def _mk_holder():
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

    class RC:
        reasoning_start_token_ids = [7]
        reasoning_end_token_ids = [9]

    return ThinkingBudgetStateHolder(
        RC(), max_num_seqs=8, num_spec_tokens=0, device=torch.device("cpu"), is_pin_memory=False
    )


def _normalize(state: dict) -> dict:
    out = {
        k: v
        for k, v in state.items()
        if k not in ("_prime_scan_pos", "output_tok_ids", "prompt_tok_ids", "spec_token_ids")
    }
    # -2 sentinel is semantically "not found" == -1
    if out.get("end_thinking") == -2:
        out["end_thinking"] = -1
    return out


@pytest.fixture(scope="module")
def think_state_impls():
    import vllm.v1.sample.thinking_budget_state as tbs

    from prime_rl.inference.vllm.sampler_perf import _patch_thinking_budget_scan

    orig_fn = tbs.ThinkingBudgetStateHolder._update_think_state
    _patch_thinking_budget_scan()
    patched_fn = tbs.ThinkingBudgetStateHolder._update_think_state
    assert patched_fn is not orig_fn
    yield orig_fn, patched_fn
    tbs.ThinkingBudgetStateHolder._update_think_state = orig_fn


@pytest.mark.parametrize("seed", list(range(8)))
def test_thinking_budget_incremental_scan_equivalence(seed, think_state_impls):
    orig_fn, patched_fn = think_state_impls

    rng = random.Random(seed)
    holder = _mk_holder()
    budget = rng.choice([3, 8, 20])
    prompt = [rng.randrange(5) for _ in range(rng.randint(1, 10))]
    if rng.random() < 0.3:
        prompt += [7]  # think starts in prompt (continue_thinking)

    state_a = holder._init_state_entry(list(prompt), budget)
    state_b = holder._init_state_entry(list(prompt), budget)
    out_a: list[int] = []
    out_b: list[int] = []
    state_a["output_tok_ids"] = out_a
    state_b["output_tok_ids"] = out_b

    # random token stream with think start/end injected at random points
    stream = []
    for _ in range(rng.randint(5, 60)):
        r = rng.random()
        if r < 0.06:
            stream.append(7)
        elif r < 0.12:
            stream.append(9)
        else:
            stream.append(rng.randrange(5))

    for tok in stream:
        out_a.append(tok)
        out_b.append(tok)
        state_a["force_index"] = []
        state_b["force_index"] = []
        orig_fn(holder, state_a)
        patched_fn(holder, state_b)
        assert _normalize(state_a) == _normalize(state_b), f"divergence after {len(out_a)} tokens (stream={stream})"
