from prime_rl.inference.vllm.ttt import truncate_ttt_prompt_tokens


def test_truncate_ttt_prompt_tokens_leaves_completion_budget():
    tokens = list(range(20))

    truncated, did_truncate, prompt_budget, completion_budget = truncate_ttt_prompt_tokens(
        tokens,
        window_seq_len=10,
        max_model_len=10,
        requested_max_tokens=3,
    )

    assert did_truncate
    assert prompt_budget == 7
    assert completion_budget == 3
    assert truncated == list(range(13, 20))


def test_truncate_ttt_prompt_tokens_noops_inside_budget():
    tokens = list(range(6))

    truncated, did_truncate, prompt_budget, completion_budget = truncate_ttt_prompt_tokens(
        tokens,
        window_seq_len=10,
        max_model_len=10,
        requested_max_tokens=3,
    )

    assert not did_truncate
    assert prompt_budget == 7
    assert completion_budget == 3
    assert truncated == tokens


def test_truncate_ttt_prompt_tokens_caps_impossible_completion_budget():
    tokens = list(range(20))

    truncated, did_truncate, prompt_budget, completion_budget = truncate_ttt_prompt_tokens(
        tokens,
        window_seq_len=10,
        max_model_len=10,
        requested_max_tokens=50,
    )

    assert did_truncate
    assert prompt_budget == 1
    assert completion_budget == 9
    assert truncated == [19]
