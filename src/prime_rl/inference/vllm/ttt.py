def requested_completion_budget(max_completion_tokens: int | None, max_tokens: int | None) -> int | None:
    """Return the OpenAI/vLLM completion budget requested by the caller."""
    return max_completion_tokens if max_completion_tokens is not None else max_tokens


def truncate_ttt_prompt_tokens(
    tokens: list[int],
    *,
    window_seq_len: int | None,
    max_model_len: int,
    requested_max_tokens: int | None,
) -> tuple[list[int], bool, int, int]:
    """Keep only the active TTT prompt window while leaving room for generation.

    Verifiers keeps the logical full-history token list. The inference server
    cannot accept that once the trajectory grows past the physical vLLM window,
    so the token endpoint trims from the left just before scheduling.
    """
    if window_seq_len is None:
        return tokens, False, len(tokens), requested_max_tokens or 0

    if window_seq_len <= 1:
        raise ValueError(f"ttt_window_seq_len must be > 1, got {window_seq_len}.")

    physical_window = min(window_seq_len, max_model_len)
    completion_budget = requested_max_tokens if requested_max_tokens is not None else 1
    completion_budget = max(1, min(completion_budget, physical_window - 1))
    prompt_budget = max(1, physical_window - completion_budget)

    if len(tokens) <= prompt_budget:
        return tokens, False, prompt_budget, completion_budget

    return tokens[-prompt_budget:], True, prompt_budget, completion_budget
