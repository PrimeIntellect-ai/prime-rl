"""CPU-side model-input validation shared by the TTT engines."""

from __future__ import annotations

from typing import Any


def model_vocab_size(model: Any) -> int:
    """Return the loaded causal LM's vocabulary size."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("TTT model does not expose config; cannot validate incoming token ids")
    vocab_size = getattr(config, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError(f"TTT model config has invalid vocab_size={vocab_size!r}")

    return vocab_size


def validate_token_ids(token_ids: list[int], vocab_size: int, *, source: str = "token_ids") -> None:
    """Reject malformed/OOB ids while they are still a CPU list."""
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int:
            raise ValueError(f"{source}[{index}] must be an integer, got {type(token_id).__name__}")
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{source}[{index}]={token_id} is outside the model vocabulary [0, {vocab_size})")


def validate_qa_pairs(qa_pairs: list) -> None:
    """Shape-check Q&A pairs so a malformed one 409s (ValueError) instead of a KeyError
    escaping from tokenization as a 500."""
    for i, pair in enumerate(qa_pairs):
        if not isinstance(pair, dict) or not isinstance(pair.get("question"), str):
            raise ValueError(
                f"malformed qa_pairs[{i}]: expected a dict with a string 'question' "
                f"(and 'answer'), got {type(pair).__name__}"
            )
        if not isinstance(pair.get("answer", ""), str):
            raise ValueError(f"malformed qa_pairs[{i}]: 'answer' must be a string when present")
