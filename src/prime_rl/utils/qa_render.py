"""Standalone Q&A chat-template rendering with a token-prefix-stability guard.

The Q&A training paths (TTT adapter updates, policy recycling, meta lessons) all render a
[system?, user, assistant] conversation twice — full and prompt-only (with the generation
prompt) — and mask the loss to the answer suffix. That only works when the prompt render is
a *token prefix* of the full render. Some chat templates aren't prefix-stable (e.g.
DeepSeek-R1-Distill injects tokens around the generation prompt); silently falling back to
``prompt_len = 0`` would train loss on the full render (system + tools + question), so
unstable pairs are skipped with a warning instead.
"""

from __future__ import annotations

from prime_rl.utils.logger import get_logger

# Tokenizer names we already warned about — one loud warning per process, debug per pair.
_WARNED_TOKENIZERS: set[str] = set()


def _render(tokenizer, conversation: list[dict], generation_prompt: bool, template_kwargs: dict) -> list[int]:
    out = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=generation_prompt, **template_kwargs
    )
    # transformers returns a list of ids or (newer) a BatchEncoding.
    return list(out["input_ids"] if not isinstance(out, list) else out)


def render_qa_pair(tokenizer, conversation: list[dict], template_kwargs: dict) -> tuple[list[int], int] | None:
    """Render ``conversation`` (assistant answer last) full + prompt-only and return
    ``(full_ids, prompt_len)``, or None when the prompt render is not a token prefix of the
    full render — training on such a pair would put loss on the whole render."""
    full = _render(tokenizer, conversation, generation_prompt=False, template_kwargs=template_kwargs)
    prompt = _render(tokenizer, conversation[:-1], generation_prompt=True, template_kwargs=template_kwargs)
    if full[: len(prompt)] != prompt:
        name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
        logger = get_logger()
        if name not in _WARNED_TOKENIZERS:
            _WARNED_TOKENIZERS.add(name)
            logger.warning(
                f"Chat template of tokenizer '{name}' is not prefix-stable (prompt render is "
                "not a token prefix of the full render) — skipping Q&A pairs instead of "
                "training loss on the full render."
            )
        logger.debug(f"Skipping non-prefix-stable Q&A pair under tokenizer '{name}'.")
        return None
    return full, len(prompt)


def assert_prefix_stable_template(tokenizer, tools: list[dict] | None = None) -> None:
    """Startup canary: render a fixed [system, user, assistant] conversation — with and
    without a dummy tool — and raise ValueError if the prompt render is not a token prefix
    of the full render. Failing at launch beats silently skipping every Q&A pair."""
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
    ]
    dummy_tool = {
        "type": "function",
        "function": {"name": "noop", "description": "does nothing", "parameters": {"type": "object", "properties": {}}},
    }
    for kwargs in ({}, {"tools": tools or [dummy_tool]}):
        full = _render(tokenizer, conversation, generation_prompt=False, template_kwargs=kwargs)
        prompt = _render(tokenizer, conversation[:-1], generation_prompt=True, template_kwargs=kwargs)
        if full[: len(prompt)] != prompt:
            name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
            raise ValueError(
                f"Chat template of tokenizer '{name}' is not prefix-stable "
                f"({'with tools' if kwargs else 'without tools'}): the prompt-only render is "
                "not a token prefix of the full render, so answer-only loss masking for Q&A "
                "training is impossible. Use a prefix-stable chat template."
            )
