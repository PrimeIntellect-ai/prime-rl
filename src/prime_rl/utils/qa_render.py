"""Standalone Q&A chat-template rendering with a token-prefix-stability guard.

The Q&A training paths (TTT adapter updates, policy recycling, meta lessons) render a
[system?, user, assistant] conversation full and prompt-only and mask loss to the answer
suffix — valid only when the prompt render is a token prefix of the full render, so
non-prefix-stable pairs are skipped with a warning instead of training on the full render.
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
        name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
        try:
            full = _render(tokenizer, conversation, generation_prompt=False, template_kwargs=kwargs)
            prompt = _render(tokenizer, conversation[:-1], generation_prompt=True, template_kwargs=kwargs)
        except Exception as e:
            # Some templates reject the canary's fixture itself (no system-role support,
            # no tools support). That still means the Q&A rendering paths can't work as
            # configured — but say so, instead of a bare jinja traceback.
            raise ValueError(
                f"Chat template of tokenizer '{name}' failed the Q&A rendering canary "
                f"({'with tools' if kwargs else 'without tools'}): {type(e).__name__}: {e}. "
                "TTT Q&A training renders [system, question, answer] conversations "
                "(with the rollout's tool schemas when present) — the template must accept "
                "them."
            ) from e
        if full[: len(prompt)] != prompt:
            raise ValueError(
                f"Chat template of tokenizer '{name}' is not prefix-stable "
                f"({'with tools' if kwargs else 'without tools'}): the prompt-only render is "
                "not a token prefix of the full render, so answer-only loss masking for Q&A "
                "training is impossible. Use a prefix-stable chat template."
            )


def tokenize_qa_pairs(
    tokenizer,
    qa_pairs: list[dict],
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
) -> list[tuple[list[int], list[bool]]]:
    """Render each Q&A pair standalone with the chat template — no branch context (the
    knowledge must come from the weights), but conditioned on the rollout's system prompt
    and tool schemas so tool lessons are learned next to the tool descriptions. Loss on
    the answer tokens only. Pairs whose answer renders to nothing are skipped."""
    template_kwargs: dict = {"tools": tools} if tools else {}
    head = [{"role": "system", "content": system_prompt}] if system_prompt else []
    sequences: list[tuple[list[int], list[bool]]] = []
    for pair in qa_pairs:
        if not str(pair.get("answer", "")).strip():
            continue  # a blank answer renders only template scaffold — nothing to learn
        conversation = [
            *head,
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
        rendered = render_qa_pair(tokenizer, conversation, template_kwargs)
        if rendered is None:
            continue  # non-prefix-stable render: skip rather than train on the full render
        full, prompt_len = rendered
        if len(full) - prompt_len < 1:
            continue
        sequences.append((full, [False] * prompt_len + [True] * (len(full) - prompt_len)))
    return sequences


def qa_pairs_to_ce_samples(
    pairs: list[dict],
    system_prompt: str | None,
    tools: list[dict] | None,
    tokenizer,
    env_name: str,
) -> "list":
    """Render Q&A pairs as ce-routed ``TrainingSample``s: answer-token NLL only
    (``rl_weights`` all zero — no advantages, no importance ratio; temperatures 1.0 —
    ce NLL is temperature-free MLE)."""
    from prime_rl.transport import TrainingSample

    # Coerce untrusted recorded pairs; tokenize_qa_pairs indexes "question" directly.
    pairs = [{"question": str(p.get("question", "")), "answer": str(p.get("answer", ""))} for p in pairs]
    samples: list[TrainingSample] = []
    for full, mask in tokenize_qa_pairs(tokenizer, pairs, system_prompt, tools):
        samples.append(
            TrainingSample(
                token_ids=full,
                mask=mask,
                logprobs=[0.0] * len(full),
                temperatures=[1.0] * len(full),
                env_name=env_name,
                rl_weights=[0.0] * len(full),
                ce_weights=[1.0 if m else 0.0 for m in mask],
            )
        )
    return samples
