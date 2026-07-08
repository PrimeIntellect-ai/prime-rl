"""qa_render: prefix-stability guard on standalone Q&A chat-template rendering.

A prompt-only render must be a token prefix of the full render for answer-only loss
masking; templates that inject tokens around the generation prompt (DeepSeek-R1-style)
must be skipped (per pair) or rejected at launch (canary), never silently trained on the
full render."""

import pytest

from prime_rl.utils.qa_render import assert_prefix_stable_template, render_qa_pair


class StableTokenizer:
    """Prefix-stable stand-in: 2 tokens per message, +1 for the generation prompt, and the
    generation-prompt token equals the next message's first scaffold token."""

    name_or_path = "stable-fake"

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, tools=None):
        n = 2 * len(conversation) + (1 if add_generation_prompt else 0)
        return list(range(n))


class UnstableTokenizer:
    """DeepSeek-R1-style: the generation prompt appends an extra token that the full render
    does not contain, so the prompt render is NOT a prefix of the full render."""

    name_or_path = "unstable-fake"

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, tools=None):
        ids = list(range(2 * len(conversation)))
        if add_generation_prompt:
            ids.append(999)  # injected token — breaks the prefix property
        return ids


CONVERSATION = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a"},
]


def test_render_qa_pair_prefix_stable():
    full, prompt_len = render_qa_pair(StableTokenizer(), CONVERSATION, {})
    assert full == list(range(4))
    assert prompt_len == 3  # [user] render (2) + generation prompt (1)


def test_render_qa_pair_unstable_returns_none():
    assert render_qa_pair(UnstableTokenizer(), CONVERSATION, {}) is None


def test_render_qa_pair_normalizes_batch_encoding():
    class DictTokenizer(StableTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            return {"input_ids": super().apply_chat_template(conversation, **kwargs)}

    full, prompt_len = render_qa_pair(DictTokenizer(), CONVERSATION, {})
    assert full == list(range(4)) and prompt_len == 3


def test_assert_prefix_stable_template():
    assert_prefix_stable_template(StableTokenizer())  # no raise
    with pytest.raises(ValueError, match="not prefix-stable"):
        assert_prefix_stable_template(UnstableTokenizer())


def test_assert_prefix_stable_template_wraps_template_errors():
    """Templates that reject the canary fixture itself (no system role, no tools) must
    fail with a self-explaining ValueError, not a bare jinja traceback."""

    class NoSystemTokenizer(StableTokenizer):
        name_or_path = "no-system-fake"

        def apply_chat_template(self, conversation, **kwargs):
            if any(m["role"] == "system" for m in conversation):
                raise RuntimeError("System role not supported")
            return super().apply_chat_template(conversation, **kwargs)

    with pytest.raises(ValueError, match="failed the Q&A rendering canary.*System role"):
        assert_prefix_stable_template(NoSystemTokenizer())
