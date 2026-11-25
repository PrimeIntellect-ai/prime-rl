"""Tiny Qwen3.5 checkpoints that answer through their LM head, and their chat tokenizer, for the tests of the strategies
that read it."""

import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from tests.unit.train.models.qwen3_5.clef_flash_release import tokenizer, write_backbone

# ChatML turns. With thinking off the prompt ends with an empty think block, as Qwen3.5's does in vLLM's tests
# (tests/entrypoints/generate/structured_decisions/test_slot_strategy.py).
CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)
# "yes" and "no" as one token each, as a noul's labels must be
MERGES = (("y", "e"), ("ye", "s"), ("n", "o"))
# What vLLM main serves as a slot decision model, with as many slots as the tokenizer has one-letter labels
MODELING_OVERRIDES = {
    "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
    "readout": "slot_lm",
    "n_slots": 26,
}


def chat_tokenizer(merges: tuple[tuple[str, str], ...] = MERGES) -> PreTrainedTokenizerFast:
    tokens = tokenizer(merges)
    tokens.chat_template = CHAT_TEMPLATE
    return tokens


def write_lm_head_release(directory: Path, modeling_overrides: dict | None = None) -> None:
    """A tiny stock Qwen3.5 checkpoint, or a slot decision model with ``modeling_overrides`` in its config.json."""
    write_backbone(directory)
    if modeling_overrides is not None:
        path = directory / "config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "modeling_overrides": modeling_overrides}))
    chat_tokenizer().save_pretrained(directory)
