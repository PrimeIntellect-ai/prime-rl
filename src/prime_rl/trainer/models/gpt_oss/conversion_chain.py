"""GPT-OSS conversion chain — empty.

The custom prime-rl GPT-OSS implementation mirrors HuggingFace's parameter
naming exactly, so the HF<->prime conversion is the identity (no ops)."""

from __future__ import annotations

from prime_rl.trainer.models.conversion_ops import ConvOp


def build_gpt_oss_chain(num_layers: int = 0) -> list[ConvOp]:
    return []
