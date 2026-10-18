"""The tie check that stands between Qwen3.5 and a silently dead output projection (#3700).

`load_dcp_from_hf` drops `lm_head.weight` from the checkpoint template whenever the config
declares tied embeddings, because HF checkpoints carry no separate head. The values then
reach the head only through the shared storage. A model that sets the flag without tying
keeps whatever `to_empty()` allocated, and training completes with uniform logits and zero
gradients everywhere upstream.

Deliberately CPU-marked: the check is pure parameter identity, so it needs no CUDA, and the
existing model coverage in `test_model.py` is gated behind `pytest.mark.gpu`, which is how
this class of bug reached a released version unnoticed.
"""

import pytest
import torch
from torch import nn

from prime_rl.trainer.model import assert_tied_word_embeddings


class TinyCausalLM(nn.Module):
    """The shape that matters: a nested embedding module and a separate `lm_head`."""

    def __init__(self, *, tied: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens


def test_accepts_a_head_that_shares_the_embedding():
    model = TinyCausalLM(tied=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight, "fixture must actually tie"

    assert_tied_word_embeddings(model)


def test_rejects_a_head_that_is_not_the_embedding():
    model = TinyCausalLM(tied=False)

    with pytest.raises(RuntimeError, match="issues/3700"):
        assert_tied_word_embeddings(model)


def test_rejects_an_equal_but_distinct_head():
    """Sharing values is not sharing storage: the load writes one parameter in place."""
    model = TinyCausalLM(tied=False)
    model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)

    with pytest.raises(RuntimeError, match="issues/3700"):
        assert_tied_word_embeddings(model)
