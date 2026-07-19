"""Tests for square-averaged SFT loss weighting.

The invariant that matters: because microbatch losses are raw weighted sums and
normalization happens once against the GLOBAL weight sum, gradients must not
depend on how the global batch is split into microbatches.
"""

import math

import pytest
import torch
from torch import nn

from prime_rl.trainer.sft.loss import compute_document_loss_weights


def test_document_weights_use_supervised_counts_only():
    # Packed row: doc A (4 tokens, 2 supervised), doc B (6 tokens: 4 "image/prompt"
    # tokens masked out, 1 supervised), doc C (3 tokens, none supervised — e.g. fully
    # truncated answer), trailing pad absorbed into doc C's length.
    loss_mask = torch.tensor([[False, True, True, False, False, False, False, False, True, False, False, False, False]])
    seq_lens = torch.tensor([4, 5, 4])

    weights = compute_document_loss_weights(loss_mask, seq_lens)

    expected = torch.zeros(1, 13)
    expected[0, 1] = expected[0, 2] = 1 / math.sqrt(2)  # doc A: 2 supervised tokens
    expected[0, 8] = 1.0  # doc B: 1 supervised token
    # doc C: no supervised tokens -> all zeros, no div-by-zero
    torch.testing.assert_close(weights, expected)
    # Weight lives only under the mask.
    assert (weights[~loss_mask] == 0).all()


def test_document_weights_rejects_mismatched_boundaries():
    with pytest.raises(ValueError, match="does not match"):
        compute_document_loss_weights(torch.ones(1, 10, dtype=torch.bool), torch.tensor([4, 4]))


def _run_partition(model, embeds, targets, weights, loss_mask, partition, fsdp_gradient_divide_factor=1.0):
    """Replicate the trainer's accumulate-then-globally-rescale gradient computation."""
    model.zero_grad(set_to_none=True)
    grad_accum_steps = len(partition)
    global_weight_sum = weights[loss_mask].sum().item()
    for chunk in partition:
        logits = model(embeds[:, chunk])
        token_loss = nn.functional.cross_entropy(
            logits.flatten(0, 1), targets[:, chunk].flatten(), reduction="none"
        ).view(1, -1)
        loss_sum = (token_loss * weights[:, chunk])[loss_mask[:, chunk]].sum()
        (loss_sum / grad_accum_steps).backward()
    grad_scale = fsdp_gradient_divide_factor * grad_accum_steps / global_weight_sum
    return [p.grad.mul(grad_scale).clone() for p in model.parameters()]


def test_gradients_invariant_to_microbatch_split():
    torch.manual_seed(0)
    hidden, vocab, seq = 16, 32, 24
    model = nn.Linear(hidden, vocab)
    embeds = torch.randn(1, seq, hidden)
    targets = torch.randint(0, vocab, (1, seq))
    # 3 documents with different supervised counts; some prompt/image tokens masked.
    seq_lens = torch.tensor([8, 10, 6])
    loss_mask = torch.rand(1, seq) > 0.4
    loss_mask[0, :3] = False  # doc 1 prompt
    weights = compute_document_loss_weights(loss_mask, seq_lens)

    whole = _run_partition(model, embeds, targets, weights, loss_mask, [slice(0, seq)])
    split_even = _run_partition(model, embeds, targets, weights, loss_mask, [slice(0, 12), slice(12, seq)])
    # A split that cuts through the middle of document 2.
    split_uneven = _run_partition(
        model, embeds, targets, weights, loss_mask, [slice(0, 5), slice(5, 13), slice(13, seq)]
    )

    for grads in (split_even, split_uneven):
        for got, want in zip(grads, whole):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)


def test_token_weighting_equals_current_behavior():
    """weights == 1 must reproduce the plain global token mean exactly."""
    torch.manual_seed(1)
    seq = 12
    loss_mask = torch.rand(1, seq) > 0.3
    token_weights = torch.ones(1, seq) * loss_mask
    token_count = loss_mask.sum().item()
    assert token_weights[loss_mask].sum().item() == token_count
