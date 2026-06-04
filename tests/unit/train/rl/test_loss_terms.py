"""Golden tests for the loss-term refactor.

``compute_loss`` now applies a *list* of loss terms and sums them; today there is
exactly one term per ``training_mode``. These tests pin that the refactor is
behavior-preserving: the summed result equals the per-sample core loss divided by
``loss_scale`` for every mode, with matching metrics. CPU-only (no GPU required).
"""

import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    build_loss_terms,
    compute_loss,
    default_loss_fn,
    opd_loss_fn,
    setup_loss_fns,
    sft_loss_fn,
)


def _inputs(seq_lens: list[int], seed: int):
    g = torch.Generator().manual_seed(seed)
    trainer = [torch.randn(n, generator=g, dtype=torch.float32) for n in seq_lens]
    inference = [torch.randn(n, generator=g, dtype=torch.float32) for n in seq_lens]
    teacher = [torch.randn(n, generator=g, dtype=torch.float32) for n in seq_lens]
    advantages = [torch.randn(n, generator=g, dtype=torch.float32) for n in seq_lens]
    # Mixed mask so the masking logic is exercised.
    loss_mask = [torch.randint(0, 2, (n,), generator=g).bool() for n in seq_lens]
    return trainer, inference, teacher, advantages, loss_mask


def _reference_loss(core, trainer, inference, teacher, advantages, loss_mask, loss_scale):
    """Sum the per-sample core loss the same way compute_loss does, then scale."""
    total = 0.0
    for t, i, te, a, m in zip(trainer, inference, teacher, advantages, loss_mask):
        total = total + core(LossInputs(t, i, te, a, m)).loss
    return total / loss_scale


def test_rl_term_matches_direct_core():
    cfg = DefaultLossConfig()
    trainer, inference, teacher, advantages, loss_mask = _inputs([7, 5], seed=1)
    loss_scale = 12

    loss, metrics = compute_loss(
        trainer,
        inference,
        teacher,
        advantages,
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns(cfg),
        loss_scale=loss_scale,
        training_mode="rl",
    )

    expected = _reference_loss(
        lambda x: default_loss_fn(x, cfg), trainer, inference, teacher, advantages, loss_mask, loss_scale
    )
    assert torch.allclose(loss, expected)

    # Metrics are stacked per-sample; they must match the direct core calls.
    expected_is_masked = torch.stack(
        [
            default_loss_fn(LossInputs(t, i, te, a, m), cfg).metrics["is_masked"]
            for t, i, te, a, m in zip(trainer, inference, teacher, advantages, loss_mask)
        ]
    )
    assert torch.allclose(metrics["is_masked"], expected_is_masked)


def test_sft_term_matches_direct_core():
    trainer, inference, teacher, advantages, loss_mask = _inputs([6, 4], seed=2)
    loss_scale = 10

    loss, _ = compute_loss(
        trainer,
        inference,
        teacher,
        advantages,
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns(DefaultLossConfig()),
        loss_scale=loss_scale,
        training_mode="sft",
    )

    expected = _reference_loss(sft_loss_fn, trainer, inference, teacher, advantages, loss_mask, loss_scale)
    assert torch.allclose(loss, expected)


def test_opd_term_matches_direct_core():
    trainer, inference, teacher, advantages, loss_mask = _inputs([8, 3], seed=3)
    loss_scale = 11

    loss, _ = compute_loss(
        trainer,
        inference,
        teacher,
        advantages,
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns(DefaultLossConfig()),
        loss_scale=loss_scale,
        training_mode="opd",
    )

    expected = _reference_loss(opd_loss_fn, trainer, inference, teacher, advantages, loss_mask, loss_scale)
    assert torch.allclose(loss, expected)


def test_build_loss_terms_is_singleton():
    cores = setup_loss_fns(DefaultLossConfig())
    terms = build_loss_terms("rl", cores)
    assert len(terms) == 1
    assert terms[0].name == "rl"
    assert terms[0].core is cores["rl"]


def test_build_loss_terms_unknown_mode_raises():
    cores = setup_loss_fns(DefaultLossConfig())
    with pytest.raises(ValueError, match="No loss fn available"):
        build_loss_terms("nope", cores)
