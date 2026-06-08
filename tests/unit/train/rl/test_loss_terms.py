"""Golden tests for the loss-term refactor.

``compute_loss`` now applies a *list* of loss terms and sums them; today there is
exactly one term per ``training_mode``. These tests pin that the refactor is
behavior-preserving: the summed result equals the per-sample core loss divided by
``loss_scale`` for every mode, with matching metrics. CPU-only (no GPU required).
"""

import pytest
import torch

from prime_rl.configs.losses import (
    CECoreConfig,
    DPPOKLCoreConfig,
    EchoAdvantageConfig,
    GRPOAdvantageConfig,
    LossTerm,
    RLLossConfig,
)
from prime_rl.trainer.rl.loss import (
    ExtraTerm,
    LossInputs,
    build_loss_terms,
    compute_loss,
    default_loss_fn,
    echo_loss_fn,
    opd_loss_fn,
    pg_loss_fn,
    setup_loss_fns,
    sft_loss_fn,
)


def _rl_term() -> LossTerm:
    """The default RL term; converts to ``RLLossConfig()`` for the direct-core reference."""
    return LossTerm(name="rl", loss=DPPOKLCoreConfig(), advantage=GRPOAdvantageConfig())


def _echo_term(roles: list[str], alpha: float) -> LossTerm:
    return LossTerm(name="echo", loss=CECoreConfig(), advantage=EchoAdvantageConfig(roles=roles, alpha=alpha))


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
    cfg = RLLossConfig()
    trainer, inference, teacher, advantages, loss_mask = _inputs([7, 5], seed=1)
    loss_scale = 12

    loss, metrics = compute_loss(
        trainer,
        inference,
        teacher,
        advantages,
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns([_rl_term()]),
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
        loss_fns=setup_loss_fns([_rl_term()]),
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
        loss_fns=setup_loss_fns([_rl_term()]),
        loss_scale=loss_scale,
        training_mode="opd",
    )

    expected = _reference_loss(opd_loss_fn, trainer, inference, teacher, advantages, loss_mask, loss_scale)
    assert torch.allclose(loss, expected)


def test_build_loss_terms_is_singleton():
    cores = setup_loss_fns([_rl_term()])
    terms = build_loss_terms("rl", cores)
    assert len(terms) == 1
    assert terms[0].name == "rl"
    assert terms[0].core is cores["rl"]


def test_build_loss_terms_unknown_mode_raises():
    cores = setup_loss_fns([_rl_term()])
    with pytest.raises(ValueError, match="No loss fn available"):
        build_loss_terms("nope", cores)


def test_echo_loss_fn_is_weighted_masked_nll():
    logprobs = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    weight = torch.tensor([0.5, 0.5, 2.0, 1.0])
    mask = torch.tensor([True, False, True, False])

    out = echo_loss_fn(LossInputs(logprobs, torch.zeros(4), None, weight, mask))

    # -(weight * logprobs)[mask].sum() = -((0.5 * -1) + (2.0 * -3)) = 6.5
    assert torch.allclose(out.loss, torch.tensor(6.5))
    assert torch.allclose(out.metrics["echo_token_count"], torch.tensor(2.0))


def test_pg_core_matches_default_loss_fn_at_rl_preset():
    """The rl preset of the parameterizable core is bit-identical to default_loss_fn (loss + metrics)."""
    g = torch.Generator().manual_seed(11)
    n = 12
    inputs = LossInputs(
        trainer_logprobs=torch.randn(n, generator=g, dtype=torch.float32),
        inference_logprobs=torch.randn(n, generator=g, dtype=torch.float32),
        teacher_logprobs=None,
        advantages=torch.randn(n, generator=g, dtype=torch.float32),
        loss_mask=torch.randint(0, 2, (n,), generator=g).bool(),
    )
    cfg = RLLossConfig()
    expected = default_loss_fn(inputs, cfg)
    got = pg_loss_fn(
        inputs, use_importance_ratio=True, clip=(cfg.dppo_mask_low, cfg.dppo_mask_high), kl_weight=cfg.kl_tau
    )
    assert torch.equal(got.loss, expected.loss)
    assert got.metrics.keys() == expected.metrics.keys()
    for k in expected.metrics:
        assert torch.equal(got.metrics[k], expected.metrics[k]), k


def test_pg_core_matches_echo_loss_fn_at_echo_preset():
    """The echo preset (no importance ratio, no clip, no KL) is bit-identical to echo_loss_fn."""
    g = torch.Generator().manual_seed(13)
    n = 10
    inputs = LossInputs(
        trainer_logprobs=torch.randn(n, generator=g, dtype=torch.float32),
        inference_logprobs=torch.randn(n, generator=g, dtype=torch.float32),
        teacher_logprobs=None,
        advantages=torch.randn(n, generator=g, dtype=torch.float32),
        loss_mask=torch.randint(0, 2, (n,), generator=g).bool(),
    )
    expected = echo_loss_fn(inputs)
    got = pg_loss_fn(inputs, use_importance_ratio=False, clip=None, kl_weight=0.0)
    assert torch.equal(got.loss, expected.loss)


def test_pg_core_matches_sft_loss_fn_at_sft_preset():
    """The sft preset (echo preset at unit weight) is bit-identical to sft_loss_fn."""
    g = torch.Generator().manual_seed(17)
    n = 9
    inputs = LossInputs(
        trainer_logprobs=torch.randn(n, generator=g, dtype=torch.float32),
        inference_logprobs=torch.zeros(n, dtype=torch.float32),
        teacher_logprobs=None,
        advantages=torch.ones(n, dtype=torch.float32),
        loss_mask=torch.randint(0, 2, (n,), generator=g).bool(),
    )
    expected = sft_loss_fn(inputs)
    got = pg_loss_fn(inputs, use_importance_ratio=False, clip=None, kl_weight=0.0)
    assert torch.equal(got.loss, expected.loss)


def test_extra_echo_term_adds_scaled_contribution():
    cfg = RLLossConfig()
    trainer, inference, teacher, advantages, loss_mask = _inputs([6, 4], seed=4)
    loss_scale, echo_scale = 9, 5

    g = torch.Generator().manual_seed(99)
    echo_masks = [torch.randint(0, 2, (n,), generator=g).bool() for n in (6, 4)]
    echo_weights = [torch.rand(n, generator=g, dtype=torch.float32) for n in (6, 4)]
    echo_term = ExtraTerm(name="echo", core=echo_loss_fn, scale=echo_scale, masks=echo_masks, weights=echo_weights)

    loss, metrics = compute_loss(
        trainer,
        inference,
        teacher,
        advantages,
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns([_rl_term()]),
        loss_scale=loss_scale,
        training_mode="rl",
        extra_terms=[echo_term],
    )

    rl_total = 0.0
    for t, i, te, a, m in zip(trainer, inference, teacher, advantages, loss_mask):
        rl_total = rl_total + default_loss_fn(LossInputs(t, i, te, a, m), cfg).loss
    echo_total = 0.0
    for t, i, te, em, ew in zip(trainer, inference, teacher, echo_masks, echo_weights):
        echo_total = echo_total + echo_loss_fn(LossInputs(t, i, te, ew, em)).loss

    expected = rl_total / loss_scale + echo_total / echo_scale
    assert torch.allclose(loss, expected)
    assert "echo/echo_nll" in metrics and "echo/echo_token_count" in metrics


def test_primary_lambda_scales_loss():
    """primary_lambda (default 1.0) linearly scales the primary term's contribution."""
    trainer, inference, teacher, advantages, loss_mask = _inputs([6, 4], seed=21)
    kwargs = dict(
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns([_rl_term()]),
        loss_scale=8,
        training_mode="rl",
    )
    base, _ = compute_loss(trainer, inference, teacher, advantages, **kwargs)
    doubled, _ = compute_loss(trainer, inference, teacher, advantages, primary_lambda=2.0, **kwargs)
    assert torch.allclose(doubled, 2.0 * base)


def test_custom_reduce_replaces_default_normalization():
    """A custom reduce overrides the default divide-by-loss_scale (proves the reduce seam)."""

    def sum_reduce(inputs):
        return sum(inputs.per_sample_losses)

    trainer, inference, teacher, advantages, loss_mask = _inputs([5, 5], seed=23)
    kwargs = dict(
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns([_rl_term()]),
        loss_scale=7,
        training_mode="rl",
    )
    summed, _ = compute_loss(trainer, inference, teacher, advantages, reduce=sum_reduce, **kwargs)
    normalized, _ = compute_loss(trainer, inference, teacher, advantages, **kwargs)
    assert torch.allclose(normalized, summed / 7)


def test_extra_terms_none_matches_rl_only():
    trainer, inference, teacher, advantages, loss_mask = _inputs([5, 5], seed=7)
    kwargs = dict(
        loss_mask=loss_mask,
        loss_fns=setup_loss_fns([_rl_term()]),
        loss_scale=8,
        training_mode="rl",
    )
    loss_default, _ = compute_loss(trainer, inference, teacher, advantages, **kwargs)
    loss_explicit_none, _ = compute_loss(trainer, inference, teacher, advantages, extra_terms=None, **kwargs)
    assert torch.allclose(loss_default, loss_explicit_none)


def test_no_rl_term_makes_rl_core_raise():
    # No primary term: the rl core must error when applied, not fabricate a default loss.
    loss_fns = setup_loss_fns([_echo_term(["assistant"], 0.5)])
    inputs = LossInputs(torch.zeros(3), torch.zeros(3), None, torch.zeros(3), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="no primary"):
        loss_fns["rl"](inputs)
