import math

import pytest
import torch

from prime_rl.configs.trainer import DefaultLossConfig, IPOLossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    compute_importance_ratio_and_mismatch_kl,
    default_loss_fn,
    ipo_loss_fn,
    ref_kl_loss_fn,
)


def _inputs(
    trainer_logprob: float,
    inference_logprob: float,
    *,
    advantage: float = 1.0,
    ref_logprob: float | None = None,
) -> LossInputs:
    return LossInputs(
        trainer_logprobs=torch.tensor([trainer_logprob], dtype=torch.float32),
        inference_logprobs=torch.tensor([inference_logprob], dtype=torch.float32),
        ref_logprobs=None if ref_logprob is None else torch.tensor([ref_logprob], dtype=torch.float32),
        advantages=torch.tensor([advantage], dtype=torch.float32),
        loss_mask=torch.tensor([True]),
    )


def _assert_finite_output(loss, metrics):
    assert torch.isfinite(loss)
    for value in metrics.values():
        assert torch.isfinite(value).all()


def test_mismatch_kl_stays_finite_for_overflow_log_ratio():
    log_ratio, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        torch.tensor([-2.0], dtype=torch.float32),
        torch.tensor([-91.0], dtype=torch.float32),
    )

    assert log_ratio.item() == pytest.approx(89.0)
    assert torch.isfinite(ratio).all()
    assert torch.isfinite(mismatch_kl).all()
    assert (mismatch_kl >= 0).all()


def test_mismatch_kl_reduction_stays_finite_for_many_overflow_ratios():
    num_tokens = 1024
    output = default_loss_fn(
        LossInputs(
            trainer_logprobs=torch.full((num_tokens,), -2.0),
            inference_logprobs=torch.full((num_tokens,), -91.0),
            ref_logprobs=None,
            advantages=torch.ones(num_tokens),
            loss_mask=torch.ones(num_tokens, dtype=torch.bool),
        ),
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0),
    )

    _assert_finite_output(output.loss, output.metrics)


def test_mismatch_kl_reduction_stays_finite_for_float16_ratios():
    num_tokens = 1024
    _, ratio, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
        torch.full((num_tokens,), -2.0, dtype=torch.float16),
        torch.full((num_tokens,), -91.0, dtype=torch.float16),
    )

    assert torch.isfinite(ratio.sum())
    assert torch.isfinite(mismatch_kl.sum())


def test_default_loss_clips_unmasked_overflow_ratio():
    inputs = _inputs(-2.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(
        inputs,
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_default_loss_preserves_unclipped_importance_gradient():
    inputs = _inputs(-2.0, -3.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(
        inputs,
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    expected = -math.e
    assert output.loss.item() == pytest.approx(expected)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(expected)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(0.0)


def test_default_loss_masked_overflow_ratio_does_not_nan():
    inputs = _inputs(-1.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = default_loss_fn(inputs, DefaultLossConfig(kl_tau=0.0))
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(0.0)
    assert output.metrics["is_masked"].item() == pytest.approx(1.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ipo_loss_clips_overflow_ratio():
    inputs = _inputs(-2.0, -91.0)
    inputs.trainer_logprobs.requires_grad_()
    output = ipo_loss_fn(
        inputs,
        IPOLossConfig(ipo_threshold=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ref_kl_loss_clips_overflow_ratio():
    inputs = _inputs(-2.0, -91.0, ref_logprob=-3.0)
    inputs.trainer_logprobs.requires_grad_()
    output = ref_kl_loss_fn(inputs)
    output.loss.backward()

    _assert_finite_output(output.loss, output.metrics)
    assert inputs.trainer_logprobs.grad.item() == pytest.approx(20.178)
    assert output.metrics["ref_kl/importance_ratio_clipped"].item() == pytest.approx(1.0)
