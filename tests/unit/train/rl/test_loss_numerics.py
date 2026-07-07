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


def test_default_loss_clips_unmasked_overflow_ratio():
    output = default_loss_fn(
        _inputs(-2.0, -91.0),
        DefaultLossConfig(dppo_mask_high=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_default_loss_masked_overflow_ratio_does_not_nan():
    output = default_loss_fn(_inputs(-1.0, -91.0), DefaultLossConfig())

    _assert_finite_output(output.loss, output.metrics)
    assert output.metrics["is_masked"].item() == pytest.approx(1.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ipo_loss_clips_overflow_ratio():
    output = ipo_loss_fn(
        _inputs(-2.0, -91.0),
        IPOLossConfig(ipo_threshold=10.0, kl_tau=0.0, importance_ratio_max=20.0),
    )

    _assert_finite_output(output.loss, output.metrics)
    assert output.loss.item() == pytest.approx(-20.0)
    assert output.metrics["importance_ratio_clipped"].item() == pytest.approx(1.0)


def test_ref_kl_loss_clips_overflow_ratio():
    output = ref_kl_loss_fn(_inputs(-2.0, -91.0, ref_logprob=-3.0))

    _assert_finite_output(output.loss, output.metrics)
    assert output.metrics["ref_kl/importance_ratio_clipped"].item() == pytest.approx(1.0)
