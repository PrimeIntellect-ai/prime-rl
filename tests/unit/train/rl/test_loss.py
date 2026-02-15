import pytest
import torch

from prime_rl.trainer.rl.config import CustomLossConfig, LossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    compute_entropy,
    compute_importance_weights,
    compute_loss,
    reject_by_geo_mean,
    reject_by_geo_mean_k3,
    reject_by_sequence_minmax,
    reject_by_sequence_sum,
    reject_by_token,
    setup_loss_fn,
)

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(LossConfig(ratio_type="token", token_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(LossConfig(ratio_type="sequence", token_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_loss_fn_with_custom_config():
    """Test setup_loss_fn with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    loss_fn = setup_loss_fn(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = loss_fn(inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )


def test_importance_weights_mask_affects_sequence_and_geo_but_not_token():
    log_ratio = torch.tensor([1.0, 20.0, 1.0], device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")

    w = compute_importance_weights(log_ratio, mask)

    torch.testing.assert_close(w.token_weights, torch.exp(log_ratio))
    torch.testing.assert_close(w.sequence_weight, torch.exp(torch.tensor(2.0, device="cuda")))
    torch.testing.assert_close(w.geo_mean_weight, torch.exp(torch.tensor(1.0, device="cuda")))


def test_importance_weights_empty_mask_degrades_to_unit():
    log_ratio = torch.tensor([1.0, 2.0], device="cuda")
    mask = torch.zeros(2, dtype=torch.bool, device="cuda")

    w = compute_importance_weights(log_ratio, mask)

    assert w.geo_mean_weight.item() == pytest.approx(1.0)
    assert w.sequence_weight.item() == pytest.approx(1.0)


def test_reject_by_token():
    log_ratio = torch.tensor([0.0, 3.0, -3.0], device="cuda")  # ratios: 1.0, ~20.1, ~0.05
    low_mask, high_mask = reject_by_token(log_ratio, low=0.1, high=5.0)

    assert low_mask.tolist() == [False, False, True]
    assert high_mask.tolist() == [False, True, False]


def test_reject_by_sequence_minmax_one_bad_token_rejects_sequence():
    log_ratio = torch.tensor([0.0, 5.0, 0.0], device="cuda")
    mask = torch.ones(3, dtype=torch.bool, device="cuda")
    _, high_mask = reject_by_sequence_minmax(log_ratio, mask, high=100.0)

    assert high_mask.item() is True


def test_reject_by_sequence_minmax_ignores_unmasked():
    log_ratio = torch.tensor([0.0, 10.0, 0.0], device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    _, high_mask = reject_by_sequence_minmax(log_ratio, mask, high=5.0)

    assert high_mask.item() is False


def test_reject_by_sequence_sum():
    # product ratio = exp(1.5) ≈ 4.48
    log_ratio = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    mask = torch.ones(3, dtype=torch.bool, device="cuda")
    
    low, high = reject_by_sequence_sum(log_ratio, mask, low=0.1, high=10.0)
    assert low.item() is False
    assert high.item() is False

    _, high = reject_by_sequence_sum(log_ratio, mask, high=4.0)
    assert high.item() is True

    # negative product ratio = exp(-1.5) ≈ 0.22
    low, _ = reject_by_sequence_sum(-log_ratio, mask, low=0.5)
    assert low.item() is True
    
    # big log_ratio at index 1, but it's unmasked → product only over indices 0, 2
    log_ratio = torch.tensor([0.1, 10.0, 0.1], device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")

    _, high = reject_by_sequence_sum(log_ratio, mask, high=5.0)
    assert high.item() is False

     # None bounds
    low, high = reject_by_sequence_sum(log_ratio, mask)
    assert low.item() is False
    assert high.item() is False

def test_k1_cancellation_hides_mismatch_k3_catches_it():
    """k1 geo mean allows cancellation (e^2 and e^-2 average to 1); k3 KL does not."""
    log_ratio = torch.tensor([2.0, -2.0], device="cuda")
    mask = torch.ones(2, dtype=torch.bool, device="cuda")

    k1_low, k1_high = reject_by_geo_mean(log_ratio, mask, low=0.5, high=2.0)
    assert k1_low.item() is False
    assert k1_high.item() is False

    k3_rejected = reject_by_geo_mean_k3(log_ratio, mask, high=0.5)
    assert k3_rejected.item() is True
