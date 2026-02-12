import pytest
import torch

from prime_rl.trainer.rl.config import CustomLossConfig, LossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    compute_entropy,
    compute_importance_weights,
    compute_loss,
    reject_by_geo_k1,
    reject_by_geo_k3,
    reject_by_sequence_max,
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


class TestComputeImportanceWeights:
    def test_basic_computation(self):
        log_ratio = torch.tensor([0.0, 1.0, -1.0], device="cuda")
        mask = torch.ones(3, dtype=torch.bool, device="cuda")

        w = compute_importance_weights(log_ratio, mask)

        torch.testing.assert_close(w.token_weights, torch.exp(log_ratio))
        expected_seq = torch.exp(torch.tensor(0.0, device="cuda"))  # exp(0+1+(-1)) = exp(0)
        torch.testing.assert_close(w.sequence_weight, expected_seq)
        expected_geo = torch.exp(log_ratio.mean())
        torch.testing.assert_close(w.geo_mean_weight, expected_geo)

    def test_large_log_ratio(self):
        log_ratio = torch.tensor([5.0, 5.0, 5.0], device="cuda")
        mask = torch.ones(3, dtype=torch.bool, device="cuda")

        w = compute_importance_weights(log_ratio, mask)

        torch.testing.assert_close(w.token_weights, torch.exp(log_ratio))
        torch.testing.assert_close(w.sequence_weight, torch.exp(torch.tensor(15.0, device="cuda")))
        torch.testing.assert_close(w.geo_mean_weight, torch.exp(torch.tensor(5.0, device="cuda")))

    def test_mask_excludes_tokens(self):
        log_ratio = torch.tensor([1.0, 20.0, 1.0], device="cuda")
        mask = torch.tensor([True, False, True], device="cuda")

        w = compute_importance_weights(log_ratio, mask)

        torch.testing.assert_close(w.token_weights, torch.exp(log_ratio))
        expected_seq = torch.exp(torch.tensor(2.0, device="cuda"))  # exp(1+1)
        torch.testing.assert_close(w.sequence_weight, expected_seq)
        expected_geo = torch.exp(torch.tensor(1.0, device="cuda"))  # exp(mean([1,1]))
        torch.testing.assert_close(w.geo_mean_weight, expected_geo)

    def test_all_false_mask(self):
        log_ratio = torch.tensor([1.0, 2.0], device="cuda")
        mask = torch.zeros(2, dtype=torch.bool, device="cuda")

        w = compute_importance_weights(log_ratio, mask)

        assert w.geo_mean_weight.item() == pytest.approx(1.0)
        assert w.sequence_weight.item() == pytest.approx(1.0)


class TestRejectByToken:
    def test_rejects_above_high(self):
        log_ratio = torch.tensor([0.0, 3.0, -1.0], device="cuda")  # weights: 1.0, ~20.1, ~0.37
        low_mask, high_mask = reject_by_token(log_ratio, high=5.0)

        assert high_mask.tolist() == [False, True, False]
        assert low_mask.tolist() == [False, False, False]

    def test_rejects_below_low(self):
        log_ratio = torch.tensor([0.0, 3.0, -3.0], device="cuda")  # weights: 1.0, ~20.1, ~0.05
        low_mask, high_mask = reject_by_token(log_ratio, low=0.1)

        assert low_mask.tolist() == [False, False, True]
        assert high_mask.tolist() == [False, False, False]

    def test_no_bounds_rejects_nothing(self):
        log_ratio = torch.tensor([10.0, -10.0], device="cuda")
        low_mask, high_mask = reject_by_token(log_ratio)

        assert not low_mask.any()
        assert not high_mask.any()


class TestRejectBySequenceMax:
    def test_rejects_sequence_when_any_token_breaches(self):
        log_ratio = torch.tensor([0.0, 5.0, 0.0], device="cuda")  # one token at w ≈ 148
        mask = torch.ones(3, dtype=torch.bool, device="cuda")
        _, high_mask = reject_by_sequence_max(log_ratio, mask, high=100.0)

        assert high_mask.item() is True

    def test_no_rejection_when_within_bounds(self):
        log_ratio = torch.tensor([0.1, -0.1, 0.2], device="cuda")
        mask = torch.ones(3, dtype=torch.bool, device="cuda")
        low_mask, high_mask = reject_by_sequence_max(log_ratio, mask, low=0.5, high=2.0)

        assert low_mask.item() is False
        assert high_mask.item() is False

    def test_ignores_unmasked_tokens(self):
        log_ratio = torch.tensor([0.0, 10.0, 0.0], device="cuda")  # middle token huge but unmasked
        mask = torch.tensor([True, False, True], device="cuda")
        _, high_mask = reject_by_sequence_max(log_ratio, mask, high=5.0)

        assert high_mask.item() is False


class TestRejectByGeoK1:
    def test_rejects_above_high(self):
        log_ratio = torch.tensor([2.0, 2.0, 2.0], device="cuda")  # geo mean = exp(2) ≈ 7.39
        mask = torch.ones(3, dtype=torch.bool, device="cuda")
        _, high_mask = reject_by_geo_k1(log_ratio, mask, high=5.0)

        assert high_mask.item() is True

    def test_rejects_below_low(self):
        log_ratio = torch.tensor([-3.0, -3.0], device="cuda")  # geo mean = exp(-3) ≈ 0.05
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        low_mask, _ = reject_by_geo_k1(log_ratio, mask, low=0.1)

        assert low_mask.item() is True

    def test_cancellation_hides_mismatch(self):
        """k1 allows cancellation: half tokens at w=e^2, half at w=e^-2 → geo mean = 1.0"""
        log_ratio = torch.tensor([2.0, -2.0], device="cuda")
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        low_mask, high_mask = reject_by_geo_k1(log_ratio, mask, low=0.5, high=2.0)

        assert low_mask.item() is False
        assert high_mask.item() is False


class TestRejectByGeoK3:
    def test_rejects_high_divergence(self):
        log_ratio = torch.tensor([2.0, -2.0], device="cuda")
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        rejected = reject_by_geo_k3(log_ratio, mask, high=0.5)

        assert rejected.item() is True

    def test_no_rejection_when_close(self):
        log_ratio = torch.tensor([0.01, -0.01], device="cuda")
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        rejected = reject_by_geo_k3(log_ratio, mask, high=1.0)

        assert rejected.item() is False

    def test_none_high_returns_false(self):
        log_ratio = torch.tensor([5.0, 5.0], device="cuda")
        mask = torch.ones(2, dtype=torch.bool, device="cuda")
        rejected = reject_by_geo_k3(log_ratio, mask, high=None)

        assert rejected.item() is False
