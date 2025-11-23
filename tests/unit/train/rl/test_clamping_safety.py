import pytest
import torch
from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_loss

pytestmark = [pytest.mark.gpu]

class TestClampingSafety:
    \"\"\"Tests for clamping safety and metric.\"\"\"
    def test_extreme_ratios_no_nan(self):
        trainer_logprobs = [torch.tensor([700.0, -700.0]).cuda()]
        inference_logprobs = [torch.zeros(2).cuda()]
        advantages = [torch.ones(2).cuda()]
        loss_mask = [torch.ones(2, dtype=torch.bool).cuda()]
        config = LossConfig(ratio_type=\"token\", mask_ratio_high=10.0)
        loss, metrics = compute_loss(trainer_logprobs, inference_logprobs, advantages, loss_mask, config, 1.0)
        assert torch.isfinite(loss)
        assert \"clamped_ratio\" in metrics
        assert metrics[\"clamped_ratio\"].mean() > 0

    def test_clamped_ratio_normals_zero(self):
        trainer_logprobs = [torch.randn(100).cuda()]
        inference_logprobs = [torch.randn(100).cuda()]
        advantages = [torch.randn(100).cuda()]
        loss_mask = [torch.ones(100, dtype=torch.bool).cuda()]
        config = LossConfig(ratio_type=\"token\")
        _, metrics = compute_loss(trainer_logprobs, inference_logprobs, advantages, loss_mask, config, 1.0)
        assert torch.allclose(metrics[\"clamped_ratio\"], torch.tensor(0.0))

    def test_grad_flow_clamped(self):
        trainer_logprobs = [torch.tensor([25.0, -25.0], requires_grad=True).cuda()]
        inference_logprobs = [torch.zeros(2).cuda()]
        advantages = [torch.ones(2).cuda()]
        loss_mask = [torch.ones(2, dtype=torch.bool).cuda()]
        config = LossConfig(ratio_type=\"token\")
        loss, _ = compute_loss(trainer_logprobs, inference_logprobs, advantages, loss_mask, config, 1.0)
        loss.backward()
        assert trainer_logprobs[0].grad is not None
        assert torch.all(torch.isfinite(trainer_logprobs[0].grad))
