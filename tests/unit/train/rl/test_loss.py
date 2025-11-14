import math

import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [
        torch.randn(50, dtype=torch.float32).cuda(),
        torch.randn(30, dtype=torch.float32).cuda(),
    ]
    inference_logprobs = [
        torch.randn(50, dtype=torch.float32).cuda(),
        torch.randn(30, dtype=torch.float32).cuda(),
    ]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", mask_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    trainer_logprobs = [
        torch.randn(40, dtype=torch.float32).cuda(),
        torch.randn(60, dtype=torch.float32).cuda(),
    ]
    inference_logprobs = [
        torch.randn(40, dtype=torch.float32).cuda(),
        torch.randn(60, dtype=torch.float32).cuda(),
    ]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", mask_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_tis_truncates_token_ratios():
    trainer_logprobs = [torch.tensor([math.log(16.0)], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(1, dtype=torch.float32).cuda()]
    advantages = [torch.ones(1, dtype=torch.float32).cuda()]
    loss_mask = [torch.ones(1, dtype=torch.bool).cuda()]

    loss, metrics = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", tis_clip=2.0, mask_ratio_high=100.0),
        loss_scale=1.0,
    )

    assert torch.isclose(loss, torch.tensor(-2.0, device=loss.device)).item()
    assert "tis_trunc_ratio_mean" in metrics
    assert torch.isclose(metrics["tis_trunc_ratio_mean"].squeeze(), torch.tensor(2.0, device=loss.device)).item()
    assert metrics["tis_clipped_fraction"].squeeze().item() == pytest.approx(1.0, rel=0, abs=1e-6)


def test_tis_sequence_ratio_broadcasts():
    trainer_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.full((3,), -math.log(4.0), dtype=torch.float32).cuda()]
    advantages = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda()]
    loss_mask = [torch.ones(3, dtype=torch.bool).cuda()]

    loss, metrics = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", tis_clip=2.0, mask_ratio_high=100.0),
        loss_scale=1.0,
    )

    assert torch.isclose(loss, torch.tensor(-12.0, device=loss.device)).item()
    assert metrics["tis_clipped_fraction"].squeeze().item() == pytest.approx(1.0, rel=0, abs=1e-6)
    assert metrics["tis_trunc_ratio_mean"].squeeze().item() == pytest.approx(2.0, rel=1e-5)
