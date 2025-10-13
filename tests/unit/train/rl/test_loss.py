import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    old_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", clip_ratio=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    old_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", clip_ratio=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
