import pytest
import torch

from prime_rl.trainer.rl.loss import compute_entropy, grpo_loss_clip, grpo_loss_ratio

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    advantages = torch.randn(10, 10).cuda()

    loss, _ = grpo_loss_clip(
        logprobs,
        old_logprobs,
        advantages,
        epsilon_low=0.2,
        epsilon_high=0.2,
        clip_ratio=10.0,
    )
    assert loss.shape == (10, 10)


def test_grpo_loss_ratio():
    logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    advantages = torch.randn(10, 10).cuda()

    loss, _ = grpo_loss_ratio(
        logprobs,
        old_logprobs,
        advantages,
        clip_ratio=10.0,
    )
    assert loss.shape == (10, 10)


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
