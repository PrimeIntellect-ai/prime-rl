import pytest
import torch

from prime_rl.trainer.loss import compute_entropy, grpo_loss_clip, grpo_loss_ratio

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()

    loss, _ = grpo_loss_clip(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask,
        epsilon_low=0.2,
        epsilon_high=0.2,
        clip_ratio=10.0,
    )
    assert loss.shape == ()
    assert loss.item() is not None


def test_grpo_loss_ratio():
    logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()

    loss, _ = grpo_loss_ratio(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask,
        clip_ratio=10.0,
    )
    assert loss.shape == ()
    assert loss.item() is not None


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_grpo_loss_padding():
    logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(10, 10, dtype=torch.float32).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    rewards = torch.ones(10, 10).cuda()

    loss_list = []
    reward_list = []
    for padding in [2, 5]:
        pad_logprobs = torch.cat([logprobs, torch.zeros(10, padding, dtype=torch.float32).cuda()], dim=1)
        pad_old_logprobs = torch.cat([old_logprobs, torch.zeros(10, padding, dtype=torch.float32).cuda()], dim=1)
        pad_advantages = torch.cat([advantages, torch.zeros(10, padding, dtype=torch.float32).cuda()], dim=1)
        pad_loss_mask = torch.cat([loss_mask, torch.zeros(10, padding, dtype=torch.int).cuda()], dim=1)
        pad_rewards = torch.cat([rewards, torch.zeros(10, padding, dtype=torch.float32).cuda()], dim=1)

        r = pad_rewards[pad_loss_mask.bool()]
        sum_rewards = r.sum()
        token_count = r.numel()

        reward = sum_rewards / token_count
        reward_list.append(reward)

        loss, _ = grpo_loss_clip(
            pad_logprobs,
            pad_old_logprobs,
            pad_advantages,
            pad_loss_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
            clip_ratio=10.0,
        )
        loss_list.append(loss)

    assert torch.allclose(reward_list[0], reward_list[1])
    assert torch.allclose(loss_list[0], loss_list[1])
