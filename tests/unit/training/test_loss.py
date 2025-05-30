import pytest
import torch

from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss(dtype):
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()

    loss = grpo_loss(
        advantages,
        loss_mask,
        100,
    )
    assert loss.shape == ()
    assert loss.item() is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_entropy_loss(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    entropy = entropy_loss(logits, loss_mask, temperature=0.6, max_tokens=100)
    assert entropy.shape == ()
    assert entropy.item() is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss_padding(dtype):
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    rewards = torch.ones(10, 10).cuda()

    loss_list = []
    reward_list = []
    for padding in [2, 5]:
        pad_advantages = torch.cat([advantages, torch.zeros(10, padding, dtype=dtype).cuda()], dim=1)
        pad_loss_mask = torch.cat([loss_mask, torch.zeros(10, padding, dtype=torch.int).cuda()], dim=1)
        pad_rewards = torch.cat([rewards, torch.zeros(10, padding, dtype=dtype).cuda()], dim=1)

        r = pad_rewards[pad_loss_mask.bool()]
        sum_rewards = r.sum()
        token_count = r.numel()

        reward = sum_rewards / token_count
        reward_list.append(reward)

        loss = grpo_loss(
            pad_advantages,
            pad_loss_mask,
            max_tokens=100,
        )
        loss_list.append(loss)

    assert torch.allclose(reward_list[0], reward_list[1])
    assert torch.allclose(loss_list[0], loss_list[1])


def test_kl_penalty():
    logprob = torch.randn(10, 9, dtype=torch.float32).cuda()
    ref_logprob = torch.randn(10, 9, dtype=torch.float32).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    kl = kl_penalty(logprob, ref_logprob, loss_mask, 100)
    assert kl.shape == ()
    assert kl.item() is not None
