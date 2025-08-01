import pytest
import torch

from prime_rl.trainer.loss import compute_entropy, grpo_loss_clip, grpo_loss_ratio

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    original_logprobs = torch.randn(10, 10, dtype=dtype).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    input_ids = torch.randint(0, 10, (10, 10)).cuda()

    loss, ratio_info = grpo_loss_clip(
        logits,
        input_ids,
        advantages,
        original_logprobs,
        loss_mask,
        temperature=0.6,
        epsilon_low=0.2,
        epsilon_high=0.2,
        clip_ratio=10.0,
    )
    assert loss.shape == ()
    assert loss.item() is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss_ratio(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    original_logprobs = torch.randn(10, 10, dtype=dtype).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    input_ids = torch.randint(0, 10, (10, 10)).cuda()

    loss, ratio_info = grpo_loss_ratio(
        logits,
        input_ids,
        advantages,
        original_logprobs,
        loss_mask,
        temperature=0.6,
        clip_ratio=10.0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_entropy_loss(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    entropy = compute_entropy(logits, loss_mask, temperature=0.6)
    assert entropy.shape == ()
    assert entropy.item() is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_loss_padding(dtype):
    logits = torch.randn(10, 10, 10, dtype=dtype).cuda()
    original_logprobs = torch.randn(10, 10, dtype=dtype).cuda()
    advantages = torch.randn(10, 10).cuda()
    loss_mask = torch.ones(10, 10).int().cuda()
    input_ids = torch.randint(0, 10, (10, 10)).cuda()
    rewards = torch.ones(10, 10).cuda()

    loss_list = []
    reward_list = []
    for padding in [2, 5]:
        pad_logits = torch.cat([logits, torch.zeros(10, padding, 10, dtype=dtype).cuda()], dim=1)
        pad_original_logprobs = torch.cat([original_logprobs, torch.zeros(10, padding, dtype=dtype).cuda()], dim=1)
        pad_advantages = torch.cat([advantages, torch.zeros(10, padding, dtype=dtype).cuda()], dim=1)
        pad_loss_mask = torch.cat([loss_mask, torch.zeros(10, padding, dtype=torch.int).cuda()], dim=1)
        pad_input_ids = torch.cat([input_ids, torch.zeros(10, padding, dtype=torch.int).cuda()], dim=1)
        pad_rewards = torch.cat([rewards, torch.zeros(10, padding, dtype=dtype).cuda()], dim=1)

        r = pad_rewards[pad_loss_mask.bool()]
        sum_rewards = r.sum()
        token_count = r.numel()

        reward = sum_rewards / token_count
        reward_list.append(reward)

        loss, ratio_info = grpo_loss_clip(
            pad_logits,
            pad_input_ids,
            pad_advantages,
            pad_original_logprobs,
            pad_loss_mask,
            temperature=0.6,
            epsilon_low=0.2,
            epsilon_high=0.2,
            clip_ratio=10.0,
        )
        loss_list.append(loss)

    assert torch.allclose(reward_list[0], reward_list[1])
    assert torch.allclose(loss_list[0], loss_list[1])
