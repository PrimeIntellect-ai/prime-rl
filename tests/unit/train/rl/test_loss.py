import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_loss_uses_expert_logprobs():
    trainer_logprobs = [torch.log(torch.tensor([0.2], device="cuda"))]
    inference_logprobs = [torch.log(torch.tensor([0.1], device="cuda"))]
    trainer_expert_logprobs = [torch.log(torch.tensor([0.5], device="cuda"))]
    inference_expert_logprobs = [torch.log(torch.tensor([0.25], device="cuda"))]
    advantages = [torch.ones(1, device="cuda")]
    loss_mask = [torch.ones(1, dtype=torch.bool, device="cuda")]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        trainer_expert_logprobs=trainer_expert_logprobs,
        inference_expert_logprobs=inference_expert_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", token_mask_high=10.0, kl_tau=0.0),
        loss_scale=1.0,
    )

    expected_logprob = torch.log(torch.tensor(0.1, device="cuda"))
    expected = -(torch.tensor(4.0, device="cuda") * expected_logprob)
    assert torch.allclose(loss, expected)


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
