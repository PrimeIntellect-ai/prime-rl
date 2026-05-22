import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, SFTLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, setup_loss_fn

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
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

    loss_fn = setup_loss_fn(DefaultLossConfig(dppo_mask_high=10.0))
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


def test_sft_loss_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(SFTLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=2,
    )

    # loss = -sum(masked logprobs) / loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics


def test_sft_loss_override_uses_masked_nll_with_default_loss_config():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.ones(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fn=loss_fn,
        loss_scale=2,
        sft_loss=True,
    )

    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_default_loss_fn_sft_mask_gradient_flows():
    """Gradient on SFT positions flows via ``trainer_logprobs``.

    Regression test: previously ``pg_loss = advantages * importance_ratio``
    routed SFT positions through ``importance_ratio`` which was forced to 1
    via ``torch.where(sft_mask, zeros_like(log_ratio), log_ratio)`` — that
    construction blocks the gradient through ``trainer_logprobs`` on the
    True branch, silently zeroing the SFT signal. Fix routes SFT positions
    through ``advantages * trainer_logprobs`` directly.

    On SFT positions: ``pg_loss = adv * trainer_logprob`` and
    ``loss = -pg_loss.sum()`` so ``d(loss)/d(trainer_logprob[s]) = -adv[s]``.
    KL is zero on SFT positions (log_ratio forced to 0), so no KL gradient
    contribution.
    """
    seq_len = 5
    sft_alpha = 0.5
    n_sft = 2
    per_token_sft_weight = sft_alpha / n_sft  # sft_tokens mode formula

    # Create requires_grad tensor directly on cuda so it's a leaf — .cuda()
    # after requires_grad=True makes the device tensor a non-leaf and .grad
    # would never populate.
    trainer_logprobs = torch.tensor(
        [-0.1, -0.2, -0.3, -0.4, -0.5], dtype=torch.float32, device="cuda", requires_grad=True
    )
    inference_logprobs = torch.tensor([-0.05, 0.0, -0.4, 0.0, -0.6], dtype=torch.float32, device="cuda")
    advantages = torch.tensor(
        [1.0, per_token_sft_weight, 1.0, per_token_sft_weight, 1.0], dtype=torch.float32, device="cuda"
    )
    loss_mask = torch.ones(seq_len, dtype=torch.bool, device="cuda")
    sft_mask = torch.tensor([False, True, False, True, False], dtype=torch.bool, device="cuda")

    loss_fn = setup_loss_fn(DefaultLossConfig(dppo_mask_high=10.0, dppo_mask_low=10.0))
    loss, _ = compute_loss(
        trainer_logprobs=[trainer_logprobs],
        inference_logprobs=[inference_logprobs],
        teacher_logprobs=None,
        advantages=[advantages],
        loss_mask=[loss_mask],
        loss_fn=loss_fn,
        loss_scale=1,
        sft_mask=[sft_mask],
    )
    loss.backward()
    grad = trainer_logprobs.grad
    assert grad is not None

    # Regression check: gradient on SFT positions must be non-zero.
    assert grad[1].abs() > 1e-6, f"SFT position 1: gradient was zeroed ({grad[1].item()})"
    assert grad[3].abs() > 1e-6, f"SFT position 3: gradient was zeroed ({grad[3].item()})"

    # Exact magnitude: -advantage on SFT positions.
    expected = torch.tensor(-per_token_sft_weight, device=grad.device)
    assert torch.isclose(grad[1], expected, atol=1e-6)
    assert torch.isclose(grad[3], expected, atol=1e-6)


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
