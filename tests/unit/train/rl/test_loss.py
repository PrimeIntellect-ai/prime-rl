import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig
from prime_rl.trainer.rl.loss import (
    LossInputs,
    LossOutputs,
    apply_top_k_mask,
    compute_entropy,
    compute_loss,
    selective_log_softmax,
    setup_loss_fn,
)

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fn = setup_loss_fn(DefaultLossConfig(ratio_type="token", token_mask_high=10.0))
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

    loss_fn = setup_loss_fn(DefaultLossConfig(ratio_type="sequence", token_mask_high=10.0))
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


def test_apply_top_k_mask():
    """Test that apply_top_k_mask keeps only top-k logits and the target token."""
    batch, seq, vocab = 2, 4, 100
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32).cuda()
    labels = torch.randint(0, vocab, (batch, seq)).cuda()
    top_k = 10

    masked = apply_top_k_mask(logits, top_k, labels)

    # Kept values retain their original value, masked values are -1e20
    kept_mask = masked > -1e20
    # At least top_k tokens should be kept per position (plus possibly the target)
    assert (kept_mask.sum(dim=-1) >= top_k).all()
    # At most top_k + 1 tokens (top_k + target if target wasn't in top_k)
    assert (kept_mask.sum(dim=-1) <= top_k + 1).all()

    # The target token is always kept (never masked)
    for b in range(batch):
        for s in range(seq):
            assert masked[b, s, labels[b, s]] > -1e20


def test_top_k_mask_logprobs_concentrate_probability():
    """Test that top-k masking concentrates probability on fewer tokens."""
    batch, seq, vocab = 1, 1, 1000
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32).cuda()
    labels = torch.zeros(batch, seq, dtype=torch.long).cuda()

    full_logprobs = selective_log_softmax(logits, labels)
    masked_logits = apply_top_k_mask(logits, top_k=50, keep_indices=labels)
    masked_logprobs = selective_log_softmax(masked_logits, labels)

    # With fewer tokens sharing the probability mass, the selected token's logprob
    # should be >= what it was under the full distribution
    assert (masked_logprobs >= full_logprobs - 1e-5).all()


def test_top_k_mask_entropy_decreases():
    """Test that top-k masking reduces entropy (fewer possible tokens)."""
    batch, seq, vocab = 1, 2, 500
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32).cuda()
    labels = torch.zeros(batch, seq, dtype=torch.long).cuda()

    full_entropy = compute_entropy(logits)
    masked_logits = apply_top_k_mask(logits, top_k=20, keep_indices=labels)
    masked_entropy = compute_entropy(masked_logits)

    assert (masked_entropy <= full_entropy + 1e-5).all()


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
