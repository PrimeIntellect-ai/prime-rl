import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import (
    chunked_selective_log_softmax,
    compute_entropy,
    compute_loss,
    selective_log_softmax,
)

pytestmark = [pytest.mark.gpu]


def test_chunked_selective_log_softmax():
    """Test that chunked version produces same results as standard version."""
    torch.manual_seed(42)
    batch, seq, vocab = 2, 2048, 32000
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32).cuda()
    index = torch.randint(0, vocab, (batch, seq)).cuda()

    # Standard implementation
    expected = selective_log_softmax(logits, index)

    # Chunked implementations with different chunk sizes
    for chunk_size in [128, 256, 512, 1024]:
        result = chunked_selective_log_softmax(logits, index, chunk_size=chunk_size)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_chunked_selective_log_softmax_short_seq():
    """Test chunked version falls back correctly for short sequences."""
    torch.manual_seed(42)
    batch, seq, vocab = 2, 512, 32000
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32).cuda()
    index = torch.randint(0, vocab, (batch, seq)).cuda()

    expected = selective_log_softmax(logits, index)
    result = chunked_selective_log_softmax(logits, index, chunk_size=1024)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="token", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    # Create list of tensors as expected by compute_loss (simulating split sequences)
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_config=LossConfig(ratio_type="sequence", token_mask_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
