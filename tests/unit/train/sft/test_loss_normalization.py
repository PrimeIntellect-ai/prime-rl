import pytest
import torch

from prime_rl.configs.sft import SFTConfig


def test_sample_normalization_allows_liger_fused():
    config = SFTConfig(loss_impl="liger_fused", loss_normalization="sample")
    assert config.loss_impl == "liger_fused"
    assert config.loss_normalization == "sample"


def test_sample_normalization_allows_cp():
    config = SFTConfig(loss_normalization="sample", model={"cp": 2, "name": "dummy"})
    assert config.loss_normalization == "sample"
    assert config.model.cp == 2


def test_token_normalization_is_default():
    config = SFTConfig()
    assert config.loss_normalization == "token"


def _sample_ids_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Reproduce the boundary detection logic from compute_loss."""
    pos = position_ids[0]
    is_boundary = torch.zeros_like(pos, dtype=torch.bool)
    is_boundary[0] = True
    is_boundary[1:] = pos[1:] <= pos[:-1]
    return is_boundary.cumsum(0) - 1


def test_sample_boundary_detection_basic():
    # Two samples: lengths 5 and 3
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
    sample_ids = _sample_ids_from_position_ids(position_ids)
    assert sample_ids.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]


def test_sample_boundary_detection_single_token_samples():
    # Three 1-token samples followed by a 2-token sample
    position_ids = torch.tensor([[0, 0, 0, 0, 1]])
    sample_ids = _sample_ids_from_position_ids(position_ids)
    assert sample_ids.tolist() == [0, 1, 2, 3, 3]


def test_sample_boundary_detection_single_sample():
    position_ids = torch.tensor([[0, 1, 2, 3, 4]])
    sample_ids = _sample_ids_from_position_ids(position_ids)
    assert sample_ids.tolist() == [0, 0, 0, 0, 0]


def _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids):
    """Reproduce cat-packing sample-level loss from compute_loss."""
    tl = token_loss[0]
    mask = loss_mask[0]
    num_samples = sample_ids[-1].item() + 1
    masked_ids = sample_ids[mask]
    masked_losses = tl[mask]
    per_sample_loss = tl.new_zeros(num_samples)
    per_sample_loss.scatter_add_(0, masked_ids, masked_losses)
    per_sample_token_count = tl.new_zeros(num_samples)
    per_sample_token_count.scatter_add_(0, masked_ids, torch.ones_like(masked_losses))
    valid = per_sample_token_count > 0
    per_sample_mean = torch.where(valid, per_sample_loss / per_sample_token_count, per_sample_loss.new_zeros(()))
    return per_sample_mean.sum(), valid.sum(dtype=torch.int64)


def _compute_sample_level_loss_stack(token_loss, loss_mask):
    """Reproduce stack-packing sample-level loss from compute_loss."""
    per_sample_token_count = loss_mask.sum(dim=1).float()
    valid = per_sample_token_count > 0
    per_sample_loss = (token_loss * loss_mask).sum(dim=1)
    per_sample_mean = torch.where(valid, per_sample_loss / per_sample_token_count, per_sample_loss.new_zeros(()))
    return per_sample_mean.sum(), valid.sum(dtype=torch.int64)


def test_sample_level_loss_cat_equal_lengths():
    """Two samples of equal length should give same result as token-level."""
    # Sample 0: tokens [1, 2, 3], Sample 1: tokens [4, 5, 6]
    token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    loss_mask = torch.tensor([[True, True, True, True, True, True]])
    sample_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    loss_sum, count = _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids)
    # Per-sample means: (1+2+3)/3=2.0, (4+5+6)/3=5.0
    # Sum of means: 7.0
    assert loss_sum.item() == pytest.approx(7.0)
    assert count.item() == 2


def test_sample_level_loss_cat_unequal_lengths():
    """Short sample should NOT be dominated by long sample."""
    # Sample 0: 1 token with loss 10.0
    # Sample 1: 4 tokens with loss 1.0 each
    token_loss = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    loss_mask = torch.tensor([[True, True, True, True, True]])
    sample_ids = torch.tensor([0, 1, 1, 1, 1])

    loss_sum, count = _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids)
    # Per-sample means: 10.0/1=10.0, (1+1+1+1)/4=1.0
    # Sum of means: 11.0
    assert loss_sum.item() == pytest.approx(11.0)
    assert count.item() == 2

    # Compare with token-level: (10+1+1+1+1)/5 = 2.8
    # Sample-level: (10.0 + 1.0)/2 = 5.5
    # The short sample contributes much more in sample-level normalization
    token_mean = token_loss.sum().item() / loss_mask.sum().item()
    sample_mean = loss_sum.item() / count.item()
    assert sample_mean > token_mean


def test_sample_level_loss_cat_with_masked_tokens():
    """Masked tokens should not contribute to per-sample means."""
    # Sample 0: 3 tokens, only last 2 are trainable (loss_mask=True)
    # Sample 1: 2 tokens, both trainable
    token_loss = torch.tensor([[99.0, 2.0, 4.0, 3.0, 5.0]])
    loss_mask = torch.tensor([[False, True, True, True, True]])
    sample_ids = torch.tensor([0, 0, 0, 1, 1])

    loss_sum, count = _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids)
    # Sample 0 trainable: [2.0, 4.0] → mean = 3.0
    # Sample 1 trainable: [3.0, 5.0] → mean = 4.0
    # Sum = 7.0
    assert loss_sum.item() == pytest.approx(7.0)
    assert count.item() == 2


def test_sample_level_loss_cat_fully_masked_sample():
    """A sample with all tokens masked should be excluded."""
    token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    loss_mask = torch.tensor([[False, False, True, True, True]])
    sample_ids = torch.tensor([0, 0, 1, 1, 1])

    loss_sum, count = _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids)
    # Sample 0: all masked → excluded
    # Sample 1: [3.0, 4.0, 5.0] → mean = 4.0
    assert loss_sum.item() == pytest.approx(4.0)
    assert count.item() == 1


def test_sample_level_loss_stack_basic():
    """Stack mode: each row is a sample."""
    token_loss = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # Sample 0: 1 trainable token
            [1.0, 1.0, 1.0],  # Sample 1: 3 trainable tokens
        ]
    )
    loss_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
        ]
    )

    loss_sum, count = _compute_sample_level_loss_stack(token_loss, loss_mask)
    # Per-sample means: 10.0/1=10.0, (1+1+1)/3=1.0
    # Sum = 11.0
    assert loss_sum.item() == pytest.approx(11.0)
    assert count.item() == 2


def test_sample_level_loss_stack_with_padding():
    """Padded rows (all masked) should be excluded."""
    token_loss = torch.tensor(
        [
            [2.0, 4.0, 6.0],
            [0.0, 0.0, 0.0],  # Padding row
        ]
    )
    loss_mask = torch.tensor(
        [
            [True, True, True],
            [False, False, False],
        ]
    )

    loss_sum, count = _compute_sample_level_loss_stack(token_loss, loss_mask)
    assert loss_sum.item() == pytest.approx(4.0)  # mean of [2, 4, 6]
    assert count.item() == 1


def test_sample_level_loss_cat_gradients():
    """Verify gradients flow correctly through scatter_add."""
    token_loss = torch.tensor([[1.0, 2.0, 3.0, 6.0]], requires_grad=True)
    loss_mask = torch.tensor([[True, True, True, True]])
    sample_ids = torch.tensor([0, 0, 0, 1])

    loss_sum, count = _compute_sample_level_loss_cat(token_loss, loss_mask, sample_ids)
    # Sample 0: mean = (1+2+3)/3 = 2.0, Sample 1: mean = 6.0/1 = 6.0
    assert loss_sum.item() == pytest.approx(8.0)

    loss_sum.backward()
    # d(loss)/d(token_i) for sample 0: 1/3 (each token contributes 1/count to mean)
    # d(loss)/d(token_i) for sample 1: 1/1
    expected_grad = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 1.0]])
    assert torch.allclose(token_loss.grad, expected_grad)
