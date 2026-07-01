import pytest
import torch
import torch.nn as nn

from prime_rl.utils.sparse_update import (
    apply_sparse_update,
    apply_sparse_update_to_params,
    save_sparse_update_from_diff,
)


def test_save_sparse_update_from_diff_basic(tmp_path):
    """Test that save_sparse_update_from_diff produces correct patches."""
    weights = {
        "layer.0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
        "layer.0.bias": torch.tensor([0.1, 0.2], dtype=torch.bfloat16),
    }
    diffs = {
        "layer.0.weight": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        "layer.0.bias": torch.tensor([False, True], dtype=torch.bool),
    }

    stats = save_sparse_update_from_diff(weights, diffs, tmp_path, step=1, base_step=0)

    assert stats.changed_numel == 3
    assert stats.patched_tensors == 2

    previous = {
        "layer.0.weight": torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.bfloat16),
        "layer.0.bias": torch.tensor([0.1, 0.0], dtype=torch.bfloat16),
    }
    apply_sparse_update(previous, tmp_path, expected_base_step=0)

    assert torch.equal(previous["layer.0.weight"], weights["layer.0.weight"])
    assert torch.equal(previous["layer.0.bias"], weights["layer.0.bias"])


def test_save_sparse_update_from_diff_no_changes(tmp_path):
    """Test that no-change diffs produce empty patches."""
    weights = {"weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)}
    diffs = {"weight": torch.tensor([False, False, False], dtype=torch.bool)}

    stats = save_sparse_update_from_diff(weights, diffs, tmp_path, step=1, base_step=0)

    assert stats.changed_numel == 0
    assert stats.sparsity == 1.0
    assert stats.patched_tensors == 0


def test_save_sparse_update_from_diff_missing_diff(tmp_path):
    """Test that a missing diff marks all elements as changed."""
    weights = {"weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)}
    diffs = {}

    stats = save_sparse_update_from_diff(weights, diffs, tmp_path, step=1, base_step=0)

    assert stats.changed_numel == 3
    assert stats.patched_tensors == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_save_sparse_update_from_diff_on_gpu(tmp_path):
    """Test that save_sparse_update_from_diff works on GPU tensors."""
    # "Previous" weights (before optimizer step)
    previous = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=torch.bfloat16).cuda()
    # "Current" weights (after optimizer step)
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16).cuda()
    # Diff: which elements changed
    diffs = {"weight": current.ne(previous)}

    stats = save_sparse_update_from_diff({"weight": current}, diffs, tmp_path, step=1, base_step=0, compute_dtype=None)

    assert stats.changed_numel == 2

    # Create a model initialized with the "previous" weights
    model = nn.Linear(2, 2, bias=False).cuda().to(torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(previous)

    apply_sparse_update_to_params(model, tmp_path, expected_base_step=0, device="cuda")

    result = dict(model.named_parameters())["weight"].data
    assert torch.equal(result, current)
