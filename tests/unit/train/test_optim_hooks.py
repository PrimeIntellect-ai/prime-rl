import pytest
import torch
import torch.nn as nn

from prime_rl.trainer.optim_hooks import (
    clear_sparse_diffs,
    ensure_diffs_on_device,
    get_sparse_diffs,
    setup_sparse_diff_hook,
)


def test_sparse_diff_hook_captures_changes():
    """Test that the optimizer hook captures boolean diffs after a step."""
    model = nn.Linear(4, 3, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    setup_sparse_diff_hook(optimizer, model)

    # Before any step: no diffs
    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 0

    # Run a forward/backward/step
    x = torch.randn(2, 4)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # After step: should have diffs for weight and bias
    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 2  # weight + bias

    # Check diffs are boolean
    for diff in diffs.values():
        assert diff.dtype == torch.bool

    # With a meaningful lr and random data, most values should change
    weight_diff = diffs[model.weight]
    assert weight_diff.any()

    # Clean up
    clear_sparse_diffs(optimizer)
    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 0


def test_sparse_diff_hook_no_change_when_grad_zero():
    """Test that zero gradients produce no diff (weight decay is small enough)."""
    model = nn.Linear(4, 3, bias=False)
    # Use lr=0 so no update happens
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    setup_sparse_diff_hook(optimizer, model)

    # Dummy forward/backward with zero gradient
    x = torch.zeros(2, 4)
    y = model(x)
    loss = y.sum() * 0  # zero loss
    loss.backward()
    optimizer.step()

    diffs = get_sparse_diffs(optimizer)
    weight_diff = diffs[model.weight]
    assert not weight_diff.any()  # nothing changed


def test_sparse_diff_hook_with_cpu_offload():
    """Test that hooks work through CPUOffloadOptimizer wrapper."""
    from prime_rl.trainer.optim import CPUOffloadOptimizer

    model = nn.Linear(4, 3, bias=True)
    inner_opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    optimizer = CPUOffloadOptimizer(inner_opt)
    setup_sparse_diff_hook(optimizer, model)

    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # After step + offload, diffs should be in state (on CPU)
    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 2  # weight + bias

    # Move to GPU and back
    ensure_diffs_on_device(optimizer, "cuda" if torch.cuda.is_available() else "cpu")

    clear_sparse_diffs(optimizer)
    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 0


def test_sparse_diff_hook_tied_weights():
    """Test that tied weights produce diffs under both names."""
    model = nn.Linear(4, 3, bias=False)
    # Tie the weight to another parameter
    model.tied_weight = model.weight  # same param object

    optimizer = torch.optim.AdamW([model.weight], lr=0.1)
    setup_sparse_diff_hook(optimizer, model)

    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    diffs = get_sparse_diffs(optimizer)
    assert len(diffs) == 1  # only one param object
    assert model.weight in diffs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_sparse_diff_hook_ensure_diffs_on_device():
    """Test moving diffs to GPU works."""
    model = nn.Linear(4, 3, bias=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    setup_sparse_diff_hook(optimizer, model)

    x = torch.randn(2, 4).cuda()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    diffs = get_sparse_diffs(optimizer)
    # Diffs should be on GPU (params are on GPU)
    for diff in diffs.values():
        assert diff.device.type == "cuda"

    # Move to CPU and back
    ensure_diffs_on_device(optimizer, "cpu")
    diffs = get_sparse_diffs(optimizer)
    for diff in diffs.values():
        assert diff.device.type == "cpu"

    ensure_diffs_on_device(optimizer, "cuda")
    diffs = get_sparse_diffs(optimizer)
    for diff in diffs.values():
        assert diff.device.type == "cuda"

    clear_sparse_diffs(optimizer)
