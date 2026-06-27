import pytest
import torch
import torch.nn as nn

from prime_rl.utils.sparse_update import (
    apply_sparse_update,
    apply_sparse_update_to_params,
    save_sparse_update,
    to_compute_tensor,
)


def test_sparse_update_reconstructs_bf16_view(tmp_path):
    previous = {
        "model.layers.0.weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "model.layers.0.bias": torch.tensor([0.25, 0.5], dtype=torch.float32),
    }
    current = {
        # The first update is BF16-invisible; the second is visible.
        "model.layers.0.weight": torch.tensor([1.0001, 2.5, 3.0], dtype=torch.float32),
        "model.layers.0.bias": torch.tensor([0.25, 0.75], dtype=torch.float32),
    }
    receiver = {name: to_compute_tensor(tensor) for name, tensor in previous.items()}

    stats = save_sparse_update(previous, current, tmp_path, step=1, base_step=0)
    manifest = apply_sparse_update(receiver, tmp_path)

    expected = {name: to_compute_tensor(tensor) for name, tensor in current.items()}
    assert torch.equal(receiver["model.layers.0.weight"], expected["model.layers.0.weight"])
    assert torch.equal(receiver["model.layers.0.bias"], expected["model.layers.0.bias"])
    assert stats.changed_numel == 2
    assert manifest["stats"]["changed_numel"] == 2


def test_sparse_update_noops_when_bf16_view_is_unchanged(tmp_path):
    previous = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    current = {"weight": torch.tensor([1.0001, 2.0001], dtype=torch.float32)}
    receiver = {name: to_compute_tensor(tensor) for name, tensor in previous.items()}

    stats = save_sparse_update(previous, current, tmp_path, step=1, base_step=0)
    apply_sparse_update(receiver, tmp_path)

    assert torch.equal(receiver["weight"], to_compute_tensor(previous["weight"]))
    assert stats.changed_numel == 0
    assert stats.sparsity == 1.0


def test_sparse_update_rejects_wrong_base_step(tmp_path):
    previous = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    current = {"weight": torch.tensor([1.0, 3.0], dtype=torch.float32)}
    receiver = {name: to_compute_tensor(tensor) for name, tensor in previous.items()}

    save_sparse_update(previous, current, tmp_path, step=4, base_step=3)

    with pytest.raises(ValueError, match="base step mismatch"):
        apply_sparse_update(receiver, tmp_path, expected_base_step=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_sparse_update_to_params_applies_on_gpu(tmp_path):
    """Test that kernel-format sparse patches apply directly to GPU model params."""
    # Build a tiny model with known weights
    model = nn.Sequential(nn.Linear(4, 3, bias=True), nn.Linear(3, 2, bias=True)).cuda()
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # Simulate a training step: change a few values
    new_params = {name: param.data.clone() for name, param in model.named_parameters()}
    new_params["0.weight"][0, 0] = 42.0
    new_params["0.weight"][1, 2] = 17.0
    new_params["1.bias"][0] = 99.0

    # Build state dicts for diffing (on CPU, as the trainer would)
    previous_state = {name: tensor.cpu() for name, tensor in original_params.items()}
    current_state = {name: tensor.cpu() for name, tensor in new_params.items()}

    # Write sparse patch in native dtype (kernel format: compute_dtype=None)
    stats = save_sparse_update(previous_state, current_state, tmp_path, step=1, base_step=0, compute_dtype=None)

    assert stats.changed_numel == 3
    assert stats.patched_tensors == 2  # 0.weight and 1.bias

    # Apply the patch directly to the GPU model
    model_step = 0
    manifest = apply_sparse_update_to_params(model, tmp_path, expected_base_step=model_step, device="cuda")

    # Verify the GPU params now match the target
    for name, expected in new_params.items():
        actual = dict(model.named_parameters())[name].data
        assert torch.equal(actual, expected), f"Mismatch for {name}"

    assert manifest["step"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_sparse_update_to_params_rejects_wrong_base_step(tmp_path):
    """Test that GPU param application validates base_step."""
    model = nn.Linear(4, 3, bias=True).cuda()
    params = dict(model.named_parameters())

    previous_state = {"weight": params["weight"].data.cpu().clone()}
    current_state = {"weight": previous_state["weight"].clone()}
    current_state["weight"][0, 0] = 5.0

    save_sparse_update(previous_state, current_state, tmp_path, step=2, base_step=1, compute_dtype=None)

    with pytest.raises(ValueError, match="base step mismatch"):
        apply_sparse_update_to_params(model, tmp_path, expected_base_step=0, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_sparse_update_to_params_preserves_unchanged_values(tmp_path):
    """Test that only patched values change; all others stay the same."""
    model = nn.Linear(8, 4, bias=False).cuda()
    original = {name: param.data.clone() for name, param in model.named_parameters()}

    # Change 2 out of 32 values
    current = original["weight"].clone()
    current[0, 0] = 1.0
    current[3, 7] = 2.0

    previous_state = {"weight": original["weight"].cpu()}
    current_state = {"weight": current.cpu()}

    stats = save_sparse_update(previous_state, current_state, tmp_path, step=1, base_step=0, compute_dtype=None)
    assert stats.changed_numel == 2

    apply_sparse_update_to_params(model, tmp_path, expected_base_step=0, device="cuda")

    result = dict(model.named_parameters())["weight"].data
    assert torch.equal(result, current)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_sparse_update_to_params_chained_patches(tmp_path):
    """Test applying two consecutive patches (simulates multi-step training)."""
    model = nn.Linear(6, 4, bias=False).cuda()

    # Step 0 -> Step 1
    s0 = {name: param.data.clone().cpu() for name, param in model.named_parameters()}
    s1 = s0["weight"].clone()
    s1[0, 0] = 10.0
    s1[1, 1] = 20.0

    save_sparse_update(s0, {"weight": s1}, tmp_path / "step_1", step=1, base_step=0, compute_dtype=None)
    apply_sparse_update_to_params(model, tmp_path / "step_1", expected_base_step=0, device="cuda")

    # Step 1 -> Step 2
    s2 = s1.clone()
    s2[0, 0] = 30.0
    s2[2, 2] = 40.0

    save_sparse_update({"weight": s1}, {"weight": s2}, tmp_path / "step_2", step=2, base_step=1, compute_dtype=None)
    apply_sparse_update_to_params(model, tmp_path / "step_2", expected_base_step=1, device="cuda")

    result = dict(model.named_parameters())["weight"].data
    assert result[0, 0].item() == 30.0
    assert result[1, 1].item() == 20.0
    assert result[2, 2].item() == 40.0
