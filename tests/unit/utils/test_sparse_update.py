import pytest
import torch

from prime_rl.utils.sparse_update import apply_sparse_update, save_sparse_update, to_compute_tensor


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
