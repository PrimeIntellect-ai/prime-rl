import json

import pytest
import torch

from prime_rl.trainer.sparse_update import (
    OptimizerSparseUpdateHook,
    SparseUpdateWriter,
    _global_indices,
    apply_sparse_update,
    commit_sparse_update,
)


def test_global_indices_for_column_shard():
    local = torch.tensor([0, 1, 2, 3])
    assert _global_indices(local, (2, 2), (2, 4), (0, 1)).tolist() == [1, 2, 5, 6]


def test_sparse_update_roundtrip_and_chain(tmp_path):
    weight = torch.arange(8, dtype=torch.bfloat16)
    writer = SparseUpdateWriter(tmp_path, rank=0)
    writer.initialize({"weight": weight}, base_step=0)

    weight[[1, 6]] = torch.tensor([-2, -3], dtype=torch.bfloat16)
    patch = writer.write({"weight": weight}, target_step=1)
    commit_sparse_update(tmp_path, target_step=1, base_step=0, world_size=1)

    assert patch.tensors[0].changed == 2
    from safetensors.torch import load_file

    assert load_file(tmp_path / "step_1" / "rank_0.safetensors")["t0_indices"].dtype == torch.int32
    payload = json.loads((tmp_path / "step_1" / "rank_0.json").read_text())
    assert payload["tensors"][0]["changed"] == 2
    receiver = {"weight": torch.arange(8, dtype=torch.bfloat16)}
    assert apply_sparse_update(receiver, tmp_path / "step_1", expected_base_step=0) == 1
    torch.testing.assert_close(receiver["weight"], weight)

    weight[3] = -4
    writer.write({"weight": weight}, target_step=2)
    commit_sparse_update(tmp_path, target_step=2, base_step=1, world_size=1)
    assert apply_sparse_update(receiver, tmp_path / "step_2", expected_base_step=1) == 2
    torch.testing.assert_close(receiver["weight"], weight)

    with pytest.raises(ValueError, match="base mismatch"):
        apply_sparse_update(receiver, tmp_path / "step_2", expected_base_step=0)


def test_optimizer_hook_captures_post_step_values(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    writer = SparseUpdateWriter(tmp_path, rank=0)
    writer.initialize({"weight": parameter}, base_step=0)
    step = 1
    hook = OptimizerSparseUpdateHook(optimizer, writer, lambda: {"weight": parameter}, lambda: step)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    hook.remove()
    commit_sparse_update(tmp_path, target_step=1, base_step=0, world_size=1)

    receiver = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    apply_sparse_update(receiver, tmp_path / "step_1", expected_base_step=0)
    torch.testing.assert_close(receiver["weight"], parameter)
