import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_SOURCE = Path(__file__).parents[4] / "src/prime_rl/trainer/models/layers/cp_mamba.py"
_SPEC = importlib.util.spec_from_file_location("cp_mamba_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
cp_mamba = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_mamba)


class _FakeMixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_size = 1
        self.conv_dim = 3
        self.num_heads = 1
        self.n_groups = 1
        self.ssm_state_size = 1
        self.head_dim = 1
        self.chunk_size = 1
        self.time_step_limit = None
        self.variance_epsilon = 1e-5

        self.in_proj = torch.nn.Linear(1, 5, bias=False)
        self.in_proj.weight.data.copy_(torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]]))
        self.A_log = torch.nn.Parameter(torch.zeros(1))
        self.D = torch.nn.Parameter(torch.ones(1))
        self.dt_bias = torch.nn.Parameter(torch.zeros(1))
        self.conv1d = torch.nn.Conv1d(3, 3, kernel_size=2, groups=3, bias=False)
        self.conv1d.weight.data.fill_(1.0)
        self.norm = types.SimpleNamespace(weight=torch.ones(1), variance_epsilon=self.variance_epsilon, group_size=1)
        self.out_proj = torch.nn.Identity()


def _install_fake_scan(monkeypatch, seen_seq_idx: list[torch.Tensor]) -> None:
    def fake_scan(hidden_states, time_step, A, B, C, **kwargs):
        seen_seq_idx.append(kwargs["seq_idx"].clone())
        return hidden_states, None

    ssd_combined = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_combined.mamba_chunk_scan_combined = fake_scan
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", types.ModuleType("mamba_ssm.ops"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton", types.ModuleType("mamba_ssm.ops.triton"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", ssd_combined)


def test_mamba_cp_forward_resets_conv_and_scan_at_packed_boundaries(monkeypatch):
    """Changing document A must not change document B under Ulysses CP."""
    seen_seq_idx: list[torch.Tensor] = []
    _install_fake_scan(monkeypatch, seen_seq_idx)
    monkeypatch.setattr(cp_mamba, "seq_to_head_parallel", lambda value, *_: value)
    monkeypatch.setattr(cp_mamba, "head_to_seq_parallel", lambda value, *_: value)

    mixer = _FakeMixer()
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    first = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    second = first.clone()
    second[0, 1, 0] = 200.0

    first_out = cp_mamba.mamba_cp_forward(mixer, first, None, 0, 1, cu_seqlens)
    second_out = cp_mamba.mamba_cp_forward(mixer, second, None, 0, 1, cu_seqlens)

    assert torch.equal(seen_seq_idx[0], torch.tensor([[0, 0, 1, 1]], dtype=torch.int32))
    assert torch.equal(seen_seq_idx[1], seen_seq_idx[0])
    torch.testing.assert_close(first_out[:, 2:], second_out[:, 2:])


def test_mamba_cp_forward_rejects_local_boundaries():
    with pytest.raises(ValueError, match="full sequence length 4"):
        cp_mamba._build_seq_idx(torch.tensor([0, 2], dtype=torch.int32), 4, torch.device("cpu"))
