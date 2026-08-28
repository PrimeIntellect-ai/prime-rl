import sys
import types

import pytest
import torch

from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2

pytestmark = [pytest.mark.gpu]


def _install_fake_scan(monkeypatch, seen_seq_idx: list[torch.Tensor]) -> None:
    def fake_scan(hidden_states, time_step, state_decay, state_input, state_output, **kwargs):
        assert kwargs["return_final_states"] is False
        seen_seq_idx.append(kwargs["seq_idx"].clone())
        return hidden_states

    ssd_combined = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_combined.mamba_chunk_scan_combined = fake_scan
    monkeypatch.setitem(sys.modules, "mamba_ssm", types.ModuleType("mamba_ssm"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", types.ModuleType("mamba_ssm.ops"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton", types.ModuleType("mamba_ssm.ops.triton"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", ssd_combined)


def _make_mamba(monkeypatch, seen_seq_idx: list[torch.Tensor]) -> NemotronHMamba2:
    _install_fake_scan(monkeypatch, seen_seq_idx)
    config = NemotronHConfig(
        hidden_size=1,
        hybrid_override_pattern="M",
        mamba_num_heads=1,
        mamba_head_dim=1,
        n_groups=1,
        ssm_state_size=1,
        conv_kernel=2,
        chunk_size=1,
        use_conv_bias=False,
    )
    mamba = NemotronHMamba2(config).cuda()
    mamba.in_proj.weight.data.fill_(1.0)
    mamba.conv1d.weight.data.fill_(1.0)
    mamba.norm.weight.data.fill_(1.0)
    mamba.out_proj = torch.nn.Identity()
    return mamba


def test_mamba_resets_conv_and_scan_at_packed_boundaries(monkeypatch):
    """Changing one document must not change the following document."""
    seen_seq_idx: list[torch.Tensor] = []
    mamba = _make_mamba(monkeypatch, seen_seq_idx)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    first = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], device="cuda")
    second = first.clone()
    second[0, 1, 0] = 200.0

    first_output = mamba(first, cu_seqlens)
    second_output = mamba(second, cu_seqlens)

    expected_seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32, device="cuda")
    assert torch.equal(seen_seq_idx[0], expected_seq_idx)
    assert torch.equal(seen_seq_idx[1], expected_seq_idx)
    torch.testing.assert_close(first_output[:, 2:], second_output[:, 2:])


def test_mamba_builds_sequence_ids_for_uneven_packs(monkeypatch):
    seen_seq_idx: list[torch.Tensor] = []
    mamba = _make_mamba(monkeypatch, seen_seq_idx)
    cu_seqlens = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device="cuda")
    hidden_states = torch.arange(6, dtype=torch.float32, device="cuda").reshape(1, 6, 1)

    mamba(hidden_states, cu_seqlens)

    assert torch.equal(seen_seq_idx[0], torch.tensor([[0, 1, 1, 1, 2, 2]], dtype=torch.int32, device="cuda"))
