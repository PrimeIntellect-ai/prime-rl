import pytest
import torch

from prime_rl.trainer.models.nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.mamba import NemotronHMamba2

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def mamba():
    torch.manual_seed(0)
    config = NemotronHConfig(
        hidden_size=256,
        hybrid_override_pattern="M",
        mamba_num_heads=8,
        mamba_head_dim=64,
        n_groups=2,
        ssm_state_size=64,
        conv_kernel=4,
        chunk_size=64,
    )
    return NemotronHMamba2(config).cuda().to(torch.bfloat16)


def test_mamba_resets_conv_and_scan_at_packed_boundaries(mamba):
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")
    first = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    second = first.detach().clone()
    second[:, :3] *= 20

    first_output = mamba(first, cu_seqlens)
    second_output = mamba(second, cu_seqlens)
    torch.testing.assert_close(first_output[:, 3:], second_output[:, 3:])
    first_output[:, 3:].float().square().sum().backward()
    assert torch.count_nonzero(first.grad[:, :3]) == 0


def test_mamba_builds_sequence_ids_for_uneven_packs(mamba):
    cu_seqlens = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(1, 6, 256, device="cuda", dtype=torch.bfloat16)
    packed = mamba(hidden_states, cu_seqlens)
    separate = torch.cat(
        [
            mamba(hidden_states[:, start:end], torch.tensor([0, end - start], dtype=torch.int32, device="cuda"))
            for start, end in [(0, 1), (1, 4), (4, 6)]
        ],
        dim=1,
    )
    torch.testing.assert_close(packed, separate, rtol=1e-2, atol=1e-2)
