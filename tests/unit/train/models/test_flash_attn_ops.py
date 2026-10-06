import pytest
import torch

from prime_rl.trainer.models.layers import attn
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig

pytestmark = [pytest.mark.gpu]

DOC_LENS = [37, 128, 91]
NUM_HEADS = 8
NUM_KV_HEADS = 2

RAW_FUNCS = {
    2: attn.flash_attn_varlen_func,
    3: attn.flash_attn_3_varlen_func,
    4: attn.flash_attn_4_varlen_func,
}
MIN_COMPUTE_CAPABILITY = {2: (8, 0), 3: (9, 0), 4: (9, 0)}
MAX_COMPUTE_CAPABILITY = {2: None, 3: (9, 0), 4: None}


def _skip_unless_supported(version: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if RAW_FUNCS[version] is None:
        pytest.skip(f"FlashAttention {version} not installed")
    capability = torch.cuda.get_device_capability()
    if capability < MIN_COMPUTE_CAPABILITY[version]:
        pytest.skip(f"FlashAttention {version} needs compute capability >= {MIN_COMPUTE_CAPABILITY[version]}")
    max_capability = MAX_COMPUTE_CAPABILITY[version]
    if max_capability is not None and capability[0] > max_capability[0]:
        pytest.skip(f"FlashAttention {version} needs compute capability {max_capability}")


def _cu_seqlens() -> torch.Tensor:
    lens = torch.tensor([0, *DOC_LENS], dtype=torch.int32, device="cuda")
    return lens.cumsum(0, dtype=torch.int32)


@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize("sliding_window", [None, 16])
def test_compiled_flash_attention_fullgraph_matches_eager(version, sliding_window):
    _skip_unless_supported(version)
    torch._dynamo.reset()
    torch.manual_seed(0)
    head_dim = 64
    config = AttentionConfig(
        hidden_size=256,
        head_dim=head_dim,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
    )
    module = ATTN_IMPL2CLASS[f"flash_attention_{version}"](config).to(device="cuda", dtype=torch.bfloat16)
    if sliding_window is not None:
        module.sliding_window = sliding_window
    compiled = torch.compile(module, fullgraph=True)

    hidden_states = torch.randn(1, sum(DOC_LENS), config.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = _cu_seqlens()
    max_seqlen = max(DOC_LENS)

    def run(fn):
        module.zero_grad()
        x = hidden_states.detach().clone().requires_grad_()
        out, _ = fn(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out.float().sum().backward()
        return out.detach(), x.grad, module.q_proj.weight.grad.clone()

    eager = run(module)
    compiled_result = run(compiled)

    out, x_grad, q_proj_grad = compiled_result
    expected_out, expected_x_grad, expected_q_proj_grad = eager
    assert torch.equal(out, expected_out)
    assert torch.equal(q_proj_grad, expected_q_proj_grad)
    # Inductor sums the q/k/v projection grads in fp32 and rounds once; eager rounds after each add.
    torch.testing.assert_close(x_grad, expected_x_grad, atol=2e-2, rtol=1.6e-2)
