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
OPS = {
    2: attn.flash_attn_2_varlen_op,
    3: attn.flash_attn_3_varlen_op,
    4: attn.flash_attn_4_varlen_op,
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


def _call(version: int, func, q, k, v, cu_seqlens, causal: bool, window_size):
    kwargs = {"causal": causal}
    if window_size is not None:
        kwargs["window_size"] = window_size
    if version == 4:
        out, _ = func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, **kwargs)
        return out
    max_seqlen = max(DOC_LENS)
    return func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)


def _forward_backward(version: int, func, q, k, v, dout, cu_seqlens, causal: bool, window_size):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = _call(version, func, q, k, v, cu_seqlens, causal, window_size)
    out.backward(dout)
    return out.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [128, 60])
@pytest.mark.parametrize(
    ("causal", "window_size"),
    [(True, None), (True, (15, 0)), (False, None)],
    ids=["causal", "sliding_window", "non_causal"],
)
def test_op_matches_library_function(version, head_dim, causal, window_size):
    _skip_unless_supported(version)
    if head_dim % 8 != 0 and version != 2:
        pytest.skip("only FA2 pads the head dim")
    torch.manual_seed(0)
    total = sum(DOC_LENS)
    q = torch.randn(total, NUM_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    cu_seqlens = _cu_seqlens()

    expected = _forward_backward(version, RAW_FUNCS[version], q, k, v, dout, cu_seqlens, causal, window_size)
    actual = _forward_backward(version, OPS[version], q, k, v, dout, cu_seqlens, causal, window_size)

    out, dq, dk, dv = actual
    expected_out, expected_dq, expected_dk, expected_dv = expected
    assert torch.equal(out, expected_out)
    assert torch.equal(dq, expected_dq)
    # FA3's GQA backward accumulates dk/dv across query heads with atomics, so even two raw calls differ.
    torch.testing.assert_close(dk, expected_dk)
    torch.testing.assert_close(dv, expected_dv)


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


@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [128, 60])
def test_compiled_op_matches_eager_for_head_major_inputs(version, head_dim):
    _skip_unless_supported(version)
    if head_dim % 8 != 0 and version != 2:
        pytest.skip("only FA2 pads the head dim")
    torch._dynamo.reset()
    torch.manual_seed(0)
    total = sum(DOC_LENS)
    q, k, v = (
        torch.randn(NUM_HEADS, total, head_dim, device="cuda", dtype=torch.bfloat16).transpose(0, 1) for _ in range(3)
    )
    dout = torch.randn(total, NUM_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = _cu_seqlens()

    def attention(q, k, v):
        return _call(version, OPS[version], q, k, v, cu_seqlens, causal=True, window_size=None)

    def run(fn):
        leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        out = fn(*leaves)
        out.backward(dout)
        return out.detach(), *(t.grad for t in leaves)

    eager = run(attention)
    compiled = run(torch.compile(attention, fullgraph=True))

    for expected_tensor, actual_tensor in zip(eager, compiled):
        assert torch.equal(actual_tensor, expected_tensor)
