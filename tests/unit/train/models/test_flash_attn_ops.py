import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from prime_rl.configs.trainer import CompileConfig
from prime_rl.trainer.model import apply_compile
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
SUPPORTS_COMPUTE_CAPABILITY = {
    2: lambda capability: capability >= (8, 0),
    3: lambda capability: capability == (9, 0),
    4: lambda capability: capability >= (9, 0),
}


def _skip_unless_supported(version: int, head_dim: int = 128) -> None:
    if not torch.cuda.is_available() or RAW_FUNCS[version] is None:
        pytest.skip(f"FlashAttention {version} not available")
    if not SUPPORTS_COMPUTE_CAPABILITY[version](torch.cuda.get_device_capability()):
        pytest.skip(f"FlashAttention {version} does not support this GPU")
    if head_dim % 8 != 0 and version != 2:
        pytest.skip("only FA2 pads the head dim")


def _attention(version: int, func, causal: bool = True, window_size: tuple[int, int] | None = None):
    cu_seqlens = torch.tensor([0, *DOC_LENS], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    max_seqlen = max(DOC_LENS)
    kwargs = {"causal": causal}
    if window_size is not None:
        kwargs["window_size"] = window_size

    def attention(q, k, v):
        if version == 4:
            out, _ = func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, **kwargs)
            return out
        return func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, **kwargs)

    return attention


def _forward_backward(attention, q, k, v, dout):
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    out = attention(*leaves)
    out.backward(dout)
    return out.detach(), *(t.grad for t in leaves)


@pytest.mark.parametrize("version", [2, 3, 4])
@pytest.mark.parametrize("head_dim", [128, 60])
@pytest.mark.parametrize(
    ("causal", "window_size"),
    [(True, None), (True, (15, 0)), (False, None)],
    ids=["causal", "sliding_window", "non_causal"],
)
def test_op_matches_library_function(version, head_dim, causal, window_size):
    _skip_unless_supported(version, head_dim)
    torch.manual_seed(0)
    total = sum(DOC_LENS)
    q = torch.randn(total, NUM_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)

    expected_out, expected_dq, expected_dk, expected_dv = _forward_backward(
        _attention(version, RAW_FUNCS[version], causal, window_size), q, k, v, dout
    )
    out, dq, dk, dv = _forward_backward(_attention(version, OPS[version], causal, window_size), q, k, v, dout)
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
    cu_seqlens = torch.tensor([0, *DOC_LENS], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
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
    _skip_unless_supported(version, head_dim)
    torch._dynamo.reset()
    torch.manual_seed(0)
    total = sum(DOC_LENS)
    q, k, v = (
        torch.randn(NUM_HEADS, total, head_dim, device="cuda", dtype=torch.bfloat16).transpose(0, 1) for _ in range(3)
    )
    dout = torch.randn(total, NUM_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    attention = _attention(version, OPS[version])

    eager = _forward_backward(attention, q, k, v, dout)
    compiled = _forward_backward(torch.compile(attention, fullgraph=True), q, k, v, dout)

    for expected_tensor, actual_tensor in zip(eager, compiled):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("version", [2, 3])
def test_apply_compile_traces_max_seqlen_symbolically(version, monkeypatch):
    _skip_unless_supported(version)
    monkeypatch.setattr(torch.compiler.config, "dynamic_sources", torch.compiler.config.dynamic_sources)
    torch._dynamo.reset()
    config = AttentionConfig(
        hidden_size=256,
        head_dim=64,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
    )
    model = nn.Module()
    model.model = nn.Module()
    attention = ATTN_IMPL2CLASS[f"flash_attention_{version}"](config).to(device="cuda", dtype=torch.bfloat16)
    model.model.layers = nn.ModuleList([checkpoint_wrapper(attention)])
    apply_compile(model, CompileConfig(fullgraph=True))

    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for doc_lens in ([128, 64, 64], [256]):
        cu_seqlens = torch.tensor([0, *doc_lens], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
        torch._dynamo.mark_dynamic(cu_seqlens, 0)
        hidden_states = torch.randn(1, 256, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out, _ = model.model.layers[0](hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max(doc_lens))
        out.sum().backward()

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before == 1
