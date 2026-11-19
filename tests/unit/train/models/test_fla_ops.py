import pytest
import torch
import torch.nn.functional as F
from fla.modules.conv import causal_conv1d as fla_causal_conv1d
from fla.modules.norm.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.utils.testing import assert_close

from prime_rl.trainer.models.layers import fla_ops

pytestmark = [pytest.mark.gpu]

CU_SEQLENS = [[0, 2048], [0, 15, 100, 333, 1000], [0, 64, 65, 300, 1500, 1501, 4096]]
OUTPUT_RATIO = 0.005
GRAD_RATIO = 0.008


def leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_() for tensor in tensors]


def strided(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape[:-1], 2 * shape[-1], device="cuda", dtype=dtype)[..., ::2]


def make_input(shape: tuple[int, ...], contiguous: bool, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    tensor = torch.randn(*shape, device="cuda", dtype=dtype) if contiguous else strided(shape, dtype)
    assert tensor.is_contiguous() == contiguous
    return tensor


def run(fn, inputs: list[torch.Tensor], grad_out: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    out = fn(*inputs)
    out.backward(grad_out)
    return out.detach(), [tensor.grad for tensor in inputs]


def check(name: str, ref: tuple[torch.Tensor, list[torch.Tensor]], ours: tuple[torch.Tensor, list[torch.Tensor]]):
    assert_close(f"{name} out", ref[0], ours[0], OUTPUT_RATIO)
    for index, (ref_grad, our_grad) in enumerate(zip(ref[1], ours[1], strict=True)):
        assert our_grad.dtype == ref_grad.dtype and our_grad.shape == ref_grad.shape
        assert_close(f"{name} grad {index}", ref_grad, our_grad, GRAD_RATIO)


@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("contiguous", [True, False])
def test_causal_conv1d_matches_fla(cu_seqlens: list[int], contiguous: bool):
    torch.manual_seed(0)
    tokens, channels, width = cu_seqlens[-1], 768, 4
    cu_seqlens = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32)
    x = make_input((1, tokens, channels), contiguous)
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)

    ref = run(
        lambda x, w: fla_causal_conv1d(x=x, weight=w, activation="silu", cu_seqlens=cu_seqlens)[0],
        leaves(x, weight),
        grad_out,
    )
    ours = run(lambda x, w: fla_ops.causal_conv1d(x, w, "silu", cu_seqlens), leaves(x, weight), grad_out)
    check("conv", ref, ours)


@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("contiguous", [True, False])
def test_chunk_gated_delta_rule_matches_fla(cu_seqlens: list[int], contiguous: bool):
    torch.manual_seed(0)
    tokens, heads, head_dim = cu_seqlens[-1], 4, 128
    cu_seqlens = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32)
    q, k, v = (make_input((1, tokens, heads, head_dim), contiguous) for _ in range(3))
    g = F.logsigmoid(make_input((1, tokens, heads), contiguous, torch.float32))
    beta = make_input((1, tokens, heads), contiguous).sigmoid()
    grad_out = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)

    def reference(q, k, v, g, beta):
        out, _ = fla_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
        )
        return out

    ref = run(reference, leaves(q, k, v, g, beta), grad_out)
    ours = run(
        lambda q, k, v, g, beta: fla_ops.chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens),
        leaves(q, k, v, g, beta),
        grad_out,
    )
    check("gdn", ref, ours)


@pytest.mark.parametrize("cu_seqlens", CU_SEQLENS)
@pytest.mark.parametrize("contiguous", [True, False])
def test_chunk_kda_matches_fla(cu_seqlens: list[int], contiguous: bool):
    torch.manual_seed(0)
    tokens, heads, head_dim, lower_bound = cu_seqlens[-1], 4, 128, -5.0
    cu_seqlens = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int32)
    q, k, v, g = (make_input((1, tokens, heads, head_dim), contiguous) for _ in range(4))
    beta = make_input((1, tokens, heads), contiguous)
    A_log = torch.rand(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads * head_dim, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)

    def reference(q, k, v, g, beta, A_log, dt_bias):
        out, _ = fla_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )
        return out

    ref = run(reference, leaves(q, k, v, g, beta, A_log, dt_bias), grad_out)
    ours = run(
        lambda q, k, v, g, beta, A_log, dt_bias: fla_ops.chunk_kda(
            q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens
        ),
        leaves(q, k, v, g, beta, A_log, dt_bias),
        grad_out,
    )
    check("kda", ref, ours)


@pytest.mark.parametrize("tokens", [1, 1000, 4096])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_rms_norm_gated_matches_fla(tokens: int, contiguous: bool, activation: str):
    torch.manual_seed(0)
    hidden, eps = 128, 1e-6
    x, gate = (make_input((tokens, hidden), contiguous) for _ in range(2))
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    ref = run(
        lambda x, gate, w: fla_rms_norm_gated(x=x, g=gate, weight=w, bias=None, activation=activation, eps=eps),
        leaves(x, gate, weight),
        grad_out,
    )
    ours = run(
        lambda x, gate, w: fla_ops.rms_norm_gated(x, gate, w, eps, activation), leaves(x, gate, weight), grad_out
    )
    check("norm", ref, ours)


def test_ops_compile_without_graph_breaks():
    torch.manual_seed(0)
    tokens, heads, head_dim = 1000, 4, 128
    cu_seqlens = torch.tensor([0, 15, 100, 333, 1000], device="cuda", dtype=torch.int32)
    hidden = heads * head_dim

    def layer(mixed, conv_weight, gate_logits, beta_logits, A_log, dt_bias, out_gate, norm_weight):
        mixed = fla_ops.causal_conv1d(mixed, conv_weight, "silu", cu_seqlens)
        q, k, v = (part.reshape(1, tokens, heads, head_dim) for part in mixed.chunk(3, dim=-1))
        gdn = fla_ops.chunk_gated_delta_rule(
            q, k, v, F.logsigmoid(gate_logits[..., 0].float()), beta_logits.sigmoid(), cu_seqlens
        )
        kda = fla_ops.chunk_kda(q, k, v, gate_logits, beta_logits, A_log, dt_bias, -5.0, cu_seqlens)
        core = (gdn + kda).reshape(-1, head_dim)
        return fla_ops.rms_norm_gated(core, out_gate.reshape(-1, head_dim), norm_weight, 1e-6, "silu")

    inputs = leaves(
        torch.randn(1, tokens, 3 * hidden, device="cuda", dtype=torch.bfloat16),
        torch.randn(3 * hidden, 4, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16),
        torch.rand(heads, device="cuda", dtype=torch.float32),
        torch.randn(hidden, device="cuda", dtype=torch.float32),
        torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(head_dim, device="cuda", dtype=torch.bfloat16),
    )
    grad_out = torch.randn(tokens * heads, head_dim, device="cuda", dtype=torch.bfloat16)
    eager = run(layer, inputs, grad_out)
    compiled = run(torch.compile(layer, fullgraph=True), leaves(*inputs), grad_out)
    check("compiled", eager, compiled)


@pytest.mark.parametrize("op_name", ["causal_conv1d", "chunk_gated_delta_rule", "chunk_kda", "rms_norm_gated"])
def test_opcheck(op_name: str):
    torch.manual_seed(0)
    tokens, heads, head_dim = 300, 2, 64
    cu_seqlens = torch.tensor([0, 15, 100, 300], device="cuda", dtype=torch.int32)

    def bf16(*shape: int, requires_grad: bool = True) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)

    def f32(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)

    qkv_shape = (1, tokens, heads, head_dim)
    op, args = {
        "causal_conv1d": (fla_ops._causal_conv1d, (bf16(1, tokens, 96), bf16(96, 4), "silu", cu_seqlens)),
        "chunk_gated_delta_rule": (
            fla_ops._chunk_gated_delta_rule,
            (
                bf16(*qkv_shape),
                bf16(*qkv_shape),
                bf16(*qkv_shape),
                F.logsigmoid(f32(1, tokens, heads)).detach().requires_grad_(),
                bf16(1, tokens, heads).sigmoid().detach().requires_grad_(),
                head_dim**-0.5,
                cu_seqlens,
            ),
        ),
        "chunk_kda": (
            fla_ops._chunk_kda,
            (
                bf16(*qkv_shape),
                bf16(*qkv_shape),
                bf16(*qkv_shape),
                bf16(*qkv_shape),
                bf16(1, tokens, heads),
                f32(heads),
                f32(heads * head_dim),
                head_dim**-0.5,
                -5.0,
                cu_seqlens,
            ),
        ),
        "rms_norm_gated": (
            fla_ops._rms_norm_gated,
            (bf16(tokens, head_dim), bf16(tokens, head_dim), bf16(head_dim), 1e-6, "silu"),
        ),
    }[op_name]
    torch.library.opcheck(op, args, test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))
