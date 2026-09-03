"""The fused DeepSeek V4 sparse-attention kernel against its float32 gather oracle.

Everything here calls `prime_rl::dsv4_sparse_attn` directly on hand-built tensors, so the kernel
is exercised without any of the modeling code that normally produces its inputs. The tests that
need a real `DeepseekV4Attention` live beside the other sparse tests in `test_deepseek_v4.py`,
which already owns the Flash-shaped config.

Only the real DeepSeek V4 Flash shapes appear below. The backward does not compile below 32 heads
and the forward does not compile at `head_dim = 32`, and no configuration this model runs is
anywhere near those, so a smaller shape would only test a kernel nobody instantiates.
"""

import math

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4.attention import sparse_attention_gather

# Guarded so collection survives on an install without tilelang: `pytest -m "not gpu"` imports every
# module before deselecting by marker, and the kernel pulls in tilelang, which only the `gpu` extra
# provides and only on linux. The CPU CI job does install it (`uv sync --all-extras`, and tilelang
# imports without a GPU), but a non-linux or extras-free checkout genuinely lacks it.
try:
    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn
except ImportError:
    dsv4_sparse_attn = None  # type: ignore

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(dsv4_sparse_attn is None, reason="the DS V4 sparse attention kernel needs tilelang"),
]

# The production DeepSeek V4 Flash CSA layer: 64 heads over 512 channels, each query gathering
# `sliding_window + index_topk = 128 + 512` slots from a single KV group, in bfloat16.
_HEADS, _DIM, _TOPK, _KV_GROUP = 64, 512, 640, 1
_SM_SCALE = _DIM**-0.5

# Three shapes. The first is aligned to both of the backward's tile sizes and the second to
# neither: `preprocess` tiles the query axis at 32 and `postprocess` tiles the KV axis at 64, so a
# remainder tile in either is a distinct code path. `200 % 32 = 8` and `1000 % 64 = 40`. The third
# carries a batch, which nothing else here does: `Q` and `Indices` are indexed by batch and the
# `dKV[by, Indices[by, ...]]` atomics scatter per batch entry, so a batch stride dropped anywhere
# in that chain is invisible at batch 1. Sequence lengths stay modest because the float32 oracle
# materializes a `(batch, seq_len, topk, head_dim)` gather, roughly 640 KB per token even before
# its backward.
_SHAPES = [(1, 256, 1024), (1, 200, 1000), (3, 128, 768)]
_SHAPE_IDS = ["aligned", "misaligned", "batched"]

# A quarter of the gather slots hold the sentinel, which is what a real query with a short
# window or a saturated top-k looks like: the masked slots still cost a load and a GEMM column.
_SENTINEL_FRACTION = 0.25

# Bounds on the largest absolute deviation against each tensor's own scale, not element-wise:
# every entry is a sum over hundreds of terms, so the near-zero entries are the ones whose
# summands cancelled, and an element-wise relative bound would read out that cancellation noise.
#
# Each bound is the tightest round number holding over the three shapes above at 60
# incoming-gradient draws each, against the float32 oracle of `_float32_leaves`. Measured worst
# case over those 180 draws, with the value the fixed-seed draws these tests actually run reach
# in parentheses:
#   out    3.4e-3 (2.4e-3), under one bfloat16 ulp at full scale (2**-8 = 3.9e-3)
#   lse    2.8e-7 (1.9e-7), float32 throughout on both sides
#   dq     5.2e-3 (3.1e-3)
#   dkv    4.1e-3 (3.1e-3)
#   dsink  8.1e-3 (5.0e-3)
_OUT_RTOL = 1e-2
_LSE_RTOL = 5e-7
_DQ_RTOL = 1e-2
# The vendored kernel this one forked from rounds `P` and `dP` to bfloat16 before the `dKV` GEMMs
# while the float32 oracle keeps them in float32, which is worth about 1.6e-3 of the 4.1e-3 above.
# The rest is the bfloat16 `kv` the two sides share. Neither effect needs a looser bound than the
# other gradients get: what used to need one was the oracle's own bfloat16 leaf, see
# `_float32_leaves`.
_DKV_RTOL = 1e-2
# The loosest fitting of the five, at 1.2x rather than the 1.9x to 2.9x the others carry. The sink
# gradient is a full reduction over every query in the row, so it cancels harder than anything
# else here and its worst draw pairs a large deviation with a small scale.
_DSINK_RTOL = 1e-2

# Compiled against eager. The forward and the log-sum-exp are bit-identical, but `dKV` is not
# comparable that way on either side: the backward scatters it with `atomic_addx4`, so its
# summation order is whatever the scheduler picks and the same eager call against itself moves by
# the same amount. Measured worst case is 3.1e-3 on `dkv`, one bfloat16 ulp of its largest entry,
# and 1.4e-7 on `dsink`; `dq` is exact.
_COMPILE_RTOL = 1e-2


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    actual, reference = actual.float(), reference.float()
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _build_indices(batch: int, seq_len: int, seq_len_kv: int, sentinel_fraction: float) -> torch.Tensor:
    """`(batch, seq_len, kv_group, topk)` int32 gather slots, a mix of valid picks and sentinel.

    Valid KV positions are `[0, seq_len_kv - 1)`; `seq_len_kv - 1` is the zero sentinel. The
    picks are drawn without replacement, since a real query never gathers the same key twice and
    a duplicate would take twice its share of the softmax on both sides of the comparison.
    """
    sentinel = seq_len_kv - 1
    n_valid = seq_len_kv - 1
    assert n_valid >= _TOPK, "not enough valid KV positions to fill the gather slots without repeats"
    picks = torch.rand(batch, seq_len, n_valid, device="cuda").argsort(dim=-1)[..., :_TOPK]
    masked = torch.rand(batch, seq_len, _TOPK, device="cuda") < sentinel_fraction
    picks = torch.where(masked, torch.full_like(picks, sentinel), picks)
    return picks.to(torch.int32).unsqueeze(2).contiguous()


def _inputs(
    batch: int, seq_len: int, seq_len_kv: int, *, sentinel_fraction: float = _SENTINEL_FRACTION
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`q`, `kv`, `indices`, `sinks` at the Flash shapes, detached values rather than leaves.

    `q` and `kv` are drawn at unit variance, so `q . k * sm_scale` has unit variance too and the
    unscaled sink logit, also unit variance, is a real competitor in the softmax rather than a
    term the dot products drown out.
    """
    torch.manual_seed(seq_len * 100003 + seq_len_kv)
    with torch.device("cuda"):
        q = torch.randn(batch, seq_len, _HEADS, _DIM, dtype=torch.bfloat16)
        kv = torch.randn(batch, seq_len_kv, _KV_GROUP, _DIM, dtype=torch.bfloat16)
        # The trailing position is the sentinel's target and reads as zeros.
        kv[:, -1] = 0
        # A float32 sinks leaf, deliberately: the kernel casts `dsink` back to the leaf's dtype,
        # so a bfloat16 leaf would round both sides onto the same coarse grid and report a
        # deviation the rounding chose rather than one the kernel earned.
        sinks = torch.randn(_HEADS, dtype=torch.float32)
    return q, kv, _build_indices(batch, seq_len, seq_len_kv, sentinel_fraction), sinks


def _leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone().requires_grad_(True) for tensor in tensors)


def _float32_leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The same values as leaves of the float32 oracle, which is what makes the oracle exact.

    `sparse_attention_gather` computes in whatever dtype it is handed, so widening the leaves is
    what puts the oracle in float32 at all. Widening changes none of the values, a bfloat16 number
    being exactly representable in float32, so the oracle answers for exactly the numbers the
    kernel saw. Feeding it the bfloat16 leaves instead would round each of the roughly 164k
    per-slot gradient contributions back to bfloat16 and accumulate about 200 of them per KV
    position on that coarse grid, worth `sqrt(200) * 2**-9`, about 2.6e-2 on `dkv`. That is an
    artifact of how the oracle is built and not a property of the kernel: measured against a
    float32-leaf oracle the same kernel deviates by 4.1e-3, and the two oracles disagree with each
    other by 2.6e-2.
    """
    return tuple(tensor.detach().float().clone().requires_grad_(True) for tensor in tensors)


def _reference_lse(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """The base-2, sink-inclusive log-sum-exp of `sparse_attention_gather`'s own softmax.

    The oracle returns only the attention output, so its denominator is recomputed here from the
    same gather and the same unscaled sink logit, in float32.
    """
    sentinel = kv.shape[1] - 1
    slot_idx = indices[:, :, 0, :].to(torch.int64)
    batch_idx = torch.arange(kv.shape[0], device=kv.device)[:, None, None]
    keys = kv[batch_idx, slot_idx, 0].float()
    logits = torch.einsum("bshd,bskd->bshk", q.float(), keys) * _SM_SCALE
    logits = logits.masked_fill((slot_idx == sentinel).unsqueeze(2), float("-inf"))
    sink_logits = sinks.float().reshape(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    return torch.cat([logits, sink_logits], dim=-1).logsumexp(dim=-1) * math.log2(math.e)


@pytest.mark.parametrize(("batch", "seq_len", "seq_len_kv"), _SHAPES, ids=_SHAPE_IDS)
def test_kernel_forward_matches_the_gather_reference(batch, seq_len, seq_len_kv):
    """Output and log-sum-exp against the float32 gather oracle, which has identical semantics."""
    q, kv, indices, sinks = _inputs(batch, seq_len, seq_len_kv)

    with torch.no_grad():
        out, lse = dsv4_sparse_attn(q, kv, indices, sinks, _SM_SCALE)
        # Float32 inputs to the oracle, which is what runs it in float32: it follows the dtype it
        # is handed. Widened here rather than inside it, so the exact answer is what the bound is
        # measured against instead of one rounded back to bfloat16.
        reference_out = sparse_attention_gather(q.float(), kv.float(), indices, sinks, _SM_SCALE)
        reference_lse = _reference_lse(q, kv, indices, sinks)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert lse.shape == (batch, seq_len, _HEADS) and lse.dtype == torch.float32
    _assert_relative(out, reference_out, _OUT_RTOL, "output")
    _assert_relative(lse, reference_lse, _LSE_RTOL, "lse")


@pytest.mark.parametrize(("batch", "seq_len", "seq_len_kv"), _SHAPES, ids=_SHAPE_IDS)
def test_kernel_backward_matches_autograd_through_the_reference(batch, seq_len, seq_len_kv):
    """All three differentiable inputs, each against its own bound.

    `dsink` is the one term the kernel forms in torch rather than in tilelang, out of the `Delta`
    the backward returns, so it is the assertion that would catch a wrong `Lse` convention.
    """
    q, kv, indices, sinks = _inputs(batch, seq_len, seq_len_kv)
    kernel_q, kernel_kv, kernel_sinks = _leaves(q, kv, sinks)
    reference_q, reference_kv, reference_sinks = _float32_leaves(q, kv, sinks)

    out, _lse = dsv4_sparse_attn(kernel_q, kernel_kv, indices, kernel_sinks, _SM_SCALE)
    # One weight tensor for both losses, so the two backwards are the same function of the same
    # numbers and any difference belongs to the kernel.
    weight = torch.randn_like(out)
    (out * weight).sum().backward()

    reference_out = sparse_attention_gather(reference_q, reference_kv, indices, reference_sinks, _SM_SCALE)
    (reference_out * weight).sum().backward()

    assert reference_sinks.grad is not None and reference_sinks.grad.norm() > 0, (
        "vacuous probe: the reference gave the sinks no gradient, so the sink bound cannot fail"
    )
    _assert_relative(kernel_q.grad, reference_q.grad, _DQ_RTOL, "dq")
    _assert_relative(kernel_kv.grad, reference_kv.grad, _DKV_RTOL, "dkv")
    _assert_relative(kernel_sinks.grad, reference_sinks.grad, _DSINK_RTOL, "dsink")


def test_kernel_traces_under_torch_compile():
    """`torch.compile(fullgraph=True)` through the op, forward and backward.

    `apply_compile` in `prime_rl/trainer/model.py` compiles each decoder layer, so every real
    training step traces this op and takes its `register_fake` and its compiled backward. Nothing
    else covers that: a fake returning the wrong shape or dtype surfaces as a downstream shape
    error at the first compiled step of a run, and an op the tracer cannot see through silently
    costs the whole graph. `fullgraph=True` is the assertion, since it refuses to break.
    """
    q, kv, indices, sinks = _inputs(1, 256, 1024)
    eager_leaves = _leaves(q, kv, sinks)
    compiled_leaves = _leaves(q, kv, sinks)

    def attend(q: torch.Tensor, kv: torch.Tensor, sinks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dsv4_sparse_attn(q, kv, indices, sinks, _SM_SCALE)

    out, lse = attend(*eager_leaves)
    # One weight tensor for both losses, so the two backwards are the same function of the same
    # numbers and any difference belongs to the tracing.
    weight = torch.randn_like(out)
    (out * weight).sum().backward()

    compiled_out, compiled_lse = torch.compile(attend, fullgraph=True)(*compiled_leaves)
    (compiled_out * weight).sum().backward()

    assert compiled_out.shape == out.shape and compiled_out.dtype == out.dtype
    assert compiled_lse.shape == lse.shape and compiled_lse.dtype == lse.dtype
    torch.testing.assert_close(compiled_out, out, rtol=0, atol=0)
    torch.testing.assert_close(compiled_lse, lse, rtol=0, atol=0)
    for label, compiled_leaf, eager_leaf in zip(("dq", "dkv", "dsink"), compiled_leaves, eager_leaves):
        assert compiled_leaf.grad is not None and compiled_leaf.grad.norm() > 0, (
            f"{label}: the compiled backward left the leaf without a gradient"
        )
        _assert_relative(compiled_leaf.grad, eager_leaf.grad, _COMPILE_RTOL, label)
