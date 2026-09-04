"""Every DeepSeek V4 kernel check that needs a GPU, in two sections.

The first calls `prime_rl::dsv4_sparse_attn` directly on hand-built tensors, so the kernel is
exercised without any of the modeling code that normally produces its inputs, and its numerics are
compared against a float32 gather oracle. The second builds real `DeepseekV4Attention` layers and
asserts that the modeling code hands the kernel inputs it can act on, that the indices it
constructs address exactly the keys the dense mask admits, and that a packed row still answers each
document as if it stood alone.

Only the real DeepSeek V4 Flash shapes appear here. The backward does not compile below 32 heads
and the forward does not compile at `head_dim = 32`, and no configuration this model runs is
anywhere near those, so a smaller shape would only test a kernel nobody instantiates. The toy
`_MODEL` config that `test_deepseek_v4.py` uses cannot reach any of this: the kernel does not tile
4 heads over 32 channels, and the sparse path's slot padding, top-k saturation and index arithmetic
are all invisible at that size.
"""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config, eager_reference
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.utils.utils import default_dtype

# Guarded so collection survives on an install without tilelang: `pytest -m "not gpu"` imports every
# module before deselecting by marker, and the kernel pulls in tilelang, which only the `gpu` extra
# provides and only on linux. The CPU CI job does install it (`uv sync --all-extras`, and tilelang
# imports without a GPU), but a non-linux or extras-free checkout genuinely lacks it.
try:
    import tilelang
    from tilelang import language as T

    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn
except ImportError:
    dsv4_sparse_attn = None  # type: ignore

pytestmark = [pytest.mark.gpu]

# The fused kernel needs tilelang, which only the `gpu` extra provides and only on linux. Several
# tests below reach the modeling code without ever compiling a kernel, so this is a per-test skip
# rather than part of `pytestmark`.
requires_tilelang = pytest.mark.skipif(
    dsv4_attention.dsv4_sparse_attn is None, reason="the sparse attention kernel needs tilelang"
)


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


def _randomize(module: nn.Module) -> None:
    """Draw non-degenerate values for every parameter.

    These modules allocate with `torch.empty`, and the values `init_weights` would write are
    themselves degenerate for testing: norm gains default to ones and the sinks and position
    biases to zeros, each of which leaves the path it controls indistinguishable from a no-op.
    The position bias is drawn wide because it is a softmax logit over a pooling window; at the
    projections' std the gate would stay near uniform.
    """
    for name, param in module.named_parameters():
        with torch.no_grad():
            if name.endswith("scale"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("base"):
                param.normal_(mean=0.0, std=0.5)
            elif name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
            elif name.endswith("sinks") or name.endswith("position_bias"):
                param.normal_(mean=0.0, std=1.0)
            else:
                param.normal_(mean=0.0, std=0.02)


def _packed_context(doc_lens: tuple[int, ...], dtype: torch.dtype, config: DeepseekV4Config) -> PackedContext:
    """The context `DeepseekV4Model` would hand its attention layers for a row of `doc_lens`.

    A single-element `doc_lens` gives back the single-document context, which is what the unpacked
    half of a packing comparison runs at. `dtype` types the mask and the rotary tables, and has to
    be the one the caller runs at.
    """
    with torch.device("cuda"), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(config)
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens, device="cuda"),
        dtype=dtype,
        device=torch.device("cuda"),
    )


def _doc_slice(doc_lens: tuple[int, ...], index: int) -> slice:
    start = sum(doc_lens[:index])
    return slice(start, start + doc_lens[index])


# The module-level cases run in float32. `kv_proj` sees a different number of rows packed than
# alone and cuBLAS may tile the two differently, so they never match bit for bit, and in bfloat16
# that floor would swallow the cross-document leakage these tests exist to catch.
_PACKED_RTOL = 1e-5
# Gradients are bounded against the tensor's own scale instead: they are sums over the whole row,
# so their near-zero entries are the ones whose summands cancelled, and an element-wise relative
# bound would read out that cancellation noise rather than a document leak.
_PACKED_GRAD_RTOL = 1e-5


def _take_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Detach whatever gradients have accumulated and clear them for the next run."""
    grads = {name: None if param.grad is None else param.grad.clone() for name, param in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return grads


def _compare_accumulated_grads(
    module: nn.Module, expected: dict[str, torch.Tensor | None], rtol: float = _PACKED_GRAD_RTOL
) -> None:
    """Compare the gradients now on `module` against a snapshot taken from an earlier backward.

    Allows for a parameter that legitimately receives nothing: the Lightning Indexer reaches the
    loss only through integer top-k indices, so neither run may hand its parameters a gradient.
    """
    for name, param in module.named_parameters():
        if expected[name] is None:
            assert param.grad is None, f"{name} received a gradient per document but not packed"
            continue
        assert param.grad is not None, f"{name} received no gradient per document"
        _assert_relative(param.grad, expected[name], rtol, name)


def _set_attn_impl(module: nn.Module, impl: str) -> None:
    """Pin the CSA attention implementation on every attention layer `module` owns.

    `modules()` yields `module` itself, so this covers a lone attention layer as well as a model
    holding several of them.
    """
    for submodule in module.modules():
        if isinstance(submodule, DeepseekV4Attention):
            submodule.attn_impl = impl


# The real DeepSeek V4-Flash attention shapes, written out as a literal so nothing here depends on a local HF
# cache. Both sections of this file run them and nothing else: the kernel does not tile smaller ones, and the
# sparse path it serves only exists at this size. The MoE fields are shrunk to nothing, since
# `DeepseekV4Attention` reads none of them.
_FLASH_MODEL = dict(
    vocab_size=64,
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=1,
    head_dim=512,
    q_lora_rank=1024,
    o_groups=8,
    o_lora_rank=1024,
    qk_rope_head_dim=64,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    sliding_window=128,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    layer_types=["compressed_sparse_attention", "heavily_compressed_attention"],
    num_hidden_layers=2,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    max_position_embeddings=65536,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=3,
    n_shared_experts=1,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hash_layers=1,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rope_scaling={
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
)

# The hand-built tensors of the first section describe the same CSA layer `_FLASH_MODEL` does, so
# they are read off it rather than written out again: 64 heads over 512 channels, each query
# gathering `sliding_window + index_topk = 128 + 512` slots from a single KV group, in bfloat16.
_HEADS = _FLASH_MODEL["num_attention_heads"]
_DIM = _FLASH_MODEL["head_dim"]
_KV_GROUP = _FLASH_MODEL["num_key_value_heads"]
_TOPK = _FLASH_MODEL["sliding_window"] + _FLASH_MODEL["index_topk"]
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

# A quarter of the gather slots are masked, which is what a real query with a short window or a
# saturated top-k looks like: the masked slots still cost a GEMM column.
_MASKED_FRACTION = 0.25

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


def _gather_reference(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor, scale: float
) -> torch.Tensor:
    """Attention of each query over the KV positions it gathers, in `q.dtype`.

    `q` is `(batch, seq_len, heads, dim)`, `kv` is `(batch, seq_len_kv, 1, dim)`, `indices` is
    `(batch, seq_len, 1, topk)` int32 addressing `kv`'s position axis, and `sinks` is `(heads,)`.
    A slot holding `-1` marks an absent key and is masked out.

    Arithmetic follows `q.dtype`, so handing it widened tensors is what makes it the float32
    oracle for a kernel that accumulates in float32.
    """
    slot_idx = indices[:, :, 0, :].to(torch.int64)
    absent = slot_idx < 0
    # Torch negative indexing wraps, so a `-1` would silently gather the last KV position. The
    # kernel needs no such clamp: TileLang guards its gather and zero-fills instead.
    batch_idx = torch.arange(kv.shape[0], device=kv.device)[:, None, None]
    keys = kv[batch_idx, slot_idx.clamp(min=0), 0].to(q.dtype)

    logits = torch.einsum("bshd,bskd->bshk", q, keys) * scale
    logits = logits.masked_fill(absent.unsqueeze(2), float("-inf"))

    # The sink logit is unscaled, as the kernel's is. Subtracting the row max keeps a fully masked
    # row finite and keeps the exponentials from overflowing.
    sink_logits = sinks.to(logits.dtype).reshape(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    combined_logits = torch.cat([logits, sink_logits], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)

    return torch.einsum("bshk,bskd->bshd", probs[..., :-1], keys)


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    actual, reference = actual.float(), reference.float()
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _build_indices(batch: int, seq_len: int, seq_len_kv: int, masked_fraction: float) -> torch.Tensor:
    """`(batch, seq_len, kv_group, topk)` int32 gather slots, a mix of valid picks and `-1`.

    Valid KV positions are `[0, seq_len_kv)`; `-1` marks an absent key. The picks are drawn
    without replacement, since a real query never gathers the same key twice and a duplicate
    would take twice its share of the softmax on both sides of the comparison.
    """
    assert seq_len_kv >= _TOPK, "not enough valid KV positions to fill the gather slots without repeats"
    picks = torch.rand(batch, seq_len, seq_len_kv, device="cuda").argsort(dim=-1)[..., :_TOPK]
    masked = torch.rand(batch, seq_len, _TOPK, device="cuda") < masked_fraction
    picks = torch.where(masked, torch.full_like(picks, -1), picks)
    return picks.to(torch.int32).unsqueeze(2).contiguous()


def _inputs(
    batch: int, seq_len: int, seq_len_kv: int, *, masked_fraction: float = _MASKED_FRACTION
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
        # A float32 sinks leaf, deliberately: the kernel casts `dsink` back to the leaf's dtype,
        # so a bfloat16 leaf would round both sides onto the same coarse grid and report a
        # deviation the rounding chose rather than one the kernel earned.
        sinks = torch.randn(_HEADS, dtype=torch.float32)
    return q, kv, _build_indices(batch, seq_len, seq_len_kv, masked_fraction), sinks


def _leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone().requires_grad_(True) for tensor in tensors)


def _float32_leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The same values as leaves of the float32 oracle, which is what makes the oracle exact.

    `_gather_reference` computes in whatever dtype it is handed, so widening the leaves is
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
    """The base-2, sink-inclusive log-sum-exp of `_gather_reference`'s own softmax.

    The oracle returns only the attention output, so its denominator is recomputed here from the
    same gather and the same unscaled sink logit, in float32.
    """
    slot_idx = indices[:, :, 0, :].to(torch.int64)
    batch_idx = torch.arange(kv.shape[0], device=kv.device)[:, None, None]
    keys = kv[batch_idx, slot_idx.clamp(min=0), 0].float()
    logits = torch.einsum("bshd,bskd->bshk", q.float(), keys) * _SM_SCALE
    logits = logits.masked_fill((slot_idx < 0).unsqueeze(2), float("-inf"))
    sink_logits = sinks.float().reshape(1, 1, -1, 1).expand(*logits.shape[:-1], 1)
    return torch.cat([logits, sink_logits], dim=-1).logsumexp(dim=-1) * math.log2(math.e)


@pytest.mark.parametrize(("batch", "seq_len", "seq_len_kv"), _SHAPES, ids=_SHAPE_IDS)
@requires_tilelang
def test_kernel_forward_matches_the_gather_reference(batch, seq_len, seq_len_kv):
    """Output and log-sum-exp against the float32 gather oracle, which has identical semantics."""
    q, kv, indices, sinks = _inputs(batch, seq_len, seq_len_kv)

    with torch.no_grad():
        out, lse = dsv4_sparse_attn(q, kv, indices, sinks, _SM_SCALE)
        # Float32 inputs to the oracle, which is what runs it in float32: it follows the dtype it
        # is handed. Widened here rather than inside it, so the exact answer is what the bound is
        # measured against instead of one rounded back to bfloat16.
        reference_out = _gather_reference(q.float(), kv.float(), indices, sinks, _SM_SCALE)
        reference_lse = _reference_lse(q, kv, indices, sinks)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert lse.shape == (batch, seq_len, _HEADS) and lse.dtype == torch.float32
    _assert_relative(out, reference_out, _OUT_RTOL, "output")
    _assert_relative(lse, reference_lse, _LSE_RTOL, "lse")


@requires_tilelang
def test_fully_masked_query_reads_as_zero_keys():
    """A query with no keys at all must emit exactly zero, on the sink term alone.

    The sink carries the softmax denominator by itself here, which is what keeps `lse` finite
    instead of dividing by a zero-seeded `sumexp`.

    This does not pin TileLang's out-of-range guard, although a fully masked query is where that
    guard does the most work: a masked slot is seeded to `-inf` before the gather is ever read, so
    its probability is zero and finite garbage would multiply out unnoticed. Only an `inf` or a
    `NaN` in an unguarded read would reach these assertions.
    `test_tilelang_zero_fills_an_out_of_range_gather` covers the guard itself.
    """
    batch, seq_len, seq_len_kv = 1, 256, 1024
    q, kv, indices, sinks = _inputs(batch, seq_len, seq_len_kv)
    indices[:, 0] = -1  # the first query gathers nothing

    with torch.no_grad():
        out, lse = dsv4_sparse_attn(q, kv, indices, sinks, _SM_SCALE)

    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0])), "a fully masked query must emit exactly zero"
    expected_lse = sinks.float() * math.log2(math.e)
    assert torch.allclose(lse[:, 0], expected_lse.expand_as(lse[:, 0])), "lse must fall back to the sink term"
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()


@requires_tilelang
def test_tilelang_zero_fills_an_out_of_range_gather():
    """An index outside `[0, n_positions)` must read as zeros, not as whatever it points at.

    Both kernels index `KV` and `dKV` by a value read from `Indices` and never clamp it, so a `-1`
    slot is safe only because TileLang's `LegalizeSafeMemoryAccess` pass wraps every global access
    it cannot prove in range with `0 <= idx < extent`. That pass is on by default and has a single
    off switch, but nothing in TileLang documents it as a contract, and this project pins
    `tilelang>=0.1.8` with no upper bound, so an ordinary dependency bump could take it away.

    The probe is a standalone gather rather than the real kernels, which cannot observe this: they
    seed a masked slot's logit to `-inf` before the gather is read, so its probability is zero and
    any finite garbage multiplies out. With nothing masking the result, a lost guard shows up
    immediately as a non-zero row.
    """

    @tilelang.jit(out_idx=[-1])
    def gather(n_positions: int, dim: int):
        n_slots = T.dynamic("n_slots")

        @T.prim_func
        def main(
            Src: T.Tensor([n_positions, dim], "float32"),
            Idx: T.Tensor([n_slots], "int32"),
            Out: T.Tensor([n_slots, dim], "float32"),
        ):
            with T.Kernel(n_slots, threads=dim) as slot:
                tile = T.alloc_shared([dim], "float32")
                for d in T.Parallel(dim):
                    tile[d] = Src[Idx[slot], d]
                for d in T.Parallel(dim):
                    Out[slot, d] = tile[d]

        return main

    n_positions, dim = 8, 32
    src = (torch.arange(n_positions * dim, device="cuda", dtype=torch.float32) + 1).view(n_positions, dim)
    # Two in range, then the four ways out: the `-1` this project marks an absent key with, a far
    # negative, one past the end, and far past it.
    slots = torch.tensor([0, 3, -1, -1000, n_positions, n_positions + 5], dtype=torch.int32, device="cuda")

    out = gather(n_positions, dim)(src, slots)

    assert torch.equal(out[0], src[0]) and torch.equal(out[1], src[3]), "an in-range index must gather its row"
    assert torch.equal(out[2:], torch.zeros_like(out[2:])), (
        "TileLang no longer zero-fills an out-of-range gather, so the kernels' unclamped `Indices` reads are unsafe"
    )


@pytest.mark.parametrize(("batch", "seq_len", "seq_len_kv"), _SHAPES, ids=_SHAPE_IDS)
@requires_tilelang
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

    reference_out = _gather_reference(reference_q, reference_kv, indices, reference_sinks, _SM_SCALE)
    (reference_out * weight).sum().backward()

    assert reference_sinks.grad is not None and reference_sinks.grad.norm() > 0, (
        "vacuous probe: the reference gave the sinks no gradient, so the sink bound cannot fail"
    )
    _assert_relative(kernel_q.grad, reference_q.grad, _DQ_RTOL, "dq")
    _assert_relative(kernel_kv.grad, reference_kv.grad, _DKV_RTOL, "dkv")
    _assert_relative(kernel_sinks.grad, reference_sinks.grad, _DSINK_RTOL, "dsink")


@requires_tilelang
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


# Everything below reaches the kernel through the modeling code rather than on hand-built tensors, at the same
# Flash shapes. What it adds is the index construction: a CSA query's slot padding, its top-k saturation and
# the arithmetic mapping a compressed entry to a buffer position all live in `DeepseekV4Attention`, not in the
# kernel, and none of them is expressible at the toy shapes `test_deepseek_v4.py` runs.

_FLASH_CSA_LAYER = 0
_FLASH_COMPRESS_RATE = _FLASH_MODEL["compress_rates"]["compressed_sparse_attention"]

# Document layouts for the sparse path, at `compress_rate = 4`. The first four leave every query
# short of `index_topk = 512` readable entries, so the `-1` padding of the pick slots carries
# the difference; `(2600,)` saturates the picks instead, which the toy shapes cannot express at
# all. `(3,)` compresses to no entries whatsoever, leaving the local window alone to answer.
_FLASH_DOC_LENS = [(517, 1019), (3,), (300,), (3, 129, 1021), (2600,)]
_FLASH_DOC_IDS = ["two-docs", "no-entries", "one-short-doc", "three-docs", "saturated-topk"]


def _flash_config(attn_impl: str = "kernel") -> DeepseekV4Config:
    return DeepseekV4Config(**_FLASH_MODEL, _attn_impl=attn_impl)


def flash_attention(layer_idx: int, dtype: torch.dtype = torch.float32, attn_impl: str = "kernel") -> nn.Module:
    """One attention layer at the real DeepSeek V4 Flash shapes, 126M parameters of it."""
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(_flash_config(attn_impl), layer_idx=layer_idx)
    _randomize(module)
    return module


def _flash_hidden_states(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two leaves carrying identical values, one per attention path.

    Batch 1: at these shapes the score tensors are the bulk of the memory and a second batch
    entry repeats the first without covering anything new.
    """
    with torch.device("cuda"):
        hidden = torch.randn(1, seq_len, _FLASH_MODEL["hidden_size"])
    return hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)


def _record_attention(monkeypatch) -> dict[str, torch.Tensor]:
    """Capture what each attention path is handed and what it returns, from real forwards.

    Both implementations are module-level functions the layer looks up by name, so patching them
    records the call the layer actually made rather than a reconstruction of it.
    """
    recorded: dict[str, torch.Tensor] = {}
    real_eager = dsv4_attention.eager_attention_with_sinks
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def eager(query, key, value, sinks, attention_mask, **kwargs):
        recorded["mask"] = attention_mask
        recorded["eager"] = real_eager(query, key, value, sinks, attention_mask, **kwargs)
        return recorded["eager"]

    def kernel(q, kv_buf, indices, sinks, scale):
        recorded["kv_buf"], recorded["indices"] = kv_buf, indices
        recorded["kernel"] = real_kernel(q, kv_buf, indices, sinks, scale)
        return recorded["kernel"]

    monkeypatch.setattr(dsv4_attention, "eager_attention_with_sinks", eager)
    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", kernel)
    return recorded


def _selected_positions(indices: torch.Tensor, n_positions: int) -> torch.Tensor:
    """`(seq_len, n_positions)` bool: which KV positions each query's gather slots address."""
    slots = indices[0, :, 0, :].long()
    # `-1` marks an absent key; it goes into one throwaway column that is sliced back off.
    safe = torch.where(slots >= 0, slots, n_positions)
    selected = torch.zeros((indices.shape[1], n_positions + 1), dtype=torch.bool, device=indices.device)
    return selected.scatter_(1, safe, True)[:, :n_positions]


@requires_tilelang
@pytest.mark.parametrize("doc_lens", _FLASH_DOC_LENS, ids=_FLASH_DOC_IDS)
def test_sparse_indices_address_exactly_the_keys_the_dense_mask_admits(doc_lens, monkeypatch):
    """A CSA layer's gather slots must reach the keys the dense rules admit, key for key.

    One selection rendered two independent ways: the dense rendering concatenates the indexer's
    picks onto a sliding mask built straight from the document boundaries rather than from
    `window_indices`; the sparse one writes the window and the picks into a single index tensor
    over a gathered KV buffer. Nothing in the layer compares them, and every way of getting
    the sparse side wrong (a window base off by one, an entry index not offset by the token count,
    a stale index left over from a previous layout) still produces a finite output.

    Pure set equality on integers, so no tolerance enters; bfloat16 is only what the kernel these
    indices are recorded from insists on. Both the index tensor and the picks the dense rendering
    is built from come from a real forward of the module on the input.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, _flash_config())
    hidden_states = _flash_hidden_states(sum(doc_lens))[0].detach().to(torch.bfloat16)
    recorded = _record_attention(monkeypatch)

    real_compressor = module.compressor.forward

    def compressor(hidden_states, q_residual, packed):
        compressed_kv, recorded["picks"] = real_compressor(hidden_states, q_residual, packed)
        return compressed_kv, recorded["picks"]

    monkeypatch.setattr(module.compressor, "forward", compressor)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    n_positions = recorded["kv_buf"].shape[1]
    seq_len, n_entries = sum(doc_lens), n_positions - sum(doc_lens)
    assert n_entries == sum(length // _FLASH_COMPRESS_RATE for length in doc_lens)
    block_bias = eager_reference.block_bias_from_indices(recorded["picks"], n_entries, torch.float32)
    sliding_mask = eager_reference.build_sliding_window_mask(
        tok_doc_idx=packed.tok_doc_idx, sliding_window=_FLASH_MODEL["sliding_window"], dtype=torch.float32
    )
    admitted = torch.cat([sliding_mask[0, 0], block_bias[0, 0]], dim=-1) == 0
    if n_entries:
        assert admitted[:, seq_len:].any(), "vacuous probe: no query reads a compressed entry"

    selected = _selected_positions(recorded["indices"], n_positions)
    assert torch.equal(selected, admitted), "the sparse and dense paths select different keys"


@requires_tilelang
@pytest.mark.parametrize("doc_lens", _FLASH_DOC_LENS, ids=_FLASH_DOC_IDS)
def test_sparse_indices_are_in_range_and_never_repeat_a_key(doc_lens, monkeypatch):
    """Every gather slot addresses a real KV position, and no query counts a key twice.

    A negative or out-of-range index reads whatever lies next to the buffer instead of raising,
    which the kernel that will consume these has no way to detect. A repeat is worse than
    wasteful: the duplicated key takes twice its share of the softmax, silently reweighting the
    output. The `-1` padding is exempt from uniqueness, since padding every query out to a fixed
    slot count is exactly what it is for.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, _flash_config())
    hidden_states = _flash_hidden_states(sum(doc_lens))[0].detach().to(torch.bfloat16)
    recorded = _record_attention(monkeypatch)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    indices, n_positions = recorded["indices"], recorded["kv_buf"].shape[1]
    # The exact width: the window plus the picks the row actually affords, tile-aligned. A row with
    # fewer entries than `index_topk` gets a narrower slot count, not a `-1`-padded one.
    n_entries = n_positions - sum(doc_lens)
    n_picks = min(_FLASH_MODEL["index_topk"], n_entries)
    tile = dsv4_attention._SLOT_TILE
    n_slots = indices.shape[-1]
    assert n_slots == ((_FLASH_MODEL["sliding_window"] + n_picks + tile - 1) // tile) * tile
    assert (indices >= -1).all(), "a gather slot addresses a KV position below the `-1` marker"
    assert (indices <= n_positions - 1).all(), "a gather slot addresses past the end of the KV buffer"

    slot_idx = indices[0, :, 0, :].long()
    # `-1` is counted in a throwaway column, since padding repeats it by design.
    safe = torch.where(slot_idx >= 0, slot_idx, n_positions)
    counts = torch.zeros((slot_idx.shape[0], n_positions + 1), dtype=torch.int32, device="cuda")
    counts.scatter_add_(1, safe, torch.ones_like(safe, dtype=torch.int32))
    assert (counts[:, :n_positions] <= 1).all(), "a query gathers the same key twice"


@requires_tilelang
@pytest.mark.parametrize("doc_lens", _FLASH_DOC_LENS, ids=_FLASH_DOC_IDS)
def test_absent_slots_are_marked_negative_rather_than_pointed_at_a_pad_row(doc_lens, monkeypatch):
    """An unused gather slot must hold `-1`, never a position that `kv_buf` actually has.

    This is the whole of the contract between `SparseAttnInputs.build` and the kernel, which masks
    on `Indices[...] < 0` and on nothing else. The design it replaced appended a zero row to
    `kv_buf` and pointed unused slots at that row's index instead, which looks equally valid: the
    pad index is in range and the key it names is zero either way.

    Under the kernel's masking the two are not equivalent at all. A non-negative pad index passes
    the `< 0` test, so the slot counts as a live key, and the zero row it names scores `q . 0 = 0`
    and enters the softmax with weight `exp(0)` rather than nothing. At these shapes the first
    query of a row carries 639 pad slots against 1 real key, so its denominator would be wrong by
    roughly three orders of magnitude, and nothing would raise.

    The neighbouring index tests do fail if the pad row comes back, but on incidental symptoms:
    every pad slot naming one row reads as "a query gathers the same key twice", and the extra row
    reads as a compressed-entry count that does not match the layout. Neither names the cause, so
    this asserts the contract directly.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, _flash_config())
    hidden_states = _flash_hidden_states(sum(doc_lens))[0].detach().to(torch.bfloat16)
    recorded = _record_attention(monkeypatch)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    indices, kv_buf = recorded["indices"], recorded["kv_buf"]
    n_entries = sum(length // _FLASH_COMPRESS_RATE for length in doc_lens)
    assert kv_buf.shape[1] == sum(doc_lens) + n_entries, (
        "kv_buf holds more than the token stream and its compressed entries, so `build` is padding "
        "it with rows that the `-1` marker makes unnecessary"
    )
    assert kv_buf[:, -1].abs().max() > 0, "the last row of kv_buf is zero, which is what a pad row looks like"

    # The first query of the row can read one key, its own token: the window clips to the document
    # and no complete compressed entry precedes it. Every other slot is padding, so this counts the
    # padding directly rather than inferring it.
    first_query = indices[0, 0, 0]
    assert (first_query >= 0).sum() == 1, (
        f"the first query holds {(first_query >= 0).sum().item()} non-negative slots against the 1 key it "
        "may read, so absent slots are addressing a KV position instead of holding `-1`"
    )
    assert (first_query[first_query < 0] == -1).all(), "an absent slot is negative but is not the `-1` marker"


# One CSA layer in bfloat16, so `_PACKED_RTOL` (float32, and three orders of magnitude tighter than a kernel
# accumulating bfloat16 inputs) does not apply, but neither does the whole-model bound `test_deepseek_v4.py`
# carries, which is sized for four hyper-connected layers amplifying a bfloat16 expert floor. Each bound below
# is the tightest round number holding over 30 seeds; the worst is 1.4e-3 on the output and 7.6e-3 on a
# gradient, against 6.9e-4 and 6.2e-3 on the fixed seed the test actually runs. The gradient bound is the
# tighter fit of the two, at 1.3x: every seed lands between 5.8e-3 and 7.6e-3, so the bound sits just above a
# well-sampled ceiling rather than above a long tail.
_KERNEL_RTOL, _KERNEL_GRAD_RTOL = 5e-3, 1e-2

# `compress_rate = 4` yields 129 + 254 = 383 compressed entries, under `index_topk = 512`, so
# every readable entry is picked and the indexer's ordering cannot differ packed from alone. A
# saturated layout would let a bfloat16 tie flip a pick and move the output for a reason that has
# nothing to do with document independence.
_KERNEL_DOC_LENS = (517, 1019)


# Document layouts for the kernel-against-eager comparison: two single-document rows and two
# packed ones. `(2600,)` is left out on purpose for the reason `_KERNEL_DOC_LENS` gives below, and
# `(3,)` is kept because it compresses to no entries at all, so almost every gather slot is the
# `-1` marker and the local window alone has to answer.
_EAGER_KERNEL_DOC_LENS = [(300,), (3,), (517, 1019), (3, 129, 1021)]
_EAGER_KERNEL_DOC_IDS = ["one-doc", "no-entries", "two-docs", "three-docs"]

# A bfloat16 kernel against a float32 dense softmax, so these are three orders of magnitude looser
# than a float32 comparison would be, and looser again than `_KERNEL_RTOL`, which compares
# two bfloat16 runs of the same path. Each is the tightest round number holding over 30 seeds on
# all four layouts: the worst observed is 6.1e-3 on the output and 1.6e-2 on a gradient, against
# 4.9e-3 and 9.8e-3 on the fixed seed the test runs.
_EAGER_KERNEL_RTOL, _EAGER_KERNEL_GRAD_RTOL = 8e-3, 2e-2


@pytest.mark.parametrize("doc_lens", _EAGER_KERNEL_DOC_LENS, ids=_EAGER_KERNEL_DOC_IDS)
@requires_tilelang
def test_sparse_attention_kernel_matches_eager(doc_lens, monkeypatch):
    """The fused kernel against the naive dense softmax, single-document and packed.

    Every other kernel test reaches eager only transitively: the kernel is compared to the gather
    reference on hand-built tensors, and the gather reference is compared to eager on a real
    module. Nothing joined the two ends on the same input, so a disagreement that the gather
    reference happened to share with the kernel would go unseen. This closes that loop, and it is
    the only place the modeling code's own index construction meets the kernel in a numeric
    comparison rather than a set-equality one.

    Both halves hold the same weights, the bfloat16 module's, one widened to float32 rather than
    drawn again, and both start from the same bfloat16-representable hidden states, so input
    rounding is not one of the differences being measured. What is left is the attention path:
    a dense mask and a full softmax on one side, a 640-slot gather and an online softmax on the
    other.

    The call count is load-bearing rather than decoration: `dsv4_sparse_attn` raises today instead
    of demoting a dtype it cannot run, but a regression that reintroduced a fallback would leave
    this comparing eager against eager and passing for the wrong reason.
    """
    seq_len = sum(doc_lens)
    kernel_module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    eager_module = copy.deepcopy(kernel_module).float()
    _set_attn_impl(eager_module, "eager")

    with torch.device("cuda"):
        base = torch.randn(1, seq_len, _FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    kernel_input = base.clone().requires_grad_(True)
    eager_input = base.float().clone().requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, kv_buf, indices, sinks, scale):
        calls.append(q.shape[1])
        return real_kernel(q, kv_buf, indices, sinks, scale)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    kernel_output, _ = kernel_module(kernel_input, packed=_packed_context(doc_lens, torch.bfloat16, _flash_config()))
    eager_output, _ = eager_module(eager_input, packed=_packed_context(doc_lens, torch.float32, _flash_config()))
    assert calls == [seq_len], f"the forward never reached the kernel, calls={calls}"
    _assert_relative(kernel_output, eager_output, _EAGER_KERNEL_RTOL, "attention output")

    # One weight tensor for both losses, so any difference belongs to the attention path alone.
    with torch.device("cuda"):
        weight = torch.randn(1, seq_len, _FLASH_MODEL["hidden_size"], dtype=torch.float32)
    (eager_output * weight).sum().backward()
    eager_grads = _take_grads(eager_module)
    assert eager_grads["sinks"] is not None and eager_grads["sinks"].norm() > 0, (
        "vacuous probe: the sinks received no gradient, so the comparison below cannot fail on them"
    )

    (kernel_output * weight.bfloat16()).sum().backward()
    _compare_accumulated_grads(kernel_module, eager_grads, rtol=_EAGER_KERNEL_GRAD_RTOL)
    _assert_relative(kernel_input.grad, eager_input.grad, _EAGER_KERNEL_GRAD_RTOL, "hidden states gradient")


@requires_tilelang
def test_sparse_attention_kernel_packed_matches_unpacked(monkeypatch):
    """The fused kernel path, end to end through one CSA layer, must respect documents.

    The same invariant its float32 neighbours assert, run in bfloat16 because that is the only
    dtype `dsv4_sparse_attn` accepts. Numerics belong to
    `test_dsv4_sparse_attn.py`, which compares the kernel against the float32 gather oracle on
    hand-built tensors; what is covered here is that the modeling code feeds the kernel inputs it
    can act on, and that nothing in `q`, the KV buffer or the indices carries the packed row's
    layout into a document's own answer.

    The call count is load-bearing, not decoration: `dsv4_sparse_attn` raises today rather than
    demoting a dtype it cannot run, but without counting the calls a regression that reintroduced
    a fallback would leave this test asserting a property of the gather reference instead.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(_KERNEL_DOC_LENS, torch.bfloat16, _flash_config())
    with torch.device("cuda"):
        hidden = torch.randn(1, sum(_KERNEL_DOC_LENS), _FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    packed_input, alone_input = hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, kv_buf, indices, sinks, scale):
        calls.append(q.shape[1])
        return real_kernel(q, kv_buf, indices, sinks, scale)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    packed_output, _ = module(packed_input, packed=packed)
    assert calls == [sum(_KERNEL_DOC_LENS)], f"the packed forward never reached the kernel, calls={calls}"
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(_KERNEL_DOC_LENS):
        span = _doc_slice(_KERNEL_DOC_LENS, index)
        alone_output, _ = module(
            alone_input[:, span], packed=_packed_context((length,), torch.bfloat16, _flash_config())
        )
        _assert_relative(packed_output[:, span], alone_output, _KERNEL_RTOL, f"document {index}")
        (alone_output * weight[:, span]).sum().backward()

    assert calls == [sum(_KERNEL_DOC_LENS), *_KERNEL_DOC_LENS], f"a forward never reached the kernel, calls={calls}"
    _compare_accumulated_grads(module, packed_grads, rtol=_KERNEL_GRAD_RTOL)
    _assert_relative(alone_input.grad, packed_input.grad, _KERNEL_GRAD_RTOL, "hidden states gradient")


@requires_tilelang
def test_sparse_attention_kernel_trains_every_parameter(monkeypatch):
    """Every parameter of a CSA layer that can train does, with the kernel in the path.

    `test_deepseek_v4_backward` makes this assertion through the assembled model, but only on the
    gather path: the kernel does not tile the toy shapes that file runs. This is the same assertion
    at module level and at the real Flash shapes, and it is not implied by its neighbour above, which
    compares two runs of the same path and would pass unchanged if both left a parameter at zero.

    The call count is load-bearing rather than decoration: `dsv4_sparse_attn` raises today
    instead of falling back, but a regression that reintroduced a fallback would leave this
    asserting a property of the gather reference.
    """
    module = flash_attention(_FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(_KERNEL_DOC_LENS, torch.bfloat16, _flash_config())
    with torch.device("cuda"):
        hidden_states = torch.randn(1, sum(_KERNEL_DOC_LENS), _FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    hidden_states.requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, kv_buf, indices, sinks, scale):
        calls.append(q.shape[1])
        return real_kernel(q, kv_buf, indices, sinks, scale)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    output, _ = module(hidden_states, packed=packed)
    assert calls == [sum(_KERNEL_DOC_LENS)], f"the forward never reached the kernel, calls={calls}"
    with torch.device("cuda"):
        weight = torch.randn_like(output)
    (output * weight).sum().backward()

    dead, unexpectedly_alive = [], []
    for name, param in module.named_parameters():
        has_grad = param.grad is not None and param.grad.norm().item() > 0
        # The same expectation `test_deepseek_v4_backward` and `_compare_accumulated_grads` carry:
        # the indexer reaches the loss only through integer top-k indices, so nothing
        # differentiates back into it.
        if ".indexer." in name:
            if has_grad:
                unexpectedly_alive.append(name)
        elif not has_grad:
            dead.append(name)

    assert not dead, f"Parameters with zero/no gradients: {dead}"
    assert not unexpectedly_alive, f"Lightning Indexer parameters received a gradient: {unexpectedly_alive}"
    assert hidden_states.grad is not None and hidden_states.grad.norm() > 0, (
        "the hidden states received no gradient, so nothing reached the layer's inputs"
    )


# The HCA layer of the Flash config, which nothing else here builds: every other test at these
# shapes takes `_FLASH_CSA_LAYER`. Documents are exact multiples of the HCA compress rate of 128,
# so both own whole entries and only the numbering, and with it the RoPE position, moves. Measured
# over 20 seeds the worst deviation is 3.0e-6 on the output and 4.6e-6 on a gradient, so
# `_PACKED_RTOL` holds here with room to spare, as it does for the gather test above.
_FLASH_HCA_LAYER = 1
_FLASH_HCA_DOCS = (256, 512)


def test_flash_hca_attention_packed_matches_unpacked():
    """An HCA layer at production shapes must answer each document as if it stood alone.

    HCA has no indexer and no sparse path: `DeepseekV4Attention` sends any layer that is not
    `compressed_sparse_attention` to the dense `_eager_with_entries` regardless of `attn_impl`,
    so there is no second implementation to compare against and this asserts self-consistency
    rather than equivalence. What it covers is the part the toy shapes cannot reach: a compress
    rate of 128 over 512 channels, where an entry pools 128 tokens and a document boundary that
    the compressor failed to respect would pull a whole other document into one entry.

    `test_attention_packed_matches_unpacked[hca]` asserts the same invariant at toy shapes and
    rate 8. This is the only test that instantiates the Flash config's HCA layer at all.
    """
    module = flash_attention(_FLASH_HCA_LAYER, dtype=torch.float32, attn_impl="eager")
    assert module.layer_type == "heavily_compressed_attention", (
        f"expected the Flash config's HCA layer, got {module.layer_type}"
    )
    seq_len = sum(_FLASH_HCA_DOCS)
    packed_input, alone_input = _flash_hidden_states(seq_len)
    packed = _packed_context(_FLASH_HCA_DOCS, torch.float32, _flash_config())

    q_residual = module.q_a_norm(module.q_a_proj(packed_input.detach()))
    _, block_bias = module.compressor(packed_input.detach(), q_residual, packed)
    assert (block_bias[:, :, _doc_slice(_FLASH_HCA_DOCS, 1)] == 0).any(), (
        "vacuous probe: no query of the second document reads a compressed entry"
    )

    packed_output, _ = module(packed_input, packed=packed)
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(_FLASH_HCA_DOCS):
        span = _doc_slice(_FLASH_HCA_DOCS, index)
        alone_output, _ = module(
            alone_input[:, span], packed=_packed_context((length,), torch.float32, _flash_config())
        )
        _assert_relative(packed_output[:, span], alone_output, _PACKED_RTOL, f"document {index}")
        (alone_output * weight[:, span]).sum().backward()

    _compare_accumulated_grads(module, packed_grads, rtol=_PACKED_GRAD_RTOL)
    _assert_relative(alone_input.grad, packed_input.grad, _PACKED_GRAD_RTOL, "hidden states gradient")
