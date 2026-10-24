import importlib.util
import math

import pytest
import torch
from torch import nn

from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4 import attention as dsv4_attention
from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Attention, PackedContext, max_sparse_attn_slots
from prime_rl.trainer.models.deepseek_v4.hyperconnections import DeepseekV4UnweightedRMSNorm
from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
from prime_rl.trainer.models.kernels.deepseek_v4 import IGNORE_SLOT
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_rope import dsv4_q_norm_rope, dsv4_rope
from prime_rl.trainer.models.layers.fp8_linear import (
    Float8BlockwiseGroupedLinear,
    replace_linear_with_fp8_blockwise_linear,
)
from prime_rl.trainer.models.layers.grouped_linear import GroupedLinear
from prime_rl.utils.utils import default_dtype
from tests.unit.train.models.deepseek_v4_markers import requires_datacenter_gpu

try:
    import tilelang
    from tilelang import language as T

    from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn import dsv4_sparse_attn
except ImportError:
    dsv4_sparse_attn = None  # type: ignore

pytestmark = [pytest.mark.gpu]

# Several tests below reach the modeling code without ever compiling a kernel, so this is a
# per-test skip rather than part of `pytestmark`.
requires_sparse_attn_kernel = pytest.mark.skipif(
    dsv4_attention.dsv4_sparse_attn is None,
    reason="the fused sparse attention kernel did not import; tilelang ships in the `gpu` extra, on linux only",
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


def _packed_context(
    doc_lens: tuple[int, ...],
    dtype: torch.dtype,
    config: DeepseekV4Config,
) -> PackedContext:
    """The context `DeepseekV4Model` would hand its attention layers for a row of `doc_lens`.

    A single-element `doc_lens` gives back the single-document context, which is what the unpacked
    half of a packing comparison runs at. `dtype` is the default dtype the rotary embedding is built
    under; the RoPE tables themselves are always fp32.
    """
    with torch.device("cuda"), default_dtype(dtype):
        rotary = DeepseekV4RotaryEmbedding(config)
    return PackedContext.build(
        rotary_emb=rotary,
        seq_lens=torch.tensor(doc_lens, device="cuda"),
        device=torch.device("cuda"),
    )


def _doc_slice(doc_lens: tuple[int, ...], index: int) -> slice:
    start = sum(doc_lens[:index])
    return slice(start, start + doc_lens[index])


def _take_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Detach whatever gradients have accumulated and clear them for the next run."""
    grads = {name: None if param.grad is None else param.grad.clone() for name, param in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    return grads


def _compare_accumulated_grads(module: nn.Module, expected: dict[str, torch.Tensor | None], rtol: float) -> None:
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


# The real DeepSeek V4-Flash attention shapes, written out as a literal so nothing here depends on a local HF
# cache. This file runs these shapes and no others. The MoE fields are shrunk to nothing, since
# `DeepseekV4Attention` reads none of them.
V4FLASH_MODEL = dict(
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
    layer_types=["compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"],
    num_hidden_layers=3,
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

# Shared by every test here: `DeepseekV4Attention` and `PackedContext.build` only read it.
V4FLASH_CONFIG = DeepseekV4Config(**V4FLASH_MODEL)

# Read off `V4FLASH_MODEL` so the hand-built tensors below describe the same CSA layer it does.
HEADS = V4FLASH_MODEL["num_attention_heads"]
DIM = V4FLASH_MODEL["head_dim"]
KV_GROUP = V4FLASH_MODEL["num_key_value_heads"]
TOPK = V4FLASH_MODEL["sliding_window"] + V4FLASH_MODEL["index_topk"]
MAX_SLOTS = max_sparse_attn_slots(V4FLASH_CONFIG)
SM_SCALE = DIM**-0.5

requires_cudnn_flashmla = pytest.mark.skipif(
    importlib.util.find_spec("flash_mla") is None
    or importlib.util.find_spec("cudnn") is None
    or not torch.cuda.is_available()
    or (torch.cuda.get_device_capability()[0] != 9 and torch.cuda.get_device_capability() not in ((10, 0), (10, 3))),
    reason="the cudnn_flashmla backend needs `flash-mla` (the `gpu` extra), the cuDNN frontend and an SM90, SM100 or SM103 GPU",
)

DSV4_BACKENDS = ["tilelang", pytest.param("cudnn_flashmla", marks=requires_cudnn_flashmla)]

if importlib.util.find_spec("flash_mla") is not None:
    # vLLM's bundled FlashMLA (`vllm._flashmla_C`) and the flash-mla wheel both define the GNU-unique
    # symbol `Fwd_Sm90_Impl::features`, so the copy loaded first in the process serves both. vLLM's
    # older list lacks TOPK_LENGTH, so load the wheel at collection, before any vLLM test loads its copy.
    import flash_mla  # noqa: F401

# What a real query with a short window or a saturated top-k looks like; a masked slot still
# costs a GEMM column.
MASKED_FRACTION = 0.25

# Compiled against eager, where the forward and the log-sum-exp are bit-identical. `dKV` is not,
# because `atomic_addx4` leaves its summation order to the scheduler, on both sides.
COMPILE_RTOL = 1e-2


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    actual, reference = actual.float(), reference.float()
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _build_indices(batch: int, seq_len: int, seq_len_kv: int, masked_fraction: float, topk: int = TOPK) -> torch.Tensor:
    """`(batch, seq_len, kv_group, topk)` int32 gather slots, a mix of valid picks and `IGNORE_SLOT` (-1).

    Valid KV positions are `[0, seq_len_kv)`; `IGNORE_SLOT` marks an absent key. The picks are drawn
    without replacement, since a real query never gathers the same key twice and a duplicate
    would take twice its share of the softmax on both sides of the comparison.
    """
    assert seq_len_kv >= topk, "not enough valid KV positions to fill the gather slots without repeats"
    picks = torch.rand(batch, seq_len, seq_len_kv, device="cuda").argsort(dim=-1)[..., :topk]
    masked = torch.rand(batch, seq_len, topk, device="cuda") < masked_fraction
    picks = torch.where(masked, torch.full_like(picks, IGNORE_SLOT), picks)
    return picks.to(torch.int32).unsqueeze(2).contiguous()


def _inputs(
    batch: int, seq_len: int, seq_len_kv: int, *, masked_fraction: float = MASKED_FRACTION
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`q`, `kv`, `indices`, `sinks` at the Flash shapes, detached values rather than leaves.

    `q` and `kv` are drawn at unit variance, so `q . k * sm_scale` has unit variance too and the
    unscaled sink logit, also unit variance, is a real competitor in the softmax rather than a
    term the dot products drown out.
    """
    torch.manual_seed(seq_len * 100003 + seq_len_kv)
    with torch.device("cuda"):
        q = torch.randn(batch, seq_len, HEADS, DIM, dtype=torch.bfloat16)
        kv = torch.randn(batch, seq_len_kv, KV_GROUP, DIM, dtype=torch.bfloat16)
        # A float32 sinks leaf, deliberately: the kernel casts `dsink` back to the leaf's dtype,
        # so a bfloat16 leaf would round both sides onto the same coarse grid and report a
        # deviation the rounding chose rather than one the kernel earned.
        sinks = torch.randn(HEADS, dtype=torch.float32)
    return q, kv, _build_indices(batch, seq_len, seq_len_kv, masked_fraction), sinks


def _leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.clone().requires_grad_(True) for tensor in tensors)


@pytest.mark.parametrize("backend", DSV4_BACKENDS)
@requires_sparse_attn_kernel
@requires_datacenter_gpu
def test_fully_masked_query_reads_as_zero_keys(backend):
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
    indices[:, 0] = IGNORE_SLOT  # the first query gathers nothing

    with torch.no_grad():
        out, lse = dsv4_sparse_attn(q, kv, indices, sinks, SM_SCALE, backend=backend, max_slots=MAX_SLOTS)

    assert torch.equal(out[:, 0], torch.zeros_like(out[:, 0])), "a fully masked query must emit exactly zero"
    expected_lse = sinks.float() * math.log2(math.e)
    assert torch.allclose(lse[:, 0], expected_lse.expand_as(lse[:, 0])), "lse must fall back to the sink term"
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()


@requires_sparse_attn_kernel
def test_tilelang_zero_fills_an_out_of_range_gather():
    """An index outside `[0, n_positions)` must read as zeros, not as whatever it points at.

    Both kernels index `KV` and `dKV` by a value read from `Indices` and never clamp it, so an
    `IGNORE_SLOT` (-1) slot is safe only because TileLang's `LegalizeSafeMemoryAccess` pass wraps
    every global access it cannot prove in range with `0 <= idx < extent`. That pass is on by
    default and has a single off switch, but nothing in TileLang documents it as a contract, and
    this project pins `tilelang>=0.1.8` with no upper bound, so an ordinary dependency bump could
    take it away.

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
    # Two in range, then the four ways out: `IGNORE_SLOT` (-1), the value this project marks an
    # absent key with, a far negative, one past the end, and far past it.
    slots = torch.tensor([0, 3, IGNORE_SLOT, -1000, n_positions, n_positions + 5], dtype=torch.int32, device="cuda")

    out = gather(n_positions, dim)(src, slots)

    assert torch.equal(out[0], src[0]) and torch.equal(out[1], src[3]), "an in-range index must gather its row"
    assert torch.equal(out[2:], torch.zeros_like(out[2:])), (
        "TileLang no longer zero-fills an out-of-range gather, so the kernels' unclamped `Indices` reads are unsafe"
    )


@pytest.mark.parametrize("backend", DSV4_BACKENDS)
@requires_sparse_attn_kernel
@requires_datacenter_gpu
def test_kernel_traces_under_torch_compile(backend):
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
        return dsv4_sparse_attn(q, kv, indices, sinks, SM_SCALE, backend=backend, max_slots=MAX_SLOTS)

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
        _assert_relative(compiled_leaf.grad, eager_leaf.grad, COMPILE_RTOL, label)


# Everything below reaches the kernel through the modeling code rather than on hand-built tensors, at the same
# Flash shapes. What it adds is the index construction: a CSA query's slot padding, its top-k saturation and
# the arithmetic mapping a compressed entry to a buffer position all live in `DeepseekV4Attention`, not in the
# kernel. `test_deepseek_v4.py` reaches the kernel through the same code, but only at toy shapes.

V4FLASH_CSA_LAYER, V4FLASH_HCA_LAYER, V4FLASH_SLIDING_LAYER = 0, 1, 2
V4FLASH_LAYERS = [V4FLASH_CSA_LAYER, V4FLASH_HCA_LAYER, V4FLASH_SLIDING_LAYER]
V4FLASH_LAYER_IDS = ["csa", "hca", "sliding"]
V4FLASH_COMPRESS_RATE = V4FLASH_MODEL["compress_rates"]["compressed_sparse_attention"]

# Document layouts for the sparse path, at `compress_rate = 4`. The first four leave every query
# short of `index_topk = 512` readable entries, so the `IGNORE_SLOT` (-1) padding of the pick slots carries
# the difference; `(2600,)` saturates the picks instead, which the toy shapes cannot express at
# all. `(3,)` compresses to no entries whatsoever, leaving the local window alone to answer.
V4FLASH_DOC_LENS = [(517, 1019), (3,), (300,), (3, 129, 1021), (2600,)]
V4FLASH_DOC_IDS = ["two-docs", "no-entries", "one-short-doc", "three-docs", "saturated-topk"]


def v4flash_attention(layer_idx: int, dtype: torch.dtype = torch.float32) -> nn.Module:
    """One attention layer at the real DeepSeek V4 Flash shapes, 126M parameters of it."""
    with torch.device("cuda"), default_dtype(dtype):
        module = DeepseekV4Attention(V4FLASH_CONFIG, layer_idx, DeepseekV4RotaryEmbedding(V4FLASH_CONFIG))
    _randomize(module)
    return module


def _v4flash_hidden_states(seq_len: int) -> torch.Tensor:
    """bfloat16 hidden states for one row of `seq_len` tokens.

    Batch 1: at these shapes the score tensors are the bulk of the memory and a second batch
    entry repeats the first without covering anything new.
    """
    with torch.device("cuda"):
        return torch.randn(1, seq_len, V4FLASH_MODEL["hidden_size"]).to(torch.bfloat16)


def _record_attention(monkeypatch) -> dict[str, torch.Tensor]:
    """Capture what the kernel is handed and what it returns, from real forwards.

    `dsv4_sparse_attn` is a module-level function the layer looks up by name, so patching it
    records the call the layer actually made rather than a reconstruction of it.
    """
    recorded: dict[str, torch.Tensor] = {}
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def kernel(q, kv_buf, indices, *args, **kwargs):
        recorded["kv_buf"], recorded["indices"] = kv_buf, indices
        recorded["kernel"] = real_kernel(q, kv_buf, indices, *args, **kwargs)
        return recorded["kernel"]

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", kernel)
    return recorded


def _entry_counts(doc_lens: tuple[int, ...], compress_rate: int) -> list[int]:
    return [length // compress_rate for length in doc_lens]


def _expected_picks(layer_type: str, doc_lens: tuple[int, ...]) -> int:
    """How many pick slots every query of this layer type gets, on top of its local window.

    CSA gets `index_topk` whatever the row holds, because `fp8_indexer` pads its output to the
    requested width and the surplus comes back as `IGNORE_SLOT`. HCA's tracks the longest document.
    """
    rate = V4FLASH_MODEL["compress_rates"].get(layer_type)
    if rate is None:
        return 0
    if layer_type == "heavily_compressed_attention":
        return max(_entry_counts(doc_lens, rate))
    return V4FLASH_MODEL["index_topk"]


@requires_sparse_attn_kernel
def test_deepseek_v4_refuses_a_shape_the_kernel_cannot_tile():
    """The constructor refuses a head count the kernels cannot tile, rather than the first forward."""
    config = DeepseekV4Config(**{**V4FLASH_MODEL, "num_attention_heads": 16})
    with pytest.raises(ValueError, match="cannot run the fused sparse-attention kernel"):
        DeepseekV4Attention(config, V4FLASH_CSA_LAYER, DeepseekV4RotaryEmbedding(config))


@requires_sparse_attn_kernel
@requires_datacenter_gpu
@pytest.mark.parametrize("doc_lens", V4FLASH_DOC_LENS, ids=V4FLASH_DOC_IDS)
@pytest.mark.parametrize("layer_idx", V4FLASH_LAYERS, ids=V4FLASH_LAYER_IDS)
def test_sparse_indices_are_in_range_and_never_repeat_a_key(doc_lens, layer_idx, monkeypatch):
    """Every gather slot addresses a real KV position, and no query counts a key twice.

    This is memory safety, not only correctness: the kernel's backward scatters through the same
    indices with an unguarded `atomic_add`, so an out-of-range slot corrupts whatever lies next to
    the buffer instead of raising, on every layer type that feeds it. A repeat is quieter but no
    better: the duplicated key takes twice its share of the softmax, silently reweighting the
    output. The `IGNORE_SLOT` (-1) padding is exempt from uniqueness, since padding every query out to a fixed
    slot count is exactly what it is for.
    """
    module = v4flash_attention(layer_idx, dtype=torch.bfloat16)
    layer_type = V4FLASH_MODEL["layer_types"][layer_idx]
    packed = _packed_context(doc_lens, torch.bfloat16, V4FLASH_CONFIG)
    hidden_states = _v4flash_hidden_states(sum(doc_lens))
    recorded = _record_attention(monkeypatch)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    indices, n_positions = recorded["indices"], recorded["kv_buf"].shape[1]
    # The exact width: the window plus the picks the row actually affords, and nothing else. A row
    # with fewer entries than the layer type's pick count gets a narrower slot count, not one padded
    # with `IGNORE_SLOT` (-1); the kernel pads to its own tile downstream of this.
    n_picks = _expected_picks(layer_type, doc_lens)
    n_slots = indices.shape[-1]
    assert n_slots == V4FLASH_MODEL["sliding_window"] + n_picks
    assert (indices >= IGNORE_SLOT).all(), "a gather slot addresses a KV position below the `IGNORE_SLOT` marker"
    assert (indices <= n_positions - 1).all(), "a gather slot addresses past the end of the KV buffer"

    slot_idx = indices[0, :, 0, :].long()
    # `IGNORE_SLOT` (-1) is counted in a throwaway column, since padding repeats it by design.
    safe = torch.where(slot_idx >= 0, slot_idx, n_positions)
    counts = torch.zeros((slot_idx.shape[0], n_positions + 1), dtype=torch.int32, device="cuda")
    counts.scatter_add_(1, safe, torch.ones_like(safe, dtype=torch.int32))
    assert (counts[:, :n_positions] <= 1).all(), "a query gathers the same key twice"


@requires_sparse_attn_kernel
@requires_datacenter_gpu
@pytest.mark.parametrize("doc_lens", V4FLASH_DOC_LENS, ids=V4FLASH_DOC_IDS)
def test_absent_slots_are_marked_negative_rather_than_pointed_at_a_pad_row(doc_lens, monkeypatch):
    """An unused gather slot must hold `IGNORE_SLOT` (-1), never a position that `kv_buf` actually has.

    This is the whole of the contract between `SparseAttnInputs.build` and the kernel, which masks
    on `Indices[...] < 0` and on nothing else. The design it replaced appended a zero row to
    `kv_buf` and pointed unused slots at that row's index instead, which looks equally valid: the
    pad index is in range and the key it names is zero either way.

    Under the kernel's masking the two are not equivalent at all. A non-negative pad index passes
    the `< 0` test, so the slot counts as a live key, and the zero row it names scores `q . 0 = 0`
    and enters the softmax with weight `exp(0)` rather than nothing. At these shapes the first
    query of a row carries 639 pad slots against 1 real key, so its denominator would be wrong by
    roughly three orders of magnitude, and nothing would raise.

    The neighbouring index test does fail if the pad row comes back, but on an incidental symptom:
    every pad slot naming one row reads as "a query gathers the same key twice". That does not name
    the cause, so this asserts the contract directly.
    """
    module = v4flash_attention(V4FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(doc_lens, torch.bfloat16, V4FLASH_CONFIG)
    hidden_states = _v4flash_hidden_states(sum(doc_lens))
    recorded = _record_attention(monkeypatch)

    with torch.no_grad():
        module(hidden_states, packed=packed)

    indices, kv_buf = recorded["indices"], recorded["kv_buf"]
    n_entries = sum(length // V4FLASH_COMPRESS_RATE for length in doc_lens)
    assert kv_buf.shape[1] == sum(doc_lens) + n_entries, (
        "kv_buf holds more than the token stream and its compressed entries, so `build` is padding "
        "it with rows that the `IGNORE_SLOT` (-1) marker makes unnecessary"
    )
    assert kv_buf[:, -1].abs().max() > 0, "the last row of kv_buf is zero, which is what a pad row looks like"

    # The first query of the row can read one key, its own token: the window clips to the document
    # and no complete compressed entry precedes it. Every other slot is padding, so this counts the
    # padding directly rather than inferring it.
    first_query = indices[0, 0, 0]
    assert (first_query >= 0).sum() == 1, (
        f"the first query holds {(first_query >= 0).sum().item()} non-negative slots against the 1 key it "
        "may read, so absent slots are addressing a KV position instead of holding `IGNORE_SLOT` (-1)"
    )
    assert (first_query[first_query < 0] == IGNORE_SLOT).all(), (
        "an absent slot is negative but is not the `IGNORE_SLOT` marker"
    )


# One CSA layer in bfloat16, so these bounds sit at the bfloat16 floor of two runs of the same kernel path
# rather than at a float32 one.
KERNEL_RTOL, KERNEL_GRAD_RTOL = 5e-3, 1e-2

# `compress_rate = 4` yields 129 + 254 = 383 compressed entries, under `index_topk = 512`, so
# every readable entry is picked and the indexer's ordering cannot differ packed from alone. A
# saturated layout would let a bfloat16 tie flip a pick and move the output for a reason that has
# nothing to do with document independence.
KERNEL_DOC_LENS = (517, 1019)


@pytest.mark.parametrize("backend", DSV4_BACKENDS)
@requires_sparse_attn_kernel
@requires_datacenter_gpu
def test_sparse_attention_kernel_packed_matches_unpacked(backend, monkeypatch):
    """The fused kernel path, end to end through one CSA layer, must respect documents.

    Run in bfloat16 because that is the only dtype `dsv4_sparse_attn` accepts. What is covered is
    that the modeling code feeds the kernel inputs it can act on, and that nothing in `q`, the KV
    buffer or the indices carries the packed row's layout into a document's own answer.

    The call count is load-bearing, not decoration: without it, a regression that routed around
    the kernel would leave this test passing without ever exercising the kernel.
    """
    module = v4flash_attention(V4FLASH_CSA_LAYER, dtype=torch.bfloat16)
    module.dsa_backend = backend
    packed = _packed_context(KERNEL_DOC_LENS, torch.bfloat16, V4FLASH_CONFIG)
    with torch.device("cuda"):
        hidden = torch.randn(1, sum(KERNEL_DOC_LENS), V4FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    packed_input, alone_input = hidden.clone().requires_grad_(True), hidden.clone().requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, *args, **kwargs):
        calls.append(q.shape[1])
        return real_kernel(q, *args, **kwargs)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    packed_output, _ = module(packed_input, packed=packed)
    assert calls == [sum(KERNEL_DOC_LENS)], f"the packed forward never reached the kernel, calls={calls}"
    with torch.device("cuda"):
        weight = torch.randn_like(packed_output)
    (packed_output * weight).sum().backward()
    packed_grads = _take_grads(module)

    for index, length in enumerate(KERNEL_DOC_LENS):
        span = _doc_slice(KERNEL_DOC_LENS, index)
        alone_output, _ = module(
            alone_input[:, span], packed=_packed_context((length,), torch.bfloat16, V4FLASH_CONFIG)
        )
        _assert_relative(packed_output[:, span], alone_output, KERNEL_RTOL, f"document {index}")
        (alone_output * weight[:, span]).sum().backward()

    assert calls == [sum(KERNEL_DOC_LENS), *KERNEL_DOC_LENS], f"a forward never reached the kernel, calls={calls}"
    _compare_accumulated_grads(module, packed_grads, rtol=KERNEL_GRAD_RTOL)
    _assert_relative(alone_input.grad, packed_input.grad, KERNEL_GRAD_RTOL, "hidden states gradient")


@requires_sparse_attn_kernel
@requires_datacenter_gpu
def test_sparse_attention_kernel_trains_every_parameter(monkeypatch):
    """Every parameter of a CSA layer that can train does, with the kernel in the path.

    `test_deepseek_v4_backward` makes this assertion through the assembled model at toy shapes.
    This is the same assertion at module level and at the real Flash shapes, and it is not implied
    by its neighbour above, which compares two runs of the same path and would pass unchanged if
    both left a parameter at zero.

    The call count is load-bearing rather than decoration: without it, a regression that routed
    around the kernel would leave this test passing without ever exercising the kernel.
    """
    module = v4flash_attention(V4FLASH_CSA_LAYER, dtype=torch.bfloat16)
    packed = _packed_context(KERNEL_DOC_LENS, torch.bfloat16, V4FLASH_CONFIG)
    with torch.device("cuda"):
        hidden_states = torch.randn(1, sum(KERNEL_DOC_LENS), V4FLASH_MODEL["hidden_size"], dtype=torch.bfloat16)
    hidden_states.requires_grad_(True)

    calls = []
    real_kernel = dsv4_attention.dsv4_sparse_attn

    def counting_kernel(q, *args, **kwargs):
        calls.append(q.shape[1])
        return real_kernel(q, *args, **kwargs)

    monkeypatch.setattr(dsv4_attention, "dsv4_sparse_attn", counting_kernel)

    output, _ = module(hidden_states, packed=packed)
    assert calls == [sum(KERNEL_DOC_LENS)], f"the forward never reached the kernel, calls={calls}"
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


ROPE_DIM = V4FLASH_MODEL["qk_rope_head_dim"]


def _rope_positions() -> torch.Tensor:
    """Three packed documents, the last starting deep into a long rollout: positions reset and reach ~128k."""
    return torch.cat([torch.arange(700), torch.arange(1500), torch.arange(129000, 130896)]).cuda()


def _rope_table(n_rows: int) -> torch.Tensor:
    """A plain fp32 `[cos | sin]` table in vLLM's layout, one entry per interleaved pair."""
    inv_freq = 1.0 / (V4FLASH_MODEL["rope_theta"] ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    freqs = torch.outer(torch.arange(n_rows, dtype=torch.float32), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).cuda()


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_dsv4_rope_matches_vllm_bit_for_bit(inverse):
    """The kernel and vLLM's in-place `rotary_embedding` op agree on every bit, nope channels included.

    Both read the same table, so this isolates the rotation: fp32 math on a bf16 input, one rounding
    on store, and the same multiply-add contraction.
    """
    vllm_ops = pytest.importorskip("vllm._custom_ops")
    positions = _rope_positions()
    cos_sin = _rope_table(int(positions.max()) + 1)
    heads, head_dim = V4FLASH_MODEL["num_attention_heads"], V4FLASH_MODEL["head_dim"]
    x = torch.randn(positions.numel(), heads, head_dim, device="cuda", dtype=torch.bfloat16)

    expected = x.clone()
    vllm_ops.rotary_embedding(
        positions, expected, None, head_dim, cos_sin, False, rope_dim_offset=head_dim - ROPE_DIM, inverse=inverse
    )
    actual = dsv4_rope(x.clone(), cos_sin, positions, inverse=inverse)

    assert torch.equal(actual, expected)


def test_q_norm_rope_matches_vllm_fused_prefill_kernel():
    """q matches vLLM's fused prefill q norm + RoPE to the last ulp, and the kv rotation bit for bit.

    vLLM rounds q to bf16 once after an fp32 RMSNorm and rotation, as `dsv4_q_norm_rope` does. The norm's
    reduction order differs, so a few elements per million land one ulp apart.
    """
    pytest.importorskip("vllm._custom_ops")
    positions = _rope_positions()
    cos_sin = _rope_table(int(positions.max()) + 1)
    n_tokens, heads, head_dim = positions.numel(), V4FLASH_MODEL["num_attention_heads"], V4FLASH_MODEL["head_dim"]
    eps, block_size = V4FLASH_MODEL["rms_norm_eps"], 64
    q = torch.randn(n_tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(n_tokens, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(n_tokens // block_size, block_size, head_dim, device="cuda", dtype=torch.bfloat16)

    expected_q = q.clone()
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        expected_q, kv, kv_cache, torch.arange(n_tokens, device="cuda"), positions, cos_sin, eps, block_size
    )
    rotated_q = dsv4_q_norm_rope(q, cos_sin, positions, eps)

    assert (rotated_q != expected_q).float().mean() < 1e-5
    _assert_relative(rotated_q, expected_q, torch.finfo(torch.bfloat16).eps, "q")
    rotated_kv = dsv4_rope(kv.clone().unsqueeze(1), cos_sin, positions)
    assert torch.equal(rotated_kv, kv_cache.view(n_tokens, 1, head_dim))


def test_q_norm_rope_matches_the_composed_norm_and_rotation():
    """Bit for bit with an fp32 RMSNorm, the rotation and one bf16 cast, eager and compiled.

    Gradients agree with the composed ops to bf16 precision rather than bit for bit: the composed
    backward carries an fp32 gradient through the rotation, where `dsv4_q_norm_rope` keeps it in bf16.
    """
    positions = _rope_positions()
    cos_sin = _rope_table(int(positions.max()) + 1)
    heads, head_dim = V4FLASH_MODEL["num_attention_heads"], V4FLASH_MODEL["head_dim"]
    eps = V4FLASH_MODEL["rms_norm_eps"]
    leaf = torch.randn(1, positions.numel(), heads, head_dim, device="cuda", dtype=torch.bfloat16)
    fused_leaf, compiled_leaf, composed_leaf = _leaves(leaf, leaf, leaf)
    weight = torch.randn_like(leaf, dtype=torch.float32)

    def composed(q: torch.Tensor) -> torch.Tensor:
        normed = DeepseekV4UnweightedRMSNorm(eps=eps, out_dtype=torch.float32)(q)
        return dsv4_rope(normed, cos_sin, positions).to(q.dtype)

    fused = dsv4_q_norm_rope(fused_leaf * 1, cos_sin, positions, eps)
    compiled = torch.compile(lambda q: dsv4_q_norm_rope(q * 1, cos_sin, positions, eps), fullgraph=True)(compiled_leaf)
    expected = composed(composed_leaf * 1)
    for out in (fused, compiled, expected):
        (out.float() * weight).sum().backward()

    assert fused.dtype == torch.bfloat16
    assert torch.equal(fused, expected)
    assert torch.equal(compiled, fused)
    assert torch.equal(compiled_leaf.grad, fused_leaf.grad)
    _assert_relative(fused_leaf.grad, composed_leaf.grad, torch.finfo(torch.bfloat16).eps, "q gradient")


O_GROUPS = V4FLASH_MODEL["o_groups"]
O_A_IN_FEATURES = V4FLASH_MODEL["num_attention_heads"] * V4FLASH_MODEL["head_dim"] // O_GROUPS
O_A_OUT_FEATURES = O_GROUPS * V4FLASH_MODEL["o_lora_rank"]
# A misaligned token count, so the weight gradient's 128-token padding is exercised.
O_A_TOKENS = 517
# Relative Frobenius error of an FP8 matmul against bf16 on Gaussian operands, with headroom.
O_A_FP8_RTOL = 0.06

requires_fp8_einsum = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="DeepGEMM's fp8_einsum runs on Hopper (SM90) and Blackwell (SM100) only",
)
requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="vLLM serves wo_a with fp32 scales on Hopper only, Blackwell packs them as INT32 ue8m0",
)


def _fp8_o_a_proj() -> nn.Module:
    """o_a_proj at Flash shapes, made FP8 by the same replacement pass the trainer runs.

    The weight is drawn at std 0.02, so every 128x128 block's amax sits far above 448 * 1e-4,
    where the trainer's scale floor and vLLM's amax floor would round differently.
    """
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = nn.ModuleDict({"o_a_proj": GroupedLinear(O_A_IN_FEATURES, O_A_OUT_FEATURES, O_GROUPS)})
    nn.init.normal_(model["o_a_proj"].weight, std=0.02)
    replace_linear_with_fp8_blockwise_linear(model, ignore_modules=[])
    assert type(model["o_a_proj"]) is Float8BlockwiseGroupedLinear
    return model["o_a_proj"]


def _o_a_proj_input() -> torch.Tensor:
    return torch.randn(1, O_A_TOKENS, O_GROUPS, O_A_IN_FEATURES, device="cuda", dtype=torch.bfloat16)


def _bf16_o_a_proj(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.einsum("btgk,gnk->btgn", x, weight.view(O_GROUPS, -1, O_A_IN_FEATURES))


def _assert_frobenius_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error <= rtol, f"{label}: relative error {error} exceeds {rtol}"


@requires_sm90
def test_fp8_o_a_proj_quantizes_and_multiplies_like_vllm_wo_a(monkeypatch):
    """The FP8 o_a_proj hands DeepGEMM bit for bit what vLLM's online-FP8 `wo_a` does, and gets its output.

    vLLM fuses the inverse RoPE into the input cast. Position 0 makes that rotation the identity, so
    its kernel quantizes exactly the tensor the trainer does.
    """
    deep_gemm = pytest.importorskip("deep_gemm")
    pytest.importorskip("vllm")
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    module = _fp8_o_a_proj()
    x = _o_a_proj_input()
    einsum_calls = []
    real_fp8_einsum = deep_gemm.fp8_einsum

    def recording_fp8_einsum(expr, a, b, d, recipe):
        einsum_calls.append((expr, a, b, recipe))
        real_fp8_einsum(expr, a, b, d, recipe=recipe)

    monkeypatch.setattr(deep_gemm, "fp8_einsum", recording_fp8_einsum)
    out = module(x)
    monkeypatch.undo()

    ((expr, (x_fp8, x_sf), (weight_fp8, weight_sf), recipe),) = einsum_calls
    heads, head_dim = V4FLASH_MODEL["num_attention_heads"], V4FLASH_MODEL["head_dim"]
    identity_rope = torch.cat([torch.ones(1, ROPE_DIM // 2), torch.zeros(1, ROPE_DIM // 2)], dim=-1).cuda()
    vllm_x_fp8, vllm_x_sf = fused_inv_rope_fp8_quant(
        x.view(O_A_TOKENS, heads, head_dim),
        torch.zeros(O_A_TOKENS, dtype=torch.int64, device="cuda"),
        identity_rope,
        n_groups=O_GROUPS,
        heads_per_group=heads // O_GROUPS,
        nope_dim=head_dim - ROPE_DIM,
        rope_dim=ROPE_DIM,
    )
    vllm_weight_fp8, vllm_weight_sf = per_block_cast_to_fp8(module.weight.detach(), [128, 128], use_ue8m0=False)
    vllm_weight_fp8 = vllm_weight_fp8.view(O_GROUPS, -1, O_A_IN_FEATURES)
    vllm_weight_sf = vllm_weight_sf.view(O_GROUPS, -1, vllm_weight_sf.size(-1))

    assert expr == "bhr,hdr->bhd"
    assert recipe == (1, 128, 128)
    assert torch.equal(x_fp8.view(torch.uint8), vllm_x_fp8.view(torch.uint8))
    assert torch.equal(x_sf, vllm_x_sf)
    assert torch.equal(weight_fp8.view(torch.uint8), vllm_weight_fp8.view(torch.uint8))
    assert torch.equal(weight_sf, vllm_weight_sf)

    vllm_out = torch.empty(O_A_TOKENS, O_GROUPS, O_A_OUT_FEATURES // O_GROUPS, device="cuda", dtype=torch.bfloat16)
    real_fp8_einsum(expr, (vllm_x_fp8, vllm_x_sf), (vllm_weight_fp8, vllm_weight_sf), vllm_out, recipe=recipe)
    assert out.shape == (1, O_A_TOKENS, O_GROUPS, O_A_OUT_FEATURES // O_GROUPS)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.view_as(vllm_out), vllm_out)
    _assert_frobenius_relative(out, _bf16_o_a_proj(x, module.weight.detach()), O_A_FP8_RTOL, "output")


@requires_fp8_einsum
def test_fp8_o_a_proj_gradients_match_bf16():
    module = _fp8_o_a_proj()
    weight = module.weight.detach().clone().requires_grad_(True)
    fp8_x, bf16_x = _leaves(_o_a_proj_input(), _o_a_proj_input())
    with torch.no_grad():
        bf16_x.copy_(fp8_x)
    upstream = torch.randn(1, O_A_TOKENS, O_GROUPS, O_A_OUT_FEATURES // O_GROUPS, device="cuda")

    (module(fp8_x).float() * upstream).sum().backward()
    (_bf16_o_a_proj(bf16_x, weight).float() * upstream).sum().backward()

    assert fp8_x.grad.dtype == torch.bfloat16 and module.weight.grad.dtype == torch.bfloat16
    _assert_frobenius_relative(fp8_x.grad, bf16_x.grad, O_A_FP8_RTOL, "dx")
    _assert_frobenius_relative(module.weight.grad, weight.grad, O_A_FP8_RTOL, "dweight")


@requires_fp8_einsum
def test_fp8_o_a_proj_traces_under_torch_compile():
    """`torch.compile(fullgraph=True)` through the FP8 o_a_proj forward and backward is bit for bit eager."""
    module = _fp8_o_a_proj()
    eager_x, compiled_x = _leaves(_o_a_proj_input(), _o_a_proj_input())
    with torch.no_grad():
        compiled_x.copy_(eager_x)
    upstream = torch.randn(1, O_A_TOKENS, O_GROUPS, O_A_OUT_FEATURES // O_GROUPS, device="cuda", dtype=torch.bfloat16)

    eager_out = module(eager_x)
    (eager_out * upstream).sum().backward()
    eager_weight_grad = module.weight.grad
    module.weight.grad = None
    compiled_out = torch.compile(module, fullgraph=True)(compiled_x)
    (compiled_out * upstream).sum().backward()

    assert compiled_out.shape == eager_out.shape and compiled_out.dtype == eager_out.dtype
    assert torch.equal(compiled_out, eager_out)
    assert torch.equal(compiled_x.grad, eager_x.grad)
    assert torch.equal(module.weight.grad, eager_weight_grad)
