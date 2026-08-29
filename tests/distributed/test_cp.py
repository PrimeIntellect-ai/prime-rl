# ruff: noqa: I001 (`prime_rl._compat` must run before the `ring_flash_attn` imports below)
import prime_rl._compat  # noqa: F401

import importlib.util

import pytest
import torch
import torch.distributed as dist

from tests.dtest import DTest

from ring_flash_attn.llama3_flash_attn_varlen import llama3_flash_attn_prepare_cu_seqlens

from prime_rl.trainer.models.layers.flash_varlen import sink_flash_attn_varlen_func
from prime_rl.trainer.models.layers.ring_attn import ring_flash_attn_varlen_func
from prime_rl.trainer.models.layers.ulysses_attn import ulysses_flash_attn_varlen_func

NHEADS = 4
NHEADS_K = 2
HEAD_DIM = 64
# Document boundary (at 48) deliberately doesn't land on the CP=2 shard boundary (at 64), so
# rank 0's shard spans into the second document and rank 1's shard sits fully inside it.
SEQLENS = (48, 80)

# Tolerances for gradients whose defining sum CP re-partitions across ranks: each rank rounds its
# partial to bf16 before the sum, where the single-GPU reference rounds the whole sum once. Measured
# at the shapes above (FA2, world_size=2), the absolute slack needed beyond rtol is 1.5e-2 for ring
# dv, 5.8e-3 for ring dk, 5.6e-3 and 3.9e-3 for the KV-replicated ulysses dv/dk, and 0 for ring
# dsink, whose error rtol absorbs on its own. Backstop at 2x the largest of those.
REDUCED_GRAD_RTOL = 2e-2
REDUCED_GRAD_ATOL = 3e-2

_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
_HAS_FLASH_ATTN_3 = importlib.util.find_spec("flash_attn_interface") is not None
_HAS_FLASH_ATTN_4 = importlib.util.find_spec("flash_attn.cute") is not None

# None when there's no CUDA device at all. DTest's own world_size check (`torch.cuda.device_count()
# < world_size`) already reports that case with a clearer message, so the hardware skipif conditions
# below stay False (never trigger) when there's no device, deferring to that check instead.
_SM_MAJOR, _SM_MINOR = torch.cuda.get_device_capability() if torch.cuda.is_available() else (None, None)
_NOT_HOPPER = _SM_MAJOR is not None and _SM_MAJOR != 9
_NOT_SM100 = _SM_MAJOR is not None and (_SM_MAJOR, _SM_MINOR) != (10, 0)


def _build_inputs(
    device: torch.device, nheads_k: int = NHEADS_K
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # Every rank computes this identically from the same seed.
    torch.manual_seed(42)
    total = sum(SEQLENS)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.int32, device=device)
    max_seqlen = max(SEQLENS)

    q = torch.randn(total, NHEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, nheads_k, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, nheads_k, HEAD_DIM, device=device, dtype=torch.bfloat16)

    return q, k, v, cu_seqlens, max_seqlen


def _build_sink(device: torch.device) -> torch.Tensor:
    """A per-head additive sink logit, standing in for GPT-OSS's `GptOssAttention.sinks`."""
    torch.manual_seed(7)
    return torch.randn(NHEADS, device=device, dtype=torch.bfloat16)


def _build_dout(q: torch.Tensor) -> torch.Tensor:
    # Seeded separately from `_build_inputs` so the upstream gradient doesn't depend on how many
    # draws happened before it.
    torch.manual_seed(1234)
    return torch.randn_like(q)


def _leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.clone().requires_grad_() for t in tensors)


def _shard(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return torch.chunk(t, world_size, dim=0)[rank]


def _reference_attn(
    flash_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    flash_attn_version: int,
) -> torch.Tensor:
    # FA4 omits max_seqlen and takes cu_seqlens as kwargs; FA2/FA3 take cu_seqlens/max_seqlen
    # positionally, twice (once for q, once for k).
    if flash_attn_version == 4:
        out = flash_fn(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, causal=True)
    else:
        out = flash_fn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
    return out[0] if isinstance(out, tuple) else out


def _assert_reference_is_meaningful(expected: torch.Tensor, what: str) -> None:
    # Two all-zero tensors compare equal, which would read as a pass while proving nothing.
    assert expected.abs().max().item() > 0, f"{what}: the reference is all zeros, the comparison is vacuous"


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Exact equality, for quantities CP computes with the reference's own arithmetic."""
    _assert_reference_is_meaningful(expected, what)
    assert torch.equal(actual, expected), f"{what} is not bitwise equal to the reference"


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str, *, rtol: float, atol: float) -> None:
    """Approximate equality, for quantities whose defining sum CP re-partitions across ranks."""
    _assert_reference_is_meaningful(expected, what)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"{what}: {m}")


def _all_reduced_grad(leaf: torch.Tensor) -> torch.Tensor:
    """This rank's gradient for a replicated parameter, summed over the CP group.

    The sink is replicated rather than sharded, so each rank only holds a partial: a sum over its
    own queries under ring, over its own head slice under ulysses. FSDP shards parameters over
    dp_shard_cp, which includes the CP dimension, so its gradient reduction already sums these.
    Emulate that here rather than making the attention wrapper communicate.
    """
    assert leaf.grad is not None, "the sink received no gradient"
    grad = leaf.grad.clone()
    dist.all_reduce(grad)
    return grad


class TestRingAttnCP(DTest):
    """Context-parallel attention must match the single-GPU reference bit for bit.

    These wrappers all-gather K/V and then make ordinary varlen calls, rather than merging
    per-step partial outputs and LSEs across ranks the way online ring attention does. Causality
    forces each rank's key range to start at its document's start, so every output row reduces
    over the same key blocks, in the same order, as the reference: identical arithmetic, hence
    identical bits. A mismatch therefore means the reduction pattern changed, not that rounding
    drifted. The same holds for dq on the backward pass. dk and dv are the exception: every rank
    computes a partial for every key, and the reduce-scatter that sums them rounds in bf16 twice
    where the reference rounds once.
    """

    default_world_size = 2

    def _check(self, flash_fn, flash_attn_version: int) -> None:
        q, k, v, cu_seqlens, max_seqlen = _build_inputs(self.device)
        dout = _build_dout(q)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens, causal=True, rank=self.rank, world_size=self.world_size
        )

        shard = (self.rank, self.world_size)

        ref_q, ref_k, ref_v = _leaves(q, k, v)
        ref_out = _reference_attn(flash_fn, ref_q, ref_k, ref_v, cu_seqlens, max_seqlen, flash_attn_version)
        ref_out.backward(dout)

        cp_q, cp_k, cp_v = _leaves(*(_shard(t, *shard) for t in (q, k, v)))
        out = ring_flash_attn_varlen_func(
            cp_q,
            cp_k,
            cp_v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
            causal=True,
            heads_k_stride=1,
            group=dist.group.WORLD,
            flash_attn_version=flash_attn_version,
        )
        out.backward(_shard(dout, *shard))

        _assert_equal(out, _shard(ref_out, *shard), "out")
        _assert_equal(cp_q.grad, _shard(ref_q.grad, *shard), "dq")
        # dk/dv are the only reduced quantities: every rank holds every key, so each contributes a
        # partial that a reduce-scatter sums.
        _assert_close(cp_k.grad, _shard(ref_k.grad, *shard), "dk", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)
        _assert_close(cp_v.grad, _shard(ref_v.grad, *shard), "dv", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=2)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_correctness(self) -> None:
        from flash_attn_interface import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=3)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_correctness(self) -> None:
        from flash_attn.cute import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=4)

    def _check_sink(self, flash_attn_version: int) -> None:
        q, k, v, cu_seqlens, max_seqlen = _build_inputs(self.device)
        sink = _build_sink(self.device)
        dout = _build_dout(q)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens, causal=True, rank=self.rank, world_size=self.world_size
        )

        shard = (self.rank, self.world_size)

        ref_q, ref_k, ref_v, ref_sink = _leaves(q, k, v, sink)
        ref_out = sink_flash_attn_varlen_func(
            ref_q,
            ref_k,
            ref_v,
            ref_sink,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            flash_attn_version=flash_attn_version,
        )
        ref_out.backward(dout)

        cp_q, cp_k, cp_v, cp_sink = _leaves(*(_shard(t, *shard) for t in (q, k, v)), sink)
        out = ring_flash_attn_varlen_func(
            cp_q,
            cp_k,
            cp_v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
            causal=True,
            heads_k_stride=1,
            group=dist.group.WORLD,
            flash_attn_version=flash_attn_version,
            sink=cp_sink,
        )
        out.backward(_shard(dout, *shard))

        _assert_equal(out, _shard(ref_out, *shard), "out")
        _assert_equal(cp_q.grad, _shard(ref_q.grad, *shard), "dq")
        # dk/dv are the only reduced quantities: every rank holds every key, so each contributes a
        # partial that a reduce-scatter sums.
        _assert_close(cp_k.grad, _shard(ref_k.grad, *shard), "dk", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)
        _assert_close(cp_v.grad, _shard(ref_v.grad, *shard), "dv", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)
        # dsink sums over this rank's queries, so it is reduced for the same reason dk/dv are.
        dsink = _all_reduced_grad(cp_sink)
        _assert_close(dsink, ref_sink.grad, "dsink", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_sink_correctness(self) -> None:
        self._check_sink(flash_attn_version=2)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_sink_correctness(self) -> None:
        self._check_sink(flash_attn_version=3)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_sink_correctness(self) -> None:
        self._check_sink(flash_attn_version=4)


class TestUlyssesCP(DTest):
    """Ulysses CP must match the single-GPU reference bit for bit.

    The all-to-all only moves tensor data between ranks (sequence-sharded <-> head-sharded); it
    never sums or rescales anything. Flash attention treats heads independently, so running it on
    a subset of heads over the full sequence produces exactly the same per-head arithmetic as
    running it on all heads together. No rounding-order difference is introduced anywhere in the
    pipeline, so, like ring attention, a mismatch means the reshaping is wrong, not that rounding
    drifted. Unlike ring attention that extends to the whole backward pass: nothing is summed across
    ranks, so dq, dk, dv and dsink are all bitwise equal too. The one exception is GQA with fewer KV
    heads than the CP size, where `_replicate_kv_heads` copies a KV head to several ranks and its
    backward sums their partials, putting dk and dv back in the same position as ring attention's.
    """

    default_world_size = 2

    def _run(
        self, flash_fn, flash_attn_version: int, nheads_k: int = NHEADS_K
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Run the reference and the CP path, backward included, and hand back what to compare."""
        q, k, v, cu_seqlens, max_seqlen = _build_inputs(self.device, nheads_k=nheads_k)
        dout = _build_dout(q)
        shard = (self.rank, self.world_size)

        ref_leaves = _leaves(q, k, v)
        ref_out = _reference_attn(flash_fn, *ref_leaves, cu_seqlens, max_seqlen, flash_attn_version)
        ref_out.backward(dout)

        cp_leaves = _leaves(*(_shard(t, *shard) for t in (q, k, v)))
        out = ulysses_flash_attn_varlen_func(
            flash_fn,
            *cp_leaves,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=flash_attn_version,
        )
        out.backward(_shard(dout, *shard))
        return out, ref_out, ref_leaves, cp_leaves

    def _check(self, flash_fn, flash_attn_version: int) -> None:
        shard = (self.rank, self.world_size)
        out, ref_out, (ref_q, ref_k, ref_v), (cp_q, cp_k, cp_v) = self._run(flash_fn, flash_attn_version)

        # Nothing is summed across ranks, so every quantity is the reference's own arithmetic.
        _assert_equal(out, _shard(ref_out, *shard), "out")
        _assert_equal(cp_q.grad, _shard(ref_q.grad, *shard), "dq")
        _assert_equal(cp_k.grad, _shard(ref_k.grad, *shard), "dk")
        _assert_equal(cp_v.grad, _shard(ref_v.grad, *shard), "dv")

    def _check_replicated_kv(self, flash_fn, flash_attn_version: int) -> None:
        # nheads_k=1 < cp_size=2 forces _replicate_kv_heads, unlike the other tests
        # (nheads_k=2 == cp_size never triggers it).
        shard = (self.rank, self.world_size)
        out, ref_out, (ref_q, ref_k, ref_v), (cp_q, cp_k, cp_v) = self._run(flash_fn, flash_attn_version, nheads_k=1)

        _assert_equal(out, _shard(ref_out, *shard), "out")
        _assert_equal(cp_q.grad, _shard(ref_q.grad, *shard), "dq")
        # Each replica of the single KV head produces a partial dk/dv that `_replicate_kv_heads`'
        # backward sums, so unlike the plain case these two are reduced.
        _assert_close(cp_k.grad, _shard(ref_k.grad, *shard), "dk", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)
        _assert_close(cp_v.grad, _shard(ref_v.grad, *shard), "dv", rtol=REDUCED_GRAD_RTOL, atol=REDUCED_GRAD_ATOL)

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=2)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_correctness(self) -> None:
        from flash_attn_interface import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=3)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_correctness(self) -> None:
        from flash_attn.cute import flash_attn_varlen_func

        self._check(flash_attn_varlen_func, flash_attn_version=4)

    def _check_sink(self, flash_fn, flash_attn_version: int) -> None:
        q, k, v, cu_seqlens, max_seqlen = _build_inputs(self.device)
        sink = _build_sink(self.device)
        dout = _build_dout(q)

        shard = (self.rank, self.world_size)

        ref_q, ref_k, ref_v, ref_sink = _leaves(q, k, v, sink)
        ref_out = sink_flash_attn_varlen_func(
            ref_q,
            ref_k,
            ref_v,
            ref_sink,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            flash_attn_version=flash_attn_version,
        )
        ref_out.backward(dout)

        cp_q, cp_k, cp_v, cp_sink = _leaves(*(_shard(t, *shard) for t in (q, k, v)), sink)
        out = ulysses_flash_attn_varlen_func(
            flash_fn,
            cp_q,
            cp_k,
            cp_v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=flash_attn_version,
            sink=cp_sink,
        )
        out.backward(_shard(dout, *shard))

        # Nothing is summed across ranks, so every quantity is the reference's own arithmetic.
        # dsink included: this rank's partial is zero outside its own heads, and adding zero is
        # exact, so the all-reduce that reassembles it changes no bits.
        _assert_equal(out, _shard(ref_out, *shard), "out")
        _assert_equal(cp_q.grad, _shard(ref_q.grad, *shard), "dq")
        _assert_equal(cp_k.grad, _shard(ref_k.grad, *shard), "dk")
        _assert_equal(cp_v.grad, _shard(ref_v.grad, *shard), "dv")
        _assert_equal(_all_reduced_grad(cp_sink), ref_sink.grad, "dsink")

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_sink_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        self._check_sink(flash_attn_varlen_func, flash_attn_version=2)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_sink_correctness(self) -> None:
        from flash_attn_interface import flash_attn_varlen_func

        self._check_sink(flash_attn_varlen_func, flash_attn_version=3)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_sink_correctness(self) -> None:
        from flash_attn.cute import flash_attn_varlen_func

        self._check_sink(flash_attn_varlen_func, flash_attn_version=4)

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_gqa_kv_head_replication_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        self._check_replicated_kv(flash_attn_varlen_func, flash_attn_version=2)
