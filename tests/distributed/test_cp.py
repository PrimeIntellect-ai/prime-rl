# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

import importlib.util

import pytest
import torch
import torch.distributed as dist

from tests.dtest import DTest

from ring_flash_attn.llama3_flash_attn_varlen import llama3_flash_attn_prepare_cu_seqlens, llama3_flash_attn_varlen_func

from prime_rl.trainer.models.layers.ring_attn import (
    _fa3_varlen_forward,
    _fa4_varlen_forward,
    ring_fa3_varlen_func,
    ring_fa4_varlen_func,
)
from prime_rl.trainer.models.layers.ulysses_attn import ulysses_flash_attn_varlen_func

NHEADS = 4
NHEADS_K = 2
HEAD_DIM = 64
# Document boundary (at 48) deliberately doesn't land on the CP=2 shard boundary (at 64), so
# rank 0's shard spans into the second document and rank 1's shard sits fully inside it.
SEQLENS = (48, 80)

_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None
_HAS_FLASH_ATTN_3 = importlib.util.find_spec("flash_attn_interface") is not None
_HAS_FLASH_ATTN_4 = importlib.util.find_spec("flash_attn.cute") is not None

# None when there's no CUDA device at all — DTest's own world_size check (`torch.cuda.device_count()
# < world_size`) already reports that case with a clearer message, so the hardware skipif conditions
# below stay False (never trigger) when there's no device, deferring to that check instead.
_SM_MAJOR, _SM_MINOR = torch.cuda.get_device_capability() if torch.cuda.is_available() else (None, None)
_NOT_HOPPER = _SM_MAJOR is not None and _SM_MAJOR != 9
_NOT_SM100 = _SM_MAJOR is not None and (_SM_MAJOR, _SM_MINOR) != (10, 0)


def _build_inputs(
    device: torch.device, nheads_k: int = NHEADS_K
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    # Every rank computes this identically from the same seed.
    torch.manual_seed(42)
    total = sum(SEQLENS)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.int32, device=device)
    max_seqlen = max(SEQLENS)
    softmax_scale = HEAD_DIM**-0.5

    q = torch.randn(total, NHEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, nheads_k, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, nheads_k, HEAD_DIM, device=device, dtype=torch.bfloat16)

    return q, k, v, cu_seqlens, max_seqlen, softmax_scale


def _assert_shard_matches_reference(ref_out: torch.Tensor, out: torch.Tensor, rank: int, world_size: int) -> None:
    expected = torch.chunk(ref_out, world_size, dim=0)[rank]
    assert torch.equal(out, expected)


class TestRingAttnCP(DTest):
    """Context-parallel attention must match the single-GPU reference bit for bit.

    These wrappers all-gather K/V and then make ordinary varlen calls, rather than merging
    per-step partial outputs and LSEs across ranks the way online ring attention does. Causality
    forces each rank's key range to start at its document's start, so every output row reduces
    over the same key blocks, in the same order, as the reference: identical arithmetic, hence
    identical bits. A mismatch therefore means the reduction pattern changed, not that rounding
    drifted.
    """

    default_world_size = 2

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        q, k, v, cu_seqlens, max_seqlen, _softmax_scale = _build_inputs(self.device)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens, causal=True, rank=self.rank, world_size=self.world_size
        )
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)

        out = llama3_flash_attn_varlen_func(
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride=1,
            local_k_slice=local_k_slice,
            causal=True,
            group=dist.group.WORLD,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_correctness(self) -> None:
        q, k, v, cu_seqlens, max_seqlen, softmax_scale = _build_inputs(self.device)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens, causal=True, rank=self.rank, world_size=self.world_size
        )
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out, _ = _fa3_varlen_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale, causal=True
        )

        out = ring_fa3_varlen_func(
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
            causal=True,
            heads_k_stride=1,
            group=dist.group.WORLD,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_correctness(self) -> None:
        q, k, v, cu_seqlens, max_seqlen, softmax_scale = _build_inputs(self.device)
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice = llama3_flash_attn_prepare_cu_seqlens(
            cu_seqlens, causal=True, rank=self.rank, world_size=self.world_size
        )
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out, _ = _fa4_varlen_forward(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale, causal=True
        )

        out = ring_fa4_varlen_func(
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            local_k_slice,
            causal=True,
            heads_k_stride=1,
            group=dist.group.WORLD,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)


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


class TestUlyssesCP(DTest):
    """Ulysses CP must match the single-GPU reference bit for bit.

    The all-to-all only moves tensor data between ranks (sequence-sharded <-> head-sharded); it
    never sums or rescales anything. Flash attention treats heads independently, so running it on
    a subset of heads over the full sequence produces exactly the same per-head arithmetic as
    running it on all heads together. No rounding-order difference is introduced anywhere in the
    pipeline, so, like ring attention, a mismatch means the reshaping is wrong, not that rounding
    drifted.
    """

    default_world_size = 2

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_correctness(self) -> None:
        from flash_attn import flash_attn_varlen_func

        q, k, v, cu_seqlens, max_seqlen, _softmax_scale = _build_inputs(self.device, nheads_k=2)
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out = _reference_attn(flash_attn_varlen_func, q, k, v, cu_seqlens, max_seqlen, flash_attn_version=2)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=2,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)

    @pytest.mark.skipif(_NOT_HOPPER, reason=f"FA3 requires Hopper (SM90); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_3, reason="flash_attn_interface not installed")
    def test_fa3_correctness(self) -> None:
        from flash_attn_interface import flash_attn_varlen_func

        q, k, v, cu_seqlens, max_seqlen, _softmax_scale = _build_inputs(self.device, nheads_k=2)
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out = _reference_attn(flash_attn_varlen_func, q, k, v, cu_seqlens, max_seqlen, flash_attn_version=3)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=3,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)

    @pytest.mark.skipif(_NOT_SM100, reason=f"FA4 requires SM100 (datacenter Blackwell); found SM{_SM_MAJOR}{_SM_MINOR}")
    @pytest.mark.skipif(not _HAS_FLASH_ATTN_4, reason="flash_attn.cute not installed")
    def test_fa4_correctness(self) -> None:
        from flash_attn.cute import flash_attn_varlen_func

        q, k, v, cu_seqlens, max_seqlen, _softmax_scale = _build_inputs(self.device, nheads_k=2)
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out = _reference_attn(flash_attn_varlen_func, q, k, v, cu_seqlens, max_seqlen, flash_attn_version=4)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=4,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)

    @pytest.mark.skipif(not _HAS_FLASH_ATTN, reason="flash_attn not installed")
    def test_fa2_gqa_kv_head_replication_correctness(self) -> None:
        # nheads_k=1 < cp_size=2 forces _replicate_kv_heads, unlike the other tests above
        # (nheads_k=2 == cp_size never triggers it).
        from flash_attn import flash_attn_varlen_func

        q, k, v, cu_seqlens, max_seqlen, _softmax_scale = _build_inputs(self.device, nheads_k=1)
        q_shard, k_shard, v_shard = (torch.chunk(t, self.world_size, dim=0)[self.rank] for t in (q, k, v))

        ref_out = _reference_attn(flash_attn_varlen_func, q, k, v, cu_seqlens, max_seqlen, flash_attn_version=2)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            q_shard,
            k_shard,
            v_shard,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=dist.group.WORLD,
            cp_size=self.world_size,
            flash_attn_version=2,
        )

        _assert_shard_matches_reference(ref_out, out, self.rank, self.world_size)
