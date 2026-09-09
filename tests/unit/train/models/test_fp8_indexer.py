from typing import NamedTuple

import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="the indexer kernel quantizes to Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]

HEADS, DIM, TOPK = 64, 128, 512

# How many queries and keys, the kernel's `S_q` and `S_k`. NUM_K >= 2 * TOPK, so the selection has
# to discard some, and neither axis is a whole number of the kernel's widest tile, so its
# ragged-tail masking runs.
NUM_Q, NUM_K = 3000, 1060

# A segment is a run of queries over a run of keys. One segment, then three uneven ones, where
# `ks` stops being zero. One segment holds more than TOPK keys, which is what it takes for a query
# to have a full ranking to disagree about.
SEGMENTS = [((NUM_Q, NUM_K),), ((700, 200), (500, 260), (1800, 600))]
SEGMENT_IDS = ["one-segment", "three-segments"]

# FP8 quantization can only move a pick across the top-k boundary, so the two selections agree on
# nearly every key. `p1` is the 1st percentile over queries: a whole tile of queries going wrong
# is more than one percent of them, so it shows up there rather than hiding in the mean.
AGREEMENT_MEAN, AGREEMENT_P1 = 0.99, 0.97

# The strongest picks sit nowhere near the boundary, so quantization does not reorder them out of
# the selection at all. These leave room for a handful of queries to lose one anyway.
STRONGEST_MEAN, STRONGEST_P1 = 0.999, 0.99

# How far from the cutoff a disagreement may sit, against the score tensor's own scale.
NEAR_TIE_RTOL = 5e-2


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)


def ranges(segments: tuple[tuple[int, int], ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """The half-open key range `[ks, ke)` each query may read, one segment after another.

    A segment's keys are contiguous, so its queries read a growing prefix of them: the first reads
    none of the segment at all, the last reads all of it.
    """
    assert sum(n_q for n_q, _ in segments) == NUM_Q
    assert sum(n_k for _, n_k in segments) == NUM_K
    ks, ke, base = [], [], 0
    for n_q, n_k in segments:
        ks.append(torch.full((n_q,), base, device="cuda"))
        ke.append(base + torch.arange(n_q, device="cuda") * n_k // (n_q - 1))
        base += n_k
    return torch.cat(ks), torch.cat(ke)


def eager_score_reference(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`score[t,s] = sum_h w[t,h] relu(sum_d q[t,h,d] k[s,d])`, in float32."""
    per_head = torch.relu(q.float() @ k.float().T)
    return torch.einsum("th,ths->ts", w.float(), per_head)


def readable_mask(ks: torch.Tensor, ke: torch.Tensor, num_k: int) -> torch.Tensor:
    """`(NUM_Q, NUM_K)` bool: key `s` is in query `t`'s range `[ks[t], ke[t])`."""
    key_idx = torch.arange(num_k, device=ks.device)
    return (key_idx[None, :] >= ks[:, None]) & (key_idx[None, :] < ke[:, None])


def isin_rows(query: torch.Tensor, table: torch.Tensor, sentinel_at: int) -> torch.Tensor:
    """Per row: whether `query[row, i]` appears in `table[row, :]`.

    Sorts `table` and binary-searches it rather than the simpler `query[..., None] ==
    table[..., None, :]` broadcast, whose `(rows, len(query), len(table))` intermediate these
    shapes cannot afford. A value `>= sentinel_at` never matches, since both sides pad unused
    slots with a sentinel and two sentinels are not agreement.
    """
    sorted_table, _ = table.sort(dim=-1)
    pos = torch.searchsorted(sorted_table, query.contiguous()).clamp(max=sorted_table.shape[-1] - 1)
    found = torch.gather(sorted_table, -1, pos) == query
    return found & (query < sentinel_at)


class Selections(NamedTuple):
    readable: torch.Tensor  # (NUM_Q, NUM_K) bool
    score_ref: torch.Tensor  # (NUM_Q, NUM_K) float32
    score_ref_masked: torch.Tensor  # score_ref with unreadable keys set to -inf
    ref_idx: torch.Tensor  # (NUM_Q, TOPK) int64, the float32 reference's picks
    ref_idx_sentinel: torch.Tensor  # ref_idx, with picks the query can't actually read replaced by NUM_K
    kernel_idx: torch.Tensor  # (NUM_Q, TOPK) int64, the FP8 kernel's picks, sentinel = NUM_K


def selections(segments: tuple[tuple[int, int], ...]) -> Selections:
    """Build q/k/w at real shapes and compare the kernel's picks against the float32 reference.

    Both sides read the same bfloat16 tensors, so the only difference the comparison measures is
    the kernel's own FP8 quantization.
    """
    device = torch.device("cuda")
    ks, ke = ranges(segments)
    readable = readable_mask(ks, ke, NUM_K)

    q_bf16 = torch.randn(NUM_Q, HEADS, DIM, device=device).bfloat16()
    k_bf16 = torch.randn(NUM_K, DIM, device=device).bfloat16()
    w_bf16 = torch.randn(NUM_Q, HEADS, device=device).bfloat16()

    score_ref = eager_score_reference(q_bf16, k_bf16, w_bf16)
    score_ref_masked = score_ref.masked_fill(~readable, float("-inf"))

    ref_idx = score_ref_masked.topk(TOPK, dim=-1).indices
    ref_valid = readable.gather(-1, ref_idx)
    ref_idx_sentinel = torch.where(ref_valid, ref_idx, torch.full_like(ref_idx, NUM_K))

    kernel_idx = fp8_indexer(q_bf16, k_bf16, w_bf16, ks.int(), ke.int(), TOPK).long()

    return Selections(readable, score_ref, score_ref_masked, ref_idx, ref_idx_sentinel, kernel_idx)


@pytest.mark.parametrize("segments", SEGMENTS, ids=SEGMENT_IDS)
def test_selection_agreement(segments):
    """Set agreement between the FP8 kernel and the float32 reference, mean and p1 over queries.

    The denominator is each query's own number of readable keys, capped at `TOPK`: an early query
    may have only a handful of keys to pick from, and comparing it against the fixed `TOPK` would
    read a mostly-empty selection as near-total disagreement.
    """
    s = selections(segments)

    matches = isin_rows(s.kernel_idx, s.ref_idx_sentinel, NUM_K)
    valid_count = (s.ref_idx_sentinel < NUM_K).sum(-1)
    agreement = matches.float().sum(-1) / valid_count.clamp(min=1)
    # A query with no readable keys has no selection to agree or disagree on.
    agreement = agreement[valid_count > 0]

    mean, p1 = agreement.mean().item(), agreement.quantile(0.01).item()
    assert mean > AGREEMENT_MEAN, f"mean set agreement {mean} below {AGREEMENT_MEAN}"
    assert p1 > AGREEMENT_P1, f"p1 set agreement {p1} below {AGREEMENT_P1}"


@pytest.mark.parametrize("segments", SEGMENTS, ids=SEGMENT_IDS)
@pytest.mark.parametrize("n_strongest", [8, 32])
def test_selection_agreement_restricted_to_strongest_picks(n_strongest, segments):
    """Agreement restricted to the strongest fp32-ranked picks, where a miss matters more."""
    s = selections(segments)

    strongest = s.ref_idx[:, :n_strongest]
    valid_strongest = s.readable.gather(-1, strongest)
    matches = isin_rows(strongest, s.kernel_idx, NUM_K) & valid_strongest

    denom = valid_strongest.sum(-1).clamp(min=1)
    agreement = matches.float().sum(-1) / denom
    agreement = agreement[valid_strongest.any(-1)]

    mean, p1 = agreement.mean().item(), agreement.quantile(0.01).item()
    assert mean > STRONGEST_MEAN, f"mean top-{n_strongest} agreement {mean} below {STRONGEST_MEAN}"
    assert p1 > STRONGEST_P1, f"p1 top-{n_strongest} agreement {p1} below {STRONGEST_P1}"


@pytest.mark.parametrize("segments", SEGMENTS, ids=SEGMENT_IDS)
def test_disagreements_are_near_ties(segments):
    """Where the kernel and the reference disagree, the key each picked scores almost the same.

    Both are scored by the float32 reference, so the disagreement sits at the top-k boundary, which
    is all quantization noise can move. Whether a boundary swap changes the layer's output is a
    separate question score proximity cannot settle.
    """
    s = selections(segments)
    scale = s.score_ref.abs().max().clamp(min=1.0)

    sorted_scores, _ = s.score_ref_masked.sort(dim=-1, descending=True)
    boundary_score = sorted_scores[:, TOPK - 1]
    next_score = sorted_scores[:, TOPK]  # NUM_K > TOPK, so there is always a next key.

    dropped_mask = s.readable.gather(-1, s.ref_idx)
    dropped_mask &= ~isin_rows(s.ref_idx, s.kernel_idx, NUM_K)
    dropped_scores = torch.gather(s.score_ref, -1, s.ref_idx).masked_fill(~dropped_mask, float("-inf"))
    max_dropped_score = dropped_scores.max(dim=-1).values

    kernel_valid = s.kernel_idx < NUM_K
    added_mask = kernel_valid & ~isin_rows(s.kernel_idx, s.ref_idx_sentinel, NUM_K)
    # The sentinel is out of range for this gather; clamped here and masked out immediately below.
    added_scores = torch.gather(s.score_ref, -1, s.kernel_idx.clamp(max=NUM_K - 1)).masked_fill(
        ~added_mask, float("inf")
    )
    min_added_score = added_scores.min(dim=-1).values

    # The near-cutoff comparison only means anything once a query has TOPK candidates to rank;
    # `test_selection_agreement` already covers the shorter ones, whose selection is every key.
    fully_ranked = torch.isfinite(boundary_score)
    assert fully_ranked.any(), "vacuous probe: no query reads TOPK keys, so no selection ranks any of them away"
    gap_drop = (max_dropped_score - boundary_score).clamp(min=0)[fully_ranked]
    gap_add = (next_score - min_added_score).clamp(min=0)[fully_ranked]
    gap = torch.maximum(gap_drop, gap_add)

    relative_gap = (gap / scale).max().item()
    assert relative_gap < NEAR_TIE_RTOL, (
        f"the largest disagreement sits {relative_gap} of score scale past the cutoff, over {NEAR_TIE_RTOL}"
    )
