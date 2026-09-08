from typing import NamedTuple

import pytest
import torch

from prime_rl.trainer.models.deepseek_v4.eager_reference import indexer_scores
from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer

# Plain tensors rather than a model: the modeling path around this kernel is covered in
# test_deepseek_v4_kernels.py, and the kernel also serves GLM DSA, which has no DeepSeek V4 in it.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="the indexer kernel quantizes to Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]

# The real DeepSeek V4 Flash Lightning Indexer shapes.
HEADS, DIM, TOPK, COMPRESS_RATE = 64, 128, 512, 4

# Twice as many entries as TOPK, so the selection has to actually discard some of them.
SEQ_LEN = 2 * TOPK * COMPRESS_RATE
N_ENTRIES = SEQ_LEN // COMPRESS_RATE

# One document, then the same tokens split into three of uneven length, where `ks` stops being
# zero. Every length is a multiple of COMPRESS_RATE, so both layouts yield exactly N_ENTRIES
# entries and the two runs differ only in how the range is laid out.
DOC_LENS = [(SEQ_LEN,), (1500, 500, 2096)]
DOC_IDS = ["one-doc", "three-docs"]

# FP8 quantization can only move a pick across the top-k boundary, so the two selections agree on
# nearly every entry; the floor allows a query whose scores cluster at that boundary.
AGREEMENT_MEAN, AGREEMENT_FLOOR = 0.95, 0.5

# How far from the cutoff a disagreement may sit, against the score tensor's own scale.
NEAR_TIE_RTOL = 5e-2


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


class _Layout(NamedTuple):
    tok_doc: torch.Tensor  # (seq_len,) int64 - which document each token belongs to
    tok_pos: torch.Tensor  # (seq_len,) int64 - position of each token within its own document
    entry_doc: torch.Tensor  # (n_entries,) int64 - which document each entry belongs to
    entry_local: torch.Tensor  # (n_entries,) int64 - entry index within its own document
    first_entry: torch.Tensor  # (n_docs,) int64 - global index of each document's first entry


def _layout(doc_lens: tuple[int, ...], device: torch.device) -> _Layout:
    """Document bookkeeping for a packed row, which a single document is the one-element case of."""
    docs = torch.arange(len(doc_lens), device=device)
    lengths = torch.tensor(doc_lens, device=device)
    counts = lengths // COMPRESS_RATE
    return _Layout(
        tok_doc=torch.repeat_interleave(docs, lengths),
        tok_pos=torch.cat([torch.arange(length, device=device) for length in doc_lens]),
        entry_doc=torch.repeat_interleave(docs, counts),
        entry_local=torch.cat([torch.arange(int(count), device=device) for count in counts]),
        first_entry=counts.cumsum(0) - counts,
    )


def _causal_range(layout: _Layout) -> tuple[torch.Tensor, torch.Tensor]:
    """The half-open entry range `[ks, ke)` each query may read, in the global entry axis.

    A document's entries are numbered consecutively, so the readable set is contiguous: it starts
    at that document's first entry and runs as far as the query's position has closed windows.
    """
    ks = layout.first_entry[layout.tok_doc]
    return ks, ks + (layout.tok_pos + 1) // COMPRESS_RATE


def _readable_mask(ks: torch.Tensor, ke: torch.Tensor, n_entries: int) -> torch.Tensor:
    """`(seq_len, n_entries)` bool: entry `e` is in query `t`'s range `[ks[t], ke[t])`."""
    entry_idx = torch.arange(n_entries, device=ks.device)
    return (entry_idx[None, :] >= ks[:, None]) & (entry_idx[None, :] < ke[:, None])


def _isin_rows(query: torch.Tensor, table: torch.Tensor, sentinel_at: int) -> torch.Tensor:
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


class _Selections(NamedTuple):
    # The raw kernel marks "no pick" with `N_ENTRIES`, not with the layer's `IGNORE_SLOT`; the
    # conversion between the two lives in `DeepseekV4Indexer.forward`, downstream of this file.
    readable: torch.Tensor  # (SEQ_LEN, N_ENTRIES) bool
    score_ref: torch.Tensor  # (SEQ_LEN, N_ENTRIES) float32
    score_ref_masked: torch.Tensor  # score_ref with unreadable entries set to -inf
    ref_idx: torch.Tensor  # (SEQ_LEN, TOPK) int64, the float32 reference's picks
    ref_idx_sentinel: torch.Tensor  # ref_idx, with picks the query can't actually read replaced by N_ENTRIES
    kernel_idx: torch.Tensor  # (SEQ_LEN, TOPK) int64, the FP8 kernel's picks, sentinel = N_ENTRIES


def _selections(doc_lens: tuple[int, ...]) -> _Selections:
    """Build q/k/w at real shapes and compare the kernel's picks against the float32 reference.

    Both sides read the same bfloat16 tensors, so the only difference the comparison measures is
    the kernel's own FP8 quantization.
    """
    device = torch.device("cuda")
    ks, ke = _causal_range(_layout(doc_lens, device))
    readable = _readable_mask(ks, ke, N_ENTRIES)

    q_bf16 = torch.randn(SEQ_LEN, HEADS, DIM, device=device).bfloat16()
    k_bf16 = torch.randn(N_ENTRIES, DIM, device=device).bfloat16()
    w_bf16 = torch.randn(SEQ_LEN, HEADS, device=device).bfloat16()

    score_ref = indexer_scores(q_bf16.unsqueeze(0), k_bf16.unsqueeze(0), w_bf16.unsqueeze(0))[0]
    score_ref_masked = score_ref.masked_fill(~readable, float("-inf"))

    ref_idx = score_ref_masked.topk(TOPK, dim=-1).indices
    ref_valid = readable.gather(-1, ref_idx)
    ref_idx_sentinel = torch.where(ref_valid, ref_idx, torch.full_like(ref_idx, N_ENTRIES))

    kernel_idx = fp8_indexer(q_bf16, k_bf16, w_bf16, ks.int(), ke.int(), TOPK).long()

    return _Selections(readable, score_ref, score_ref_masked, ref_idx, ref_idx_sentinel, kernel_idx)


@pytest.mark.parametrize("doc_lens", DOC_LENS, ids=DOC_IDS)
def test_causal_range_matches_entry_span(doc_lens):
    """`[ks, ke)` must equal a causality check derived independently, from each entry's token span.

    On a packed row the two coordinate systems disagree for every document after the first, which
    is where a range-based mask would go wrong.
    """
    layout = _layout(doc_lens, torch.device("cuda"))
    ks, ke = _causal_range(layout)

    entry_last_token = (layout.entry_local + 1) * COMPRESS_RATE - 1
    same_document = layout.entry_doc[None, :] == layout.tok_doc[:, None]
    expected = same_document & (entry_last_token[None, :] <= layout.tok_pos[:, None])

    assert torch.equal(_readable_mask(ks, ke, N_ENTRIES), expected), (
        "the readable range disagrees with the entries' token spans"
    )


@pytest.mark.parametrize("doc_lens", DOC_LENS, ids=DOC_IDS)
def test_selection_agreement(doc_lens):
    """Set agreement between the FP8 kernel and the float32 reference, mean/p1/min over queries.

    The denominator is each query's own number of readable entries, capped at `TOPK`: an early
    query may have only a handful of entries to pick from, and comparing it against the fixed
    `TOPK` would read a mostly-empty selection as near-total disagreement.
    """
    s = _selections(doc_lens)

    matches = _isin_rows(s.kernel_idx, s.ref_idx_sentinel, N_ENTRIES)
    valid_count = (s.ref_idx_sentinel < N_ENTRIES).sum(-1)
    agreement = matches.float().sum(-1) / valid_count.clamp(min=1)
    # A query with no readable entries has no selection to agree or disagree on.
    agreement = agreement[valid_count > 0]

    mean, p1, minimum = agreement.mean().item(), agreement.quantile(0.01).item(), agreement.min().item()
    assert mean > AGREEMENT_MEAN, f"mean set agreement {mean} below {AGREEMENT_MEAN}"
    assert p1 > AGREEMENT_FLOOR, f"p1 set agreement {p1} below {AGREEMENT_FLOOR}"
    assert minimum > AGREEMENT_FLOOR, f"min set agreement {minimum} below {AGREEMENT_FLOOR}"


@pytest.mark.parametrize("doc_lens", DOC_LENS, ids=DOC_IDS)
@pytest.mark.parametrize("restrict_to", [8, 32])
def test_selection_agreement_restricted_to_strongest_picks(restrict_to, doc_lens):
    """Agreement restricted to the strongest fp32-ranked picks, where a miss matters more."""
    s = _selections(doc_lens)

    strongest = s.ref_idx[:, :restrict_to]
    valid_strongest = s.readable.gather(-1, strongest)
    matches = _isin_rows(strongest, s.kernel_idx, N_ENTRIES) & valid_strongest

    denom = valid_strongest.sum(-1).clamp(min=1)
    agreement = matches.float().sum(-1) / denom
    agreement = agreement[valid_strongest.any(-1)]

    mean, minimum = agreement.mean().item(), agreement.min().item()
    assert mean > AGREEMENT_MEAN, f"mean top-{restrict_to} agreement {mean} below {AGREEMENT_MEAN}"
    assert minimum > AGREEMENT_FLOOR, f"min top-{restrict_to} agreement {minimum} below {AGREEMENT_FLOOR}"


@pytest.mark.parametrize("doc_lens", DOC_LENS, ids=DOC_IDS)
def test_disagreements_are_near_ties(doc_lens):
    """Where the kernel and the reference disagree, the entry each picked scores almost the same.

    Both are scored by the float32 reference, so the disagreement sits at the top-k boundary, which
    is all quantization noise can move. Whether a boundary swap changes the layer's output is a
    separate question score proximity cannot settle.
    """
    s = _selections(doc_lens)
    scale = s.score_ref.abs().max().clamp(min=1.0)

    sorted_scores, _ = s.score_ref_masked.sort(dim=-1, descending=True)
    boundary_score = sorted_scores[:, TOPK - 1]
    next_score = sorted_scores[:, TOPK]  # N_ENTRIES > TOPK, so there is always a next entry.

    dropped_mask = s.readable.gather(-1, s.ref_idx)
    dropped_mask &= ~_isin_rows(s.ref_idx, s.kernel_idx, N_ENTRIES)
    dropped_scores = torch.gather(s.score_ref, -1, s.ref_idx).masked_fill(~dropped_mask, float("-inf"))
    max_dropped_score = dropped_scores.max(dim=-1).values

    kernel_valid = s.kernel_idx < N_ENTRIES
    added_mask = kernel_valid & ~_isin_rows(s.kernel_idx, s.ref_idx_sentinel, N_ENTRIES)
    # The sentinel is out of range for this gather; clamped here and masked out immediately below.
    added_scores = torch.gather(s.score_ref, -1, s.kernel_idx.clamp(max=N_ENTRIES - 1)).masked_fill(
        ~added_mask, float("inf")
    )
    min_added_score = added_scores.min(dim=-1).values

    # The near-cutoff comparison only means anything once a query has TOPK candidates to rank;
    # `test_selection_agreement` already covers the shorter ones, whose selection is every entry.
    fully_ranked = torch.isfinite(boundary_score)
    gap_drop = (max_dropped_score - boundary_score).clamp(min=0)[fully_ranked]
    gap_add = (next_score - min_added_score).clamp(min=0)[fully_ranked]
    gap = torch.maximum(gap_drop, gap_add)

    relative_gap = (gap / scale).max().item()
    assert relative_gap < NEAR_TIE_RTOL, (
        f"the largest disagreement sits {relative_gap} of score scale past the cutoff, over {NEAR_TIE_RTOL}"
    )
