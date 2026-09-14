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
AGREEMENT_MEAN, AGREEMENT_P1 = 0.98, 0.97


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
    """`score[t,s] = sum_h w[t,h] relu(sum_d q[t,h,d] k[s,d])`, in float32.

    The kernel's own formula, through the `(NUM_Q, HEADS, NUM_K)` intermediate the kernel exists to
    avoid, so the only difference between the two is the kernel's FP8 quantization. The ReLU is
    in place because a second buffer that size is most of a gigabyte.
    """
    per_head = (q.float() @ k.float().T).relu_()
    return torch.einsum("th,ths->ts", w.float(), per_head)


def readable_mask(ks: torch.Tensor, ke: torch.Tensor) -> torch.Tensor:
    """`(NUM_Q, NUM_K)` bool: key `s` is in query `t`'s range `[ks[t], ke[t])`."""
    key_idx = torch.arange(NUM_K, device=ks.device)
    return (key_idx[None, :] >= ks[:, None]) & (key_idx[None, :] < ke[:, None])


def selected_mask(picks: torch.Tensor) -> torch.Tensor:
    """`(NUM_Q, NUM_K)` bool: which keys each query's picks name.

    A query's unused slots come back holding `NUM_K`, the kernel's "no pick" sentinel, so they
    scatter into one throwaway column that is sliced back off.
    """
    mask = torch.zeros(NUM_Q, NUM_K + 1, dtype=torch.bool, device=picks.device)
    return mask.scatter_(1, picks, True)[:, :NUM_K]


@pytest.mark.parametrize("segments", SEGMENTS, ids=SEGMENT_IDS)
def test_selection_agreement(segments):
    """The FP8 kernel's picks against the float32 reference's, as sets, per query.

    Both sides read the same bfloat16 tensors, so the only difference measured is the kernel's own
    FP8 quantization. A query with no more than `TOPK` readable keys has nothing to rank away and
    must name that whole set exactly; the statistics cover the saturated queries, the only ones a
    pick can move across the top-k boundary of.
    """
    ks, ke = ranges(segments)
    readable = readable_mask(ks, ke)

    q = torch.randn(NUM_Q, HEADS, DIM, device="cuda").bfloat16()
    k = torch.randn(NUM_K, DIM, device="cuda").bfloat16()
    w = torch.randn(NUM_Q, HEADS, device="cuda").bfloat16()

    scores = eager_score_reference(q, k, w).masked_fill(~readable, float("-inf"))
    # A query with fewer than TOPK readable keys fills its surplus picks from the -inf columns.
    ref_selected = selected_mask(scores.topk(TOPK, dim=-1).indices) & readable
    kernel_selected = selected_mask(fp8_indexer(q, k, w, ks.int(), ke.int(), TOPK).long())

    saturated = readable.sum(-1) > TOPK
    assert torch.equal(kernel_selected[~saturated], ref_selected[~saturated]), (
        "a query that could hold every key it may read did not pick them all"
    )
    assert saturated.any(), "vacuous probe: no query has more readable keys than it can pick"

    agreement = (ref_selected & kernel_selected).sum(-1)[saturated] / TOPK
    mean, p1 = agreement.mean().item(), agreement.quantile(0.01).item()
    assert mean > AGREEMENT_MEAN, f"mean set agreement {mean} below {AGREEMENT_MEAN}"
    assert p1 > AGREEMENT_P1, f"p1 set agreement {p1} below {AGREEMENT_P1}"
