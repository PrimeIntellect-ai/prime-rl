import pytest
import torch

import prime_rl.trainer.models.kernels.fp8_indexer as fp8_indexer_module
import prime_rl.trainer.models.kernels.indexer_topk as indexer_topk
from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer_prequantized
from prime_rl.trainer.models.kernels.indexer_topk import exact_topk_bf16, fp32_topk_reference

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not fp8_indexer_module._deep_select_supported(torch.device("cuda")),
        reason="DeepSelect not installed or not built for this GPU",
    ),
]

N_COLS = 65536
# Row lengths at every edge of the visit orders: short rows, around top-k, around the FP32 kernel's index-order window
# (8192), inside the rows the BF16 kernel visits in index order and the FP32 kernel does not, around the BF16 kernel's
# window (16384), around the tail and segment boundaries of permuted rows, and full rows.
EDGE_LENGTHS = [0, 1, 511, 512, 513, 2047, 2048, 2049, 4097, 8191, 8192, 8193, 8704, 8705, 12288, 12289, 16383, 16384,
                16385, 16896, 16897, 24576, 24577, 32768, 40000, 65535, 65536]  # fmt: skip

TIE_PATTERNS = {
    "bf16-normal": lambda rows: torch.randn(rows, N_COLS, device="cuda"),
    "ints-0-to-2": lambda rows: torch.randint(0, 3, (rows, N_COLS), device="cuda").float(),
    "all-equal": lambda rows: torch.ones(rows, N_COLS, device="cuda"),
    "signed-zeros-and-ones": lambda rows: torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")[
        torch.randint(0, 4, (rows, N_COLS), device="cuda")
    ],
    "relu-like-many-zeros": lambda rows: torch.relu(torch.randn(rows, N_COLS, device="cuda") - 2.5),
    "negatives": lambda rows: -torch.rand(rows, N_COLS, device="cuda"),
    "ascending-steps": lambda rows: (torch.arange(N_COLS, device="cuda") // 64).float().expand(rows, -1),
    "sparse-spikes": lambda rows: (torch.rand(rows, N_COLS, device="cuda") < 0.004).float(),
}


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)


def row_lengths(rows: int) -> torch.Tensor:
    lengths = torch.randint(0, N_COLS + 1, (rows,), device="cuda", dtype=torch.int32)
    lengths[: len(EDGE_LENGTHS)] = torch.tensor(EDGE_LENGTHS, device="cuda", dtype=torch.int32)[:rows]
    return lengths


def visit_order(length: int, index_order_max: int) -> torch.Tensor:
    """The order in which a DeepSelect kernel visits the first `length` logits of a row (see indexer_topk.py)."""
    if length <= index_order_max:
        return torch.arange(length, device="cuda")
    n_perm = (-(-length // 512) - 1) // 8 * 8
    mul = indexer_topk.PERM_MUL_PRIME % n_perm
    segments = torch.tensor([(lin * mul + indexer_topk.PERM_ADD_BASE) % n_perm for lin in range(n_perm)], device="cuda")
    permuted = (segments[:, None] * 512 + torch.arange(512, device="cuda")).flatten()
    return torch.cat([torch.arange(n_perm * 512, length, device="cuda"), permuted])


def visit_order_model(logits: torch.Tensor, lengths: torch.Tensor, topk: int, index_order_max: int) -> torch.Tensor:
    """The top-k under (value descending, visit rank ascending), in visit order; `0..E-1` then -1 for E <= topk."""
    picks = torch.full((logits.shape[0], topk), -1, dtype=torch.int64, device="cuda")
    for row, length in enumerate(lengths.tolist()):
        if length <= topk:
            picks[row, :length] = torch.arange(length, device="cuda")
            continue
        order = visit_order(length, index_order_max)
        values = logits[row, order].float() + 0.0  # -0.0 -> +0.0: the kernels compare as floats
        kept = torch.sort(values, descending=True, stable=True).indices[:topk]
        picks[row] = order[kept.sort().values]
    return picks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32-kernel", "bf16-kernel"])
@pytest.mark.parametrize("pattern", ["bf16-normal", "ints-0-to-2", "signed-zeros-and-ones"])
def test_visit_order_model_matches_deep_select(pattern, dtype):
    """Both DeepSelect kernels pick, and order, exactly as the visit-order model says, on logits that tie at the
    k-th place in most rows; they differ only in how long a row they visit in index order (8192 vs 16384)."""
    import deep_select

    rows, topk = 64, 512
    lengths = row_lengths(rows)
    logits = TIE_PATTERNS[pattern](rows).bfloat16().to(dtype)
    _, picks = deep_select.topk(
        logits, topk, end=lengths, indices_type=torch.int32, idx_oob_fill_value=-1, return_value=False
    )

    index_order_max = indexer_topk.FP32_INDEX_ORDER_MAX if dtype == torch.float32 else indexer_topk.BF16_INDEX_ORDER_MAX
    assert torch.equal(picks.long(), visit_order_model(logits, lengths, topk, index_order_max))


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("batch", ["single-wave", "multi-wave"])
@pytest.mark.parametrize("pattern", list(TIE_PATTERNS))
def test_exact_topk_bf16_matches_fp32_path(pattern, batch, topk):
    """`exact_topk_bf16` writes exactly what DeepSelect's FP32 kernel picks from the FP32 copy, picks and order.

    Rows end at every visit-order edge and hold garbage past their end (+inf, huge finite values, NaN), which
    `fp8_fp4_mqa_logits` leaves uncleaned. DeepSelect's BF16 kernel runs another configuration for a batch of at
    most one wave of CTAs than for a larger one; both are covered.
    """
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = 64 if batch == "single-wave" else 2 * n_sms + 64
    lengths = row_lengths(rows)
    ks = torch.randint(0, 1000, (rows,), device="cuda", dtype=torch.int32)
    sentinel = N_COLS + 1000
    logits = TIE_PATTERNS[pattern](rows).bfloat16()
    past_end = torch.arange(N_COLS, device="cuda")[None, :] >= lengths[:, None]
    garbage = torch.tensor([float("inf"), 3e38, float("nan"), -3e38], device="cuda").bfloat16()
    logits = torch.where(past_end, garbage[torch.randint(0, 4, (rows, N_COLS), device="cuda")], logits)

    expected = fp32_topk_reference(logits, lengths, ks, topk, sentinel)
    picks = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    exact_topk_bf16(logits, lengths, ks, topk, sentinel, picks)

    assert torch.equal(picks, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the indexer scores into BF16 logits on SM100 only",
)
@pytest.mark.parametrize("narrowed", [False, True], ids=["full-width", "narrowed"])
def test_indexer_picks_match_fp32_path(narrowed, monkeypatch):
    """The indexer picks, and orders, exactly what it picked through the FP32 copy of its logits.

    The keys are a few distinct rows repeated, so most scores tie. Queries come in chunks of 512: the first chunk
    reads at most 200 keys, so its logits narrowed to its longest row are narrower than the top-k and take the FP32
    path, and two segments of 20,000 keys give the others every row length up to 20,000, across both index-order
    windows. Narrowed, each chunk's logits are only as wide as its longest row.
    """
    heads, dim, topk = 64, 128, 512
    segments = [(512, 200), (1792, 20000), (1792, 20000)]
    n_q, n_k = sum(n for n, _ in segments), sum(n for _, n in segments)
    ks, ke, base = [], [], 0
    for seg_q, seg_k in segments:
        ks.append(torch.full((seg_q,), base))
        ke.append(base + torch.arange(seg_q) * seg_k // (seg_q - 1))
        base += seg_k
    ks, ke = torch.cat(ks).int().cuda(), torch.cat(ke).int().cuda()
    q = torch.randn(n_q * heads, dim, device="cuda").bfloat16()
    k = torch.randn(7, dim, device="cuda").bfloat16()[torch.randint(0, 7, (n_k,), device="cuda")]
    w = torch.randn(n_q, heads, device="cuda").bfloat16()
    q_fp8, q_scales = fp8_indexer_module.per_token_group_quant_fp8(q, dim)
    q_fp8, w = q_fp8.view(n_q, heads, dim), w * q_scales.view(n_q, heads)
    host_row_lengths = (ke - ks).cpu()[None] if narrowed else None
    monkeypatch.setattr(fp8_indexer_module, "LOGITS_CHUNK_BYTES", 1)

    picks = fp8_indexer_prequantized(q_fp8, k, w, ks, ke, topk, host_row_lengths=host_row_lengths)
    monkeypatch.setattr(fp8_indexer_module, "exact_topk_bf16_supported", lambda topk, n_cols: False)
    fp32_picks = fp8_indexer_prequantized(q_fp8, k, w, ks, ke, topk, host_row_lengths=host_row_lengths)

    assert torch.equal(picks, fp32_picks)


def test_check_catches_a_changed_visit_order(monkeypatch):
    """The once-per-process check raises when the assumed visit order is not DeepSelect's."""
    indexer_topk._check_deep_select_visit_orders.cache_clear()
    monkeypatch.setattr(indexer_topk, "PERM_ADD_BASE", indexer_topk.PERM_ADD_BASE + 1)
    try:
        with pytest.raises(RuntimeError, match="visits rows in another order"):
            indexer_topk._check_deep_select_visit_orders(torch.device("cuda"), 512)
    finally:
        indexer_topk._check_deep_select_visit_orders.cache_clear()
