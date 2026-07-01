import torch

from prime_rl.inference.patches import _trim_logits_to_org_vocab


def _synthetic_top_p_with_contiguous_row_assumption(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    assumed_rows = torch.as_strided(
        logits,
        size=logits.shape,
        stride=(logits.shape[-1], 1),
    )

    for row in assumed_rows:
        sorted_probs, sorted_indices = torch.sort(torch.softmax(row, dim=-1), descending=True)
        remove_sorted = sorted_probs.cumsum(dim=-1) > top_p
        remove_sorted[0] = False

        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(0, sorted_indices, remove_sorted)
        row[remove] = -torch.inf

    return logits


def test_trim_logits_from_padded_vocab_returns_contiguous_rows():
    org_vocab_size = 5
    padded_vocab_size = 8
    logits = torch.arange(3 * padded_vocab_size, dtype=torch.float32).reshape(3, padded_vocab_size)

    sliced = logits[..., :org_vocab_size]
    guarded = _trim_logits_to_org_vocab(logits, org_vocab_size)

    assert not sliced.is_contiguous()
    assert sliced.stride() == (padded_vocab_size, 1)
    assert guarded.is_contiguous()
    assert guarded.stride() == (org_vocab_size, 1)
    torch.testing.assert_close(guarded, sliced)


def test_trim_logits_without_padding_preserves_contiguous_storage():
    org_vocab_size = 5
    logits = torch.arange(3 * org_vocab_size, dtype=torch.float32).reshape(3, org_vocab_size)

    guarded = _trim_logits_to_org_vocab(logits, org_vocab_size)

    assert guarded.is_contiguous()
    assert guarded.stride() == logits.stride()
    assert guarded.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(guarded, logits)


def test_trim_logits_preserves_per_row_top_values():
    org_vocab_size = 4
    logits = torch.tensor(
        [
            [0.1, 3.0, -2.0, 1.0, 99.0],
            [4.0, -3.0, 5.0, 0.5, 88.0],
            [-1.0, 2.5, 2.0, 7.0, 77.0],
        ],
        dtype=torch.float32,
    )
    sliced = logits[..., :org_vocab_size]
    guarded = _trim_logits_to_org_vocab(logits, org_vocab_size)

    assert torch.equal(guarded.argmax(dim=-1), sliced.argmax(dim=-1))
    torch.testing.assert_close(
        torch.topk(guarded, k=2, dim=-1).values,
        torch.topk(sliced, k=2, dim=-1).values,
    )


def test_guarded_logits_avoid_nan_log_softmax_with_contiguous_row_top_p_assumption():
    org_vocab_size = 3
    padded_logits = torch.tensor(
        [
            [0.0, -2.0, -3.0, 100.0],
            [5.0, 4.0, -torch.inf, 0.0],
        ],
        dtype=torch.float32,
    )

    unguarded = padded_logits.clone()[..., :org_vocab_size]
    assert torch.isfinite(unguarded[1]).any()

    _synthetic_top_p_with_contiguous_row_assumption(unguarded, top_p=0.9)

    assert torch.isneginf(unguarded[1]).all()
    assert torch.isnan(torch.log_softmax(unguarded[1], dim=-1)).all()

    guarded = _trim_logits_to_org_vocab(padded_logits, org_vocab_size)
    _synthetic_top_p_with_contiguous_row_assumption(guarded, top_p=0.9)

    assert torch.isfinite(guarded[1]).any()
    assert not torch.isnan(torch.log_softmax(guarded[1], dim=-1)).any()
