import math

import torch

from prime_rl.trainer.pretrain.diagnostics import MIN_VISIBLE_KEYS, attention_stats, logit_stats


def test_attention_stats_respect_documents_and_causality():
    # Two documents; zero queries give uniform attention, so each query's entropy is log(visible keys).
    lengths = (MIN_VISIBLE_KEYS + 6, MIN_VISIBLE_KEYS + 2)
    num_tokens = sum(lengths)
    cu_seqlens = torch.tensor([0, lengths[0], num_tokens], dtype=torch.int32)
    q = torch.zeros(num_tokens, 2, 4)
    k = torch.randn(num_tokens, 1, 4)
    q[-1, 1] = 1.0  # one non-zero query in the second head, last token of the second document
    max_logit, entropy_sum, queries = attention_stats(q, k, cu_seqlens, scale=0.5)

    visible = [i + 1 for length in lengths for i in range(length)]
    assert int(queries) == sum(v >= MIN_VISIBLE_KEYS for v in visible)
    assert torch.isclose(entropy_sum[0], torch.tensor(sum(math.log(v) for v in visible if v >= MIN_VISIBLE_KEYS)))
    second_document_keys = k[lengths[0] :, 0]
    assert torch.isclose(max_logit[1], ((second_document_keys @ q[-1, 1]).max() * 0.5).clamp(min=0))  # other queries: 0


def test_logit_stats_match_full_logits():
    hidden, weight = torch.randn(2, 5, 8), torch.randn(37, 8)
    logits = hidden.reshape(10, 8) @ weight.T
    max_logit, lse_sum, lse_min, lse_max = logit_stats(hidden, weight, chunk_size=3)
    lse = logits.logsumexp(dim=-1)
    torch.testing.assert_close(
        torch.stack([max_logit, lse_sum, lse_min, lse_max]),
        torch.stack([logits.max(), lse.sum(), lse.min(), lse.max()]),
    )
