import math

import numpy as np

from prime_rl.trainer.pretrain.suite import pack_requests, summarize_ctxcont


def test_pack_requests_covers_every_request_within_the_budget():
    requests = [([0] * n, 1) for n in (5, 9, 3, 9, 2)]
    batches = pack_requests(requests, token_budget=10)
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    assert all(sum(len(requests[i][0]) - 1 for i in batch) <= 10 for batch in batches)


def test_summarize_ctxcont_scores_the_gold_choice():
    rows = [
        {"context": "Q", "continuation": "ab", "choices": ["ab", "cd"], "gold": 0},
        {"context": "Q", "continuation": "e", "choices": ["x", "e"], "gold": 1},
    ]
    options = [[("Q", c) for c in row["choices"]] for row in rows]
    logprobs = [np.array([-1.0, -1.0]), np.array([-3.0]), np.array([-0.5]), np.array([-2.0])]
    hits = [np.array([True, False]), np.array([False]), np.array([True]), np.array([False])]
    summary, _ = summarize_ctxcont(rows, options, logprobs, hits)
    assert summary["acc"] == 0.5  # item 0: gold wins; item 1: the other choice wins
    assert math.isclose(summary["bpb"], (2.0 + 2.0) / (3 * math.log(2)))  # gold nll / gold continuation bytes
    assert summary["greedy_em"] == 0.0
