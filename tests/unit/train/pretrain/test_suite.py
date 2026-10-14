import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prime_rl.trainer.pretrain.suite import pack_requests, prepare_multi_length_set, summarize_ctxcont


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


class CharTokenizer:
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def test_multi_length_set_scores_shared_requests_once(tmp_path):
    (tmp_path / "docs").mkdir()
    pq.write_table(pa.table({"text": ["short", "a much longer document"]}), tmp_path / "docs" / "test.parquet")
    (tmp_path / "docs" / "meta.json").write_text(json.dumps({"kind": "ppl", "parquet_sha256": "x"}))
    multi = prepare_multi_length_set("docs", tmp_path, CharTokenizer(), [8, 32])
    # "short" (5 chars + BOS + EOS) is one window at both lengths; the long document is 3 windows at 8, 1 at 32.
    assert len(multi.versions[8].requests) == 4 and len(multi.versions[32].requests) == 2
    assert len(multi.requests) == 5
    assert multi.index[8][0] == multi.index[32][0]
