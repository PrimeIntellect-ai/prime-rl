import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
from torch import nn

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.pretrain.suite import (
    head_logprobs,
    pack_requests,
    prepare_multi_length_set,
    prepare_set,
    score_requests,
    score_sets,
    set_metrics,
    summarize,
    summarize_ctxcont,
)


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
    metrics = set_metrics("mc", 2048, summary)
    assert metrics["eval/mc/acc@2k"] == 0.5 and all(type(value) is float for value in metrics.values())


class CharTokenizer:
    bos_token_id, eos_token_id = 1, 2

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        if isinstance(texts, str):
            return {
                "input_ids": [ord(char) for char in texts],
                "offset_mapping": [(i, i + 1) for i in range(len(texts))],
            }
        return {
            "input_ids": [[ord(char) for char in text] for text in texts],
            "offset_mapping": [[(i, i + 1) for i in range(len(text))] for text in texts],
        }


def test_multi_length_set_scores_shared_requests_once(tmp_path):
    (tmp_path / "docs").mkdir()
    pq.write_table(pa.table({"text": ["short", "a much longer document"]}), tmp_path / "docs" / "test.parquet")
    (tmp_path / "docs" / "meta.json").write_text(json.dumps({"kind": "ppl", "parquet_sha256": "x"}))
    multi = prepare_multi_length_set("docs", tmp_path, CharTokenizer(), [8, 32], bos=True)
    # "short" (5 chars + BOS + EOS) is one window at both lengths; the long document is 3 windows at 8, 1 at 32.
    assert len(multi.versions[8].requests) == 4 and len(multi.versions[32].requests) == 2
    assert len(multi.requests) == 5
    assert multi.index[8][0] == multi.index[32][0]


def test_head_logprobs_chunks_match_the_full_softmax():
    hidden, weight = torch.randn(7, 4, dtype=torch.bfloat16), torch.randn(11, 4, dtype=torch.bfloat16)
    targets = torch.randint(0, 11, (7,))
    logprobs, hits = head_logprobs(hidden, weight, targets, chunk=3)
    logits = hidden.float() @ weight.float().T
    torch.testing.assert_close(logprobs, logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0])
    assert torch.equal(hits, logits.argmax(-1) == targets)


class TokenPositionModel(nn.Module):
    """Scores each label by its value and position, so a request's scores do not depend on how it is packed."""

    def forward(self, input_ids, position_ids, labels, seq_lens):
        keep = labels != IGNORE_INDEX
        return {"logprobs": (labels + position_ids / 1000)[keep].double(), "entropy": (labels % 2 == 0)[keep]}


@pytest.mark.gpu
def test_score_sets_matches_scoring_each_set_alone(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        for name, texts in {"a": ["short", "a much longer document"], "b": ["xyz", "another text here"]}.items():
            (tmp_path / name).mkdir()
            pq.write_table(pa.table({"text": texts}), tmp_path / name / "test.parquet")
            (tmp_path / name / "meta.json").write_text(json.dumps({"kind": "ppl", "parquet_sha256": "x"}))
        eval_sets = [prepare_multi_length_set(name, tmp_path, CharTokenizer(), [8, 32], bos=True) for name in "ab"]
        together = score_sets(TokenPositionModel(), eval_sets, token_budget=40, group=dist.group.WORLD)
        for eval_set, (logprobs, hits) in zip(eval_sets, together):
            alone_logprobs, alone_hits = score_requests(
                TokenPositionModel(), eval_set.requests, token_budget=40, group=dist.group.WORLD
            )
            assert all(np.array_equal(x, y) for x, y in zip(logprobs, alone_logprobs))
            assert all(np.array_equal(x, y) for x, y in zip(hits, alone_hits))
    finally:
        dist.destroy_process_group()


def test_pairs_requests_fit_the_window(tmp_path):
    (tmp_path / "pairs").mkdir()
    table = pa.table({"good": ["a long good sentence", "ok"], "bad": ["a long bad sentence!", "no"]})
    pq.write_table(table, tmp_path / "pairs" / "test.parquet")
    (tmp_path / "pairs" / "meta.json").write_text(json.dumps({"kind": "pairs", "parquet_sha256": "x"}))
    eval_set = prepare_set("pairs", tmp_path, CharTokenizer(), seq_len=8, bos=True)
    assert all(len(ids) - 1 <= 8 for ids, _ in eval_set.requests)
    logprobs = [np.full(len(ids) - start, -1.0) for ids, start in eval_set.requests]
    summary, _ = summarize(eval_set, logprobs, [np.ones_like(lp, dtype=bool) for lp in logprobs])
    assert set_metrics("pairs", 8, summary)["eval/pairs/acc@8"] == 0.0  # equal scores: good never beats bad


def test_requests_without_bos_start_at_the_text(tmp_path):
    for name, kind, columns in [
        ("docs", "ppl", {"text": ["hello"]}),
        ("cc", "ctxcont", {"context": ["Q:"], "continuation": [" yes"]}),
        ("pairs", "pairs", {"good": ["ab"], "bad": ["ba"]}),
    ]:
        (tmp_path / name).mkdir()
        pq.write_table(pa.table(columns), tmp_path / name / "test.parquet")
        (tmp_path / name / "meta.json").write_text(json.dumps({"kind": kind, "parquet_sha256": "x"}))
    docs = prepare_set("docs", tmp_path, CharTokenizer(), seq_len=32, bos=False)
    assert docs.requests == [([ord(char) for char in "hello"] + [2], 1)]
    assert docs.context["bytes"].tolist() == [4]  # the first character is context only
    with_bos = prepare_set("docs", tmp_path, CharTokenizer(), seq_len=32, bos=True)
    assert with_bos.requests[0][0][0] == 1 and with_bos.context["bytes"].tolist() == [5]
    ids, start = prepare_set("cc", tmp_path, CharTokenizer(), seq_len=32, bos=False).requests[0]
    assert ids == [ord(char) for char in "Q: yes"] and start == 2
    assert prepare_set("pairs", tmp_path, CharTokenizer(), seq_len=32, bos=False).requests[0] == ([97, 98], 1)
