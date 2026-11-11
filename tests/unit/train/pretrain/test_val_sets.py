import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from prime_rl.trainer.pretrain.val_sets import (
    head_logprobs,
    prepare_set,
    requests_of,
    set_metrics,
    summarize,
    summarize_ctxcont,
)


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


def test_head_logprobs_chunks_match_the_full_softmax():
    hidden, weight = torch.randn(7, 4, dtype=torch.bfloat16), torch.randn(11, 4, dtype=torch.bfloat16)
    targets = torch.randint(0, 11, (7,))
    logprobs, hits = head_logprobs(hidden, weight, targets, chunk=3)
    logits = hidden.float() @ weight.float().T
    torch.testing.assert_close(logprobs, logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0])
    assert torch.equal(hits, logits.argmax(-1) == targets)


def test_pairs_requests_fit_the_window(tmp_path):
    (tmp_path / "pairs").mkdir()
    table = pa.table({"good": ["a long good sentence", "ok"], "bad": ["a long bad sentence!", "no"]})
    pq.write_table(table, tmp_path / "pairs" / "test.parquet")
    (tmp_path / "pairs" / "meta.json").write_text(json.dumps({"kind": "pairs"}))
    eval_set = prepare_set("pairs", tmp_path, CharTokenizer(), seq_len=8, bos=True)
    assert all(len(ids) - 1 <= 8 for ids, _ in eval_set.requests)
    logprobs = [np.full(len(ids) - start, -1.0) for ids, start in eval_set.requests]
    summary, _ = summarize(eval_set, logprobs, [np.ones_like(lp, dtype=bool) for lp in logprobs])
    assert set_metrics("pairs", 8, summary)["val/pairs/acc@8"] == 0.0  # equal scores: good never beats bad


def test_requests_without_bos_start_at_the_text(tmp_path):
    for name, kind, columns in [
        ("docs", "ppl", {"text": ["hello"]}),
        ("cc", "ctxcont", {"context": ["Q:"], "continuation": [" yes"]}),
        ("pairs", "pairs", {"good": ["ab"], "bad": ["ba"]}),
    ]:
        (tmp_path / name).mkdir()
        pq.write_table(pa.table(columns), tmp_path / name / "test.parquet")
        (tmp_path / name / "meta.json").write_text(json.dumps({"kind": kind}))
    docs = prepare_set("docs", tmp_path, CharTokenizer(), seq_len=32, bos=False)
    assert docs.requests == [([ord(char) for char in "hello"] + [2], 1)]
    assert docs.context["bytes"].tolist() == [4]  # the first character is context only
    with_bos = prepare_set("docs", tmp_path, CharTokenizer(), seq_len=32, bos=True)
    assert with_bos.requests[0][0][0] == 1 and with_bos.context["bytes"].tolist() == [5]
    ids, start = prepare_set("cc", tmp_path, CharTokenizer(), seq_len=32, bos=False).requests[0]
    assert ids == [ord(char) for char in "Q: yes"] and start == 2
    assert prepare_set("pairs", tmp_path, CharTokenizer(), seq_len=32, bos=False).requests[0] == ([97, 98], 1)


def test_ctxcont_picks_the_likeliest_choice_with_and_without_length_normalization():
    rows = [{"context": "Q:", "continuation": " ab", "choices": [" ab", " c"], "gold": 0}]
    options = [requests_of(row) for row in rows]
    # The gold choice has the lower total log-prob (-2 vs -1.5) but the higher log-prob per byte (-2/3 vs -3/4).
    logprobs = [np.array([-1.0, -1.0]), np.array([-1.5])]
    hits = [np.array([True, False]), np.array([True])]
    summary, _ = summarize_ctxcont(rows, options, logprobs, hits)
    assert summary["acc"] == 0.0 and summary["acc_norm"] == 1.0
    assert summary["loss"] == 1.0 and summary["bpb"] == pytest.approx(2 / (3 * np.log(2)))
    assert summary["greedy_em"] == 0.0 and summary["greedy_token_acc"] == 0.5
