from transformers import AutoTokenizer

from prime_rl.trainer.pretrain.evals.common import token_byte_lengths
from prime_rl.trainer.pretrain.evals.loglik import LoglikTask, pack_requests, score_task
from prime_rl.trainer.pretrain.evals.tasks import TaskItem


def test_token_byte_lengths_count_utf8_bytes():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    lengths = token_byte_lengths(tokenizer)
    text = "Grüße aus Zürich — 東京 🚀 naïve café"
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert int(lengths[token_ids].sum()) == len(text.encode())
    assert lengths[tokenizer.eos_token_id] == 0


def test_pack_requests_spans_cover_continuations():
    requests = [([1, 2, 3], [4, 5]), ([6] * 10, [7]), ([8, 9], [10, 11, 12])]
    rows = pack_requests(requests, seq_len=8)
    found = {}
    for row in rows:
        assert len(row.sample["input_ids"]) == 8
        assert sum(row.sample["seq_lens"]) == 8
        for index, start, end in row.spans:
            found[index] = row.sample["target_ids"][start:end]
    assert found == {0: [4, 5], 1: [7], 2: [10, 11, 12]}


def test_score_task():
    items = [
        TaskItem(requests=[("q", " a"), ("q", " bbbb")], gold=1),
        TaskItem(requests=[("q", " a"), ("q", " bbbb")], gold=0),
    ]
    task = LoglikTask(name="toy", items=items, rows=[])
    metrics = score_task(task, [-1.0, -2.0, -1.0, -6.0])
    assert metrics["evals/toy/acc"] == 0.5
    assert metrics["evals/toy/acc_norm"] == 1.0
