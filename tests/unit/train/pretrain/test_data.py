from itertools import islice
from pathlib import Path

import numpy as np

from prime_rl.configs.pretrain import TokenSourceConfig
from prime_rl.trainer.pretrain.data import TokenDataset, TokenShards, collate_rows, source_tokens


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def test_token_shards_read_across_files_and_wrap(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2, 3], [4, 5]])
    write_shard(tmp_path / "a" / "01.ds", [[6, 7, 8, 9]])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"))
    assert shards.num_tokens == 9
    tokens, doc_starts = shards.read(1, 6)
    assert tokens.tolist() == [2, 3, 4, 5, 6, 7]
    assert doc_starts == [2, 4]
    tokens, doc_starts = shards.read(7, 4)
    assert tokens.tolist() == [8, 9, 1, 2]
    assert doc_starts == [2]


def test_shuffled_epochs_read_every_window_once_in_a_new_order(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(0, 30)), list(range(30, 50))])
    write_shard(tmp_path / "a" / "01.ds", [list(range(50, 72))])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    # 72 tokens hold 10 windows of 7; the 2-token tail of each epoch is skipped.
    orders = []
    for epoch in range(3):
        starts = [shards.window_start(position, 7) for position in range(epoch * 70, epoch * 70 + 70, 7)]
        tokens = [token for start in starts for token in shards.read(start, 7)[0].tolist()]
        assert len(set(tokens)) == 70
        orders.append(tuple(starts))
    assert len(set(orders)) == 3
    # Another reader (a dataloader worker, a resumed run) draws the same order.
    again = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    assert tuple(again.window_start(position, 7) for position in range(70, 140, 7)) == orders[1]


def test_token_dataset_resumes_and_continues_each_source(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(100, 200))])
    write_shard(tmp_path / "b" / "00.ds", [list(range(200, 300))])
    sources = {name: TokenShards(TokenSourceConfig(path=tmp_path / name)) for name in "ab"}

    def dataset():
        return TokenDataset(sources, {"a": 0.5, "b": 0.5}, seq_len=4)

    full = [window["tokens"].tolist() for window in islice(dataset(), 8)]
    for start in (100, 200):
        windows = [window for window in full if start <= window[0] < start + 100]
        assert windows and windows == [list(range(start + 4 * i, start + 4 * i + 5)) for i in range(len(windows))]

    first = dataset()
    iterator = iter(first)
    for _ in range(3):
        next(iterator)
    resumed = dataset()
    resumed.load_state_dict(first.state_dict())
    assert [window["tokens"].tolist() for window in islice(resumed, 5)] == full[3:8]


def test_source_tokens_count_every_window_of_the_mixture(tmp_path):
    for name in "ab":
        write_shard(tmp_path / name / "00.ds", [list(range(100))])
    sources = {name: TokenShards(TokenSourceConfig(path=tmp_path / name)) for name in "ab"}
    dataset = TokenDataset(sources, {"a": 0.3, "b": 0.7}, seq_len=4)
    list(islice(dataset, 32))
    # The training loop's global per-source counts equal the tokens the dataset itself read.
    assert source_tokens(dataset, 0, 32) == dataset.tokens
    assert sum(source_tokens(dataset, 8, 16).values()) == 8 * 4


def test_without_document_masking_every_window_is_one_sequence(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2, 3], [4, 5], [6, 7, 8, 9]])
    sources = {"a": TokenShards(TokenSourceConfig(path=tmp_path / "a"))}
    masked = [window["seq_lens"].tolist() for window in TokenDataset(sources, {"a": 1.0}, seq_len=4, num_windows=2)]
    assert masked == [[3, 1], [1, 3]]
    dataset = TokenDataset(sources, {"a": 1.0}, seq_len=4, num_windows=2, document_masking=False)
    assert [window["seq_lens"].tolist() for window in dataset] == [[4], [4]]


def test_collate_rows_restarts_positions_at_documents():
    rows = [
        {"tokens": np.arange(5), "seq_lens": np.array([3, 1])},
        {"tokens": np.arange(10, 15), "seq_lens": np.array([4])},
    ]
    batch = collate_rows(rows)
    assert batch["input_ids"].tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]
    assert batch["target_ids"].tolist() == [[1, 2, 3, 4, 11, 12, 13, 14]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 0, 1, 2, 3]]
    assert batch["seq_lens"].tolist() == [3, 1, 4]
