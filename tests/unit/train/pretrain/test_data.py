from itertools import islice
from pathlib import Path

import numpy as np

from prime_rl.configs.pretrain import MixturePhaseConfig, TokenSourceConfig
from prime_rl.trainer.pretrain import data
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


def test_shuffled_epochs_read_every_token_once_in_a_new_order(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "BLOCK_TOKENS", 4)
    write_shard(tmp_path / "a" / "00.ds", [list(range(0, 30)), list(range(30, 50))])
    write_shard(tmp_path / "a" / "01.ds", [list(range(50, 70))])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    epochs = [shards.read(epoch * 70, 70)[0].tolist() for epoch in range(3)]
    assert all(sorted(epoch) == list(range(70)) for epoch in epochs)
    assert len({tuple(epoch) for epoch in epochs}) == 3
    # Reading in windows gives the same stream, so resuming at any position continues it.
    windows = [shards.read(start, 7)[0].tolist() for start in range(70, 140, 7)]
    assert sum(windows, []) == epochs[1]


def test_window_shuffle_reads_every_window_of_an_epoch_once(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "WINDOW_GROUP", 5)
    write_shard(tmp_path / "a" / "00.ds", [list(range(70))])
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    starts = [shards.window_start(position, 7) for position in range(0, 140, 7)]
    groups = [starts[i : i + 5] for i in range(0, 20, 5)]
    assert [sorted(group) for group in groups] == [list(range(i * 7, i * 7 + 35, 7)) for i in range(0, 20, 5)]
    assert starts != list(range(0, 140, 7))
    # A group never reaches into the next epoch.
    monkeypatch.setattr(data, "WINDOW_GROUP", 100)
    shards = TokenShards(TokenSourceConfig(path=tmp_path / "a"), seed=(0, "a"))
    starts = [shards.window_start(position, 7) for position in range(0, 140, 7)]
    assert sorted(starts[:10]) == list(range(0, 70, 7)) and sorted(starts[10:]) == list(range(70, 140, 7))


def test_token_dataset_resumes_and_follows_phases(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(100, 200))])
    write_shard(tmp_path / "b" / "00.ds", [list(range(200, 300))])
    sources = {name: TokenShards(TokenSourceConfig(path=tmp_path / name)) for name in "ab"}
    phases = [MixturePhaseConfig(weights={"a": 1.0}), MixturePhaseConfig(start=0.5, weights={"b": 1.0})]

    def dataset():
        return TokenDataset(sources, phases, seq_len=4, batch_size=2, max_steps=4)

    full = [window["tokens"].tolist() for window in islice(dataset(), 8)]
    assert full[:4] == [
        [100, 101, 102, 103, 104],
        [104, 105, 106, 107, 108],
        [108, 109, 110, 111, 112],
        [112, 113, 114, 115, 116],
    ]
    assert full[4][0] == 200

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
    dataset = TokenDataset(
        sources, [MixturePhaseConfig(weights={"a": 0.3, "b": 0.7})], seq_len=4, batch_size=8, max_steps=4
    )
    list(islice(dataset, 32))
    # The training loop's global per-source counts equal the tokens the dataset itself read.
    assert source_tokens(dataset, 0, 32) == dataset.tokens
    assert sum(source_tokens(dataset, 8, 16).values()) == 8 * 4


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
