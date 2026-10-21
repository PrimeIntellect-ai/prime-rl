from itertools import islice
from pathlib import Path

import numpy as np

from prime_rl.trainer.pretrain.data import TokenDataset, TokenShards, collate_rows


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def test_token_shards_pool_folders_and_wrap(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2, 3], [4, 5]])
    write_shard(tmp_path / "b" / "00.ds", [[6, 7, 8, 9]])
    shards = TokenShards([tmp_path / "a", tmp_path / "b"])
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
    shards = TokenShards([tmp_path / "a"], seed=0)
    # 72 tokens hold 10 windows of 7; the 2-token tail of each epoch is skipped.
    orders = []
    for epoch in range(3):
        starts = [shards.window_start(position, 7) for position in range(epoch * 70, epoch * 70 + 70, 7)]
        tokens = [token for start in starts for token in shards.read(start, 7)[0].tolist()]
        assert len(set(tokens)) == 70
        orders.append(tuple(starts))
    assert len(set(orders)) == 3
    # Another reader (a dataloader worker, a resumed run) draws the same order.
    again = TokenShards([tmp_path / "a"], seed=0)
    assert tuple(again.window_start(position, 7) for position in range(70, 140, 7)) == orders[1]


def test_token_dataset_resumes(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(100, 200))])
    shards = TokenShards([tmp_path / "a"], seed=0)
    full = [window["tokens"].tolist() for window in islice(TokenDataset(shards, seq_len=4), 8)]
    first = TokenDataset(shards, seq_len=4)
    iterator = iter(first)
    for _ in range(3):
        next(iterator)
    resumed = TokenDataset(shards, seq_len=4)
    resumed.load_state_dict(first.state_dict())
    assert [window["tokens"].tolist() for window in islice(resumed, 5)] == full[3:8]


def test_without_document_masking_every_window_is_one_sequence(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2], [3, 4], [5, 6], [7, 8]])
    shards = TokenShards([tmp_path / "a"])
    # Every window of 4 holds a document start.
    masked = [window["seq_lens"] for window in islice(TokenDataset(shards, seq_len=4), 8)]
    assert all(len(seq_lens) > 1 for seq_lens in masked)
    unmasked = islice(TokenDataset(shards, seq_len=4, document_masking=False), 8)
    assert [window["seq_lens"].tolist() for window in unmasked] == [[4]] * 8


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
