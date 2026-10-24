from itertools import islice
from pathlib import Path

import numpy as np

from prime_rl.trainer.sft.data import CatDataset, TokenDataset, TokenShards


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def test_token_dataset_resumes(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [list(range(100, 200))])
    shards = TokenShards([tmp_path / "a"], seed=0)
    full = [window["input_ids"] for window in islice(TokenDataset(shards, seq_len=4), 8)]
    first = TokenDataset(shards, seq_len=4)
    iterator = iter(first)
    for _ in range(3):
        next(iterator)
    resumed = TokenDataset(shards, seq_len=4)
    resumed.load_state_dict(first.state_dict())
    assert [window["input_ids"] for window in islice(resumed, 5)] == full[3:8]


def test_packed_windows_keep_their_documents(tmp_path):
    write_shard(tmp_path / "a" / "00.ds", [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]])
    shards = TokenShards([tmp_path / "a"], seed=0)
    windows = list(islice(TokenDataset(shards, seq_len=4), 2))
    row = next(iter(CatDataset(TokenDataset(shards, seq_len=4), seq_len=8, pad_to_multiple_of=1)))
    assert row["seq_lens"] == windows[0]["seq_lens"] + windows[1]["seq_lens"]
    assert len(row["seq_lens"]) > 2
    assert row["position_ids"] == [p for length in row["seq_lens"] for p in range(length)]
    assert row["input_ids"] == windows[0]["input_ids"] + windows[1]["input_ids"]
