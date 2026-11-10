from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import prime_rl.trainer.pretrain.data as pretrain_data
from prime_rl.configs.pretrain import DatatroveDataConfig
from prime_rl.trainer.data import setup_dataloader
from prime_rl.trainer.pretrain.data import (
    TokenDataset,
    TokenShards,
    detect_separator,
    document_edges,
    warn_if_no_document_separator,
)
from prime_rl.trainer.world import reset_world

# Documents of the pool, in two folders, with an empty one. A token's id is its position in the pool.
DOCUMENTS = [[range(0, 10), range(10, 30), range(30, 30), range(30, 60)], [range(60, 103)]]
NUM_TOKENS = 103
DOCUMENT_STARTS = {0, 10, 30, 60}


@pytest.fixture
def folders(tmp_path: Path) -> list[Path]:
    folders = []
    for index, documents in enumerate(DOCUMENTS):
        folder = tmp_path / str(index)
        folder.mkdir()
        np.array([token for document in documents for token in document], dtype="<u4").tofile(folder / "00.ds")
        np.cumsum([len(document) for document in documents], dtype="<u8").tofile(folder / "00.ds.index")
        (folder / "00.ds.metadata").write_text("tokenizer|4\n")
        folders.append(folder)
    return folders


def write_shard(path: Path, documents: list[list[int]]) -> None:
    """A datatrove DocumentTokenizer shard: uint32 tokens, uint64 cumulative document ends, metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.array([token for document in documents for token in document], dtype="<u4").tofile(path)
    np.cumsum([len(document) for document in documents], dtype="<u8").tofile(f"{path}.index")
    Path(f"{path}.metadata").write_text("tokenizer|4\n0\n0")


def consecutive_documents(lengths: list[int]) -> list[list[int]]:
    """Documents of the given lengths whose tokens are their positions in the stream."""
    starts = np.cumsum([0, *lengths]).tolist()
    return [list(range(start, start + length)) for start, length in zip(starts, lengths)]


def take(dataset: TokenDataset, num_samples: int) -> list[dict]:
    samples = iter(dataset)
    return [next(samples) for _ in range(num_samples)]


def test_every_epoch_reads_every_window_once_in_a_new_order(folders: list[Path]):
    seq_len = 8
    num_windows = NUM_TOKENS // seq_len
    samples = take(TokenDataset(TokenShards(folders, seed=0), seq_len), 3 * num_windows)
    epochs = [samples[i : i + num_windows] for i in range(0, len(samples), num_windows)]
    epoch_starts = [[sample["input_ids"][0] for sample in epoch] for epoch in epochs]
    for epoch, starts in zip(epochs, epoch_starts):
        for sample in epoch:
            window = [(sample["input_ids"][0] + i) % NUM_TOKENS for i in range(seq_len + 1)]
            assert sample["input_ids"] == window[:-1]
            assert sample["target_ids"] == window[1:]
        # The windows of an epoch tile the pool from one of them on.
        tilings = [sorted((start - first) % NUM_TOKENS for start in starts) for first in starts]
        assert list(range(0, num_windows * seq_len, seq_len)) in tilings
    # Shuffled: few consecutive windows are neighbors in the pool.
    for starts in epoch_starts:
        assert sum((b - a) % NUM_TOKENS == seq_len for a, b in zip(starts, starts[1:])) < num_windows // 2
    # The window boundaries move between epochs.
    assert len({frozenset(starts) for starts in epoch_starts}) == 3

    assert take(TokenDataset(TokenShards(folders, seed=0), seq_len), num_windows) == epochs[0]
    assert take(TokenDataset(TokenShards(folders, seed=1), seq_len), num_windows) != epochs[0]


@pytest.mark.parametrize("document_masking", [True, False])
def test_document_masking(folders: list[Path], document_masking: bool):
    seq_len = 16
    for sample in take(TokenDataset(TokenShards(folders), seq_len, document_masking), 20):
        start = sample["input_ids"][0]
        if document_masking:
            # Documents start at the pool's document boundaries and where it wraps around.
            boundaries = [i for i in range(1, seq_len) if (start + i) % NUM_TOKENS in DOCUMENT_STARTS]
            seq_lens = np.diff([0, *boundaries, seq_len]).tolist()
        else:
            seq_lens = [seq_len]
        assert sample["seq_lens"] == seq_lens
        assert sample["position_ids"] == [position for length in seq_lens for position in range(length)]
        assert sample["loss_mask"] == [True] * seq_len


def set_world(monkeypatch, rank: int, world_size: int) -> None:
    reset_world()
    for name, value in (
        ("RANK", rank),
        ("LOCAL_RANK", rank),
        ("WORLD_SIZE", world_size),
        ("LOCAL_WORLD_SIZE", world_size),
    ):
        monkeypatch.setenv(name, str(value))


@pytest.mark.parametrize(("num_samples", "expected"), [(5, 5), (None, 12), (100, 12)])
def test_validation_stops_every_rank_at_the_same_window(
    folders: list[Path], monkeypatch, num_samples: int | None, expected: int
):
    """One epoch capped at num_samples windows: two data ranks together read the first windows of the sequence one
    rank reads, each window once."""
    seq_len = 8
    set_world(monkeypatch, rank=0, world_size=1)
    sequence = take(TokenDataset(TokenShards(folders), seq_len), expected)
    samples_by_rank = []
    for rank in range(2):
        set_world(monkeypatch, rank=rank, world_size=2)
        samples_by_rank.append(list(TokenDataset(TokenShards(folders), seq_len, max_epochs=1, num_samples=num_samples)))
    reset_world()
    assert [samples_by_rank[w % 2][w // 2] for w in range(expected)] == sequence
    assert sum(map(len, samples_by_rank)) == expected


def first_tokens(batch: dict, seq_len: int) -> list[int]:
    """The first token of every window in a micro-batch row, which is the window's position in the pool."""
    return batch["input_ids"][0, ::seq_len].tolist()


def test_resume_on_another_number_of_ranks_continues_the_window_sequence(folders: list[Path], monkeypatch):
    """Two data ranks read 3 micro-batches each and checkpoint; one rank resumes from rank 0's checkpoint. Together
    they read the windows an uninterrupted single-rank run reads."""
    config = DatatroveDataConfig(paths=folders, seq_len=8, batch_size=2, micro_batch_size=2)

    def dataloader(rank: int, world_size: int):
        set_world(monkeypatch, rank, world_size)
        return setup_dataloader(TokenDataset(TokenShards(config.paths), config.seq_len), config, cp=1)

    uninterrupted = iter(dataloader(rank=0, world_size=1))
    expected = [token for _ in range(12) for token in first_tokens(next(uninterrupted), config.seq_len)]

    windows_by_rank, states = [], []
    for rank in range(2):
        loader = dataloader(rank=rank, world_size=2)
        batches = iter(loader)
        windows_by_rank.append([token for _ in range(3) for token in first_tokens(next(batches), config.seq_len)])
        states.append(loader.state_dict())
    # Rank r reads the windows r, r + 2, ...
    windows = [windows_by_rank[w % 2][w // 2] for w in range(12)]

    resumed = dataloader(rank=0, world_size=1)
    resumed.load_state_dict(states[0])
    batches = iter(resumed)
    windows += [token for _ in range(6) for token in first_tokens(next(batches), config.seq_len)]
    reset_world()
    assert windows == expected


def test_separator_check_skips_empty_files_and_documents(tmp_path: Path, monkeypatch):
    """Both folders start with an empty file and an empty document. Only the one whose first real document has no
    separator warns, and only real documents are read: an empty first document does not wrap around to the last
    token."""
    eos = 1
    for name, document in (("separated", [5, 6, eos]), ("run-together", [5, 6, 7])):
        folder = tmp_path / name
        folder.mkdir()
        for file, documents in (("00.ds", []), ("01.ds", [[], document])):
            np.array([token for document in documents for token in document], dtype="<u4").tofile(folder / file)
            np.cumsum([len(document) for document in documents], dtype="<u8").tofile(folder / f"{file}.index")
            (folder / f"{file}.metadata").write_text("tokenizer|4\n")
    warnings = []
    monkeypatch.setattr(pretrain_data, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    tokenizer = SimpleNamespace(all_special_ids=[eos], added_tokens_decoder={}, bos_token_id=None, decode=str)
    shards = TokenShards([tmp_path / "separated", tmp_path / "run-together"])
    warn_if_no_document_separator(shards, tokenizer)
    assert len(warnings) == 1
    assert "run-together" in warnings[0]
    edges = {folder.name: edges for folder, edges in document_edges(shards, 16).items()}
    assert edges == {"separated": ([5], [eos]), "run-together": ([5], [7])}
    with pytest.raises(ValueError, match=r"run-together: .*\(7\)"):
        detect_separator(TokenShards([tmp_path / "run-together"]), tokenizer)


EOS = 999  # every document ends with it, as the tokenizer writes the shards


def test_eod_mask_loss(tmp_path):
    # Documents of words (their positions in the stream) ending in EOS; lengths so that windows hold document ends and
    # cut documents.
    words = consecutive_documents([2, 6, 1, 11, 3, 4, 9, 5, 2, 7] * 6)
    write_shard(tmp_path / "a" / "00.ds", [[*document, EOS] for document in words])
    shards = TokenShards([tmp_path / "a"], seed=0)
    seq_len = 8
    off = list(islice(TokenDataset(shards, seq_len), 40))
    on = list(islice(TokenDataset(shards, seq_len, eod_mask_token=EOS), 40))
    eos_inputs = word_to_eos = 0
    for plain, masked in zip(off, on):
        # Only the loss mask changes.
        assert {k: v for k, v in plain.items() if k != "loss_mask"} == {
            k: v for k, v in masked.items() if k != "loss_mask"
        }
        for token, target, before, after in zip(
            masked["input_ids"], masked["target_ids"], plain["loss_mask"], masked["loss_mask"]
        ):
            # (a) an EOS input has no loss; every other position keeps its loss as before.
            assert after == (before and token != EOS)
            eos_inputs += before and token == EOS
            # (b) the last word of a document still learns to predict EOS.
            if before and target == EOS and token != EOS:
                assert after
                word_to_eos += 1
    assert eos_inputs > 0 and word_to_eos > 0


def test_eod_mask_loss_detects_the_shared_separator(tmp_path):
    """The separator is the special token all checked documents of all folders end with."""
    for name, separator in (("a", EOS), ("b", EOS), ("c", EOS + 1)):
        documents = [[*document, separator] for document in consecutive_documents([3, 5, 2] * 8)]
        write_shard(tmp_path / name / "00.ds", documents)
    # EOS + 1 is special only as an added token, like Qwen's <|endoftext|> once the trainer sets pad to EOS.
    added_tokens_decoder = {EOS + 1: SimpleNamespace(special=True)}
    tokenizer = SimpleNamespace(
        all_special_ids=[EOS], added_tokens_decoder=added_tokens_decoder, bos_token_id=None, decode=str
    )
    assert detect_separator(TokenShards([tmp_path / "a", tmp_path / "b"]), tokenizer) == EOS
    assert detect_separator(TokenShards([tmp_path / "c"]), tokenizer) == EOS + 1
    with pytest.raises(ValueError, match=r"/a: .*\(999\); .*/c: .*\(1000\)"):
        detect_separator(TokenShards([tmp_path / "a", tmp_path / "c"]), tokenizer)


def test_eod_mask_loss_masks_nothing_on_bos_separated_shards(tmp_path, monkeypatch):
    """Documents that start with BOS and end with ordinary tokens have nothing to mask: BOS predicts each document's
    first token."""
    bos = 1
    write_shard(tmp_path / "a" / "00.ds", [[bos, *document] for document in consecutive_documents([3, 5, 2] * 8)])
    tokenizer = SimpleNamespace(all_special_ids=[bos], added_tokens_decoder={}, bos_token_id=bos, decode=str)
    infos = []
    monkeypatch.setattr(pretrain_data, "get_logger", lambda: SimpleNamespace(info=infos.append))
    assert detect_separator(TokenShards([tmp_path / "a"]), tokenizer) is None
    assert len(infos) == 1
