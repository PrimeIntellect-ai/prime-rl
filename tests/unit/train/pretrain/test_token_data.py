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
    setup_document_dataset,
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


def scored_segments(batch: dict) -> list[tuple[list[int], list[int], list[int]]]:
    """Inputs, targets and positions of every scored segment of a micro-batch row."""
    segments, start = [], 0
    for length in batch["seq_lens"].tolist():
        end = start + length
        if batch["loss_mask"][0, start:end].all():
            segments.append(
                tuple(batch[key][0, start:end].tolist() for key in ("input_ids", "target_ids", "position_ids"))
            )
        else:
            assert not batch["loss_mask"][0, start:end].any()
        start = end
    return segments


@pytest.mark.parametrize("num_documents", [None, 4])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_validation_scores_every_document_on_its_own(
    folders: list[Path], monkeypatch, world_size: int, num_documents: int | None
):
    """The data ranks together score every document (or the first num_documents) as if it were alone: windows of
    seq_len scored tokens, each with positions from 0 and no token before its own first one, every token after the
    document's first scored once. Every rank runs the same number of micro-batches."""
    config = DatatroveDataConfig(paths=folders, seq_len=8, batch_size=2, micro_batch_size=2)
    expected = []
    for document in [list(document) for documents in DOCUMENTS for document in documents][:num_documents]:
        for start in range(0, len(document) - 1, config.seq_len):
            window = document[start : start + config.seq_len + 1]
            expected.append((window[:-1], window[1:], list(range(len(window) - 1))))

    batches_by_rank = []
    for rank in range(world_size):
        set_world(monkeypatch, rank, world_size)
        batches_by_rank.append(
            list(setup_dataloader(setup_document_dataset(config, num_documents=num_documents), config, cp=1))
        )
    reset_world()

    assert len({len(batches) for batches in batches_by_rank}) == 1
    for batches in batches_by_rank:
        assert all(batch["input_ids"].shape == (1, config.seq_len * config.micro_batch_size) for batch in batches)
    segments = [segment for batches in batches_by_rank for batch in batches for segment in scored_segments(batch)]
    assert sorted(segments) == sorted(expected)


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
    """Both folders start with an empty file and an empty document; only the one whose first real document has no
    separator warns."""
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
    tokenizer = SimpleNamespace(all_special_ids=[eos], bos_token_id=None, decode=str)
    warn_if_no_document_separator(TokenShards([tmp_path / "separated", tmp_path / "run-together"]), tokenizer)
    assert len(warnings) == 1
    assert "run-together" in warnings[0]
