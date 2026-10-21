from pathlib import Path

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.pretrain import PretrainDataConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class TokenShards:
    """Datatrove tokenized shards in one or more folders, the ``num_tokens`` first tokens of their files (in folder
    order, then file name order), read as one cyclic token stream.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document. File
    boundaries are document boundaries.

    Each epoch reads all ``num_tokens // seq_len`` windows of the stream once, in a random order and from a random
    offset drawn from ``seed`` and the epoch, so epochs differ in order and window boundaries, and part of an epoch is
    a sample of the whole stream (``window_start``). The last ``num_tokens % seq_len`` tokens of an epoch are
    skipped."""

    def __init__(self, paths: list[Path], seed: int = 0):
        self.files, self.first_files = [], []
        for path in paths:
            files = sorted(path.glob("*.ds"))
            if not files:
                raise FileNotFoundError(f"No datatrove .ds files in {path}")
            self.first_files.append(len(self.files))
            self.files += files
        self.dtypes = [token_dtype(file) for file in self.files]
        sizes = [file.stat().st_size // dtype.itemsize for file, dtype in zip(self.files, self.dtypes)]
        self.starts = np.cumsum([0, *sizes])
        self.num_tokens = int(self.starts[-1])
        self.seed = seed
        self.memmaps: dict[int, tuple[np.memmap, np.memmap]] = {}
        # The current epoch, its window order and its offset.
        self.epoch_order: tuple[int | None, np.ndarray | None, int] = (None, None, 0)

    def open(self, file_index: int) -> tuple[np.memmap, np.memmap]:
        # Opened lazily, so every dataloader worker maps the files itself.
        if file_index not in self.memmaps:
            file = self.files[file_index]
            tokens = np.memmap(file, dtype=self.dtypes[file_index], mode="r")
            doc_ends = np.memmap(f"{file}.index", dtype="<u8", mode="r")
            self.memmaps[file_index] = tokens, doc_ends
        return self.memmaps[file_index]

    def locate(self, position: int, length: int) -> tuple[int, int, int]:
        """File index, offset in the file and length of the contiguous file run at stream ``position``,
        at most ``length`` tokens."""
        position %= self.num_tokens
        length = min(length, self.num_tokens - position)
        file_index = int(np.searchsorted(self.starts, position, side="right")) - 1
        length = min(length, int(self.starts[file_index + 1]) - position)
        return file_index, position - int(self.starts[file_index]), length

    def read(self, start: int, length: int) -> tuple[np.ndarray, list[int]]:
        """Tokens ``[start, start + length)`` of the cyclic stream and the offsets in them where a document starts.
        The wrap from the end of the stream to its start begins a new document."""
        pieces, doc_starts = [], []
        offset = 0
        while offset < length:
            if offset > 0:
                doc_starts.append(offset)
            file_index, local, size = self.locate(start + offset, length - offset)
            tokens, doc_ends = self.open(file_index)
            pieces.append(tokens[local : local + size])
            first, last = np.searchsorted(doc_ends, [local, local + size], side="right")
            doc_starts.extend(offset + int(end) - local for end in doc_ends[first:last] if end < local + size)
            offset += size
        return np.concatenate(pieces).astype(np.int64), doc_starts

    def window_start(self, position: int, seq_len: int) -> int:
        """Stream offset of the window to read when ``position`` tokens have been read."""
        num_windows = self.num_tokens // seq_len
        epoch, window = divmod(position // seq_len, num_windows)
        if self.epoch_order[0] != epoch:
            rng = np.random.default_rng([self.seed, epoch])
            order = rng.permutation(num_windows).astype(np.uint32)
            self.epoch_order = epoch, order, int(rng.integers(self.num_tokens))
        _, order, offset = self.epoch_order
        return (offset + int(order[window]) * seq_len) % self.num_tokens

    def first_documents(self) -> list[np.ndarray]:
        """The first document of every folder."""
        documents = []
        for file_index in self.first_files:
            tokens, doc_ends = self.open(file_index)
            documents.append(np.asarray(tokens[: int(doc_ends[0])]))
        return documents


def token_dtype(file: Path) -> np.dtype:
    first_line = Path(f"{file}.metadata").read_text().splitlines()[0]
    return np.dtype(f"<u{int(first_line.rsplit('|', 1)[1])}")


class RandomTokens:
    """An endless stream of uniformly random token ids without document boundaries, for smoke tests."""

    num_tokens = 2**62

    def __init__(self, vocab_size: int, seed: int):
        self.vocab_size = vocab_size
        self.seed = seed

    def read(self, start: int, length: int) -> tuple[np.ndarray, list[int]]:
        return np.random.default_rng([self.seed, start]).integers(0, self.vocab_size, length), []

    def window_start(self, position: int, seq_len: int) -> int:
        return position


class TokenDataset(StatefulIterableDataset):
    """``seq_len`` windows of a token source, window ``w`` read from ``source.window_start(w * seq_len)``.

    Every data rank walks the whole window sequence but reads only its own windows; the state is the window counter.
    Each window reads one token past its end for the last target. Without ``document_masking``, a window is one
    sequence: attention and positions run over the whole window."""

    def __init__(
        self,
        source: TokenShards | RandomTokens,
        seq_len: int,
        non_dp_size: int = 1,
        document_masking: bool = True,
    ):
        super().__init__(non_dp_size)
        self.source = source
        self.seq_len = seq_len
        self.document_masking = document_masking

    def state_dict(self) -> dict:
        return {"dataset": {"step": self.step}}

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["dataset"]["step"]

    def __iter__(self):
        self._setup_world_info()
        while True:
            window = self.step
            self.step += 1
            if window % self.data_world_size != self.data_rank:
                continue
            start = self.source.window_start(window * self.seq_len, self.seq_len)
            tokens, doc_starts = self.source.read(start, self.seq_len + 1)
            doc_starts = [s for s in doc_starts if s < self.seq_len] if self.document_masking else []
            yield {"tokens": tokens, "seq_lens": np.diff([0, *doc_starts, self.seq_len])}


def collate_rows(rows: list[dict]) -> dict:
    """Concatenate windows into one packed sample of shape ``[1, len(rows) * seq_len]``. Positions restart at
    every document; attention never crosses a document or a window."""
    tokens = torch.from_numpy(np.stack([row["tokens"] for row in rows]))
    seq_lens = torch.from_numpy(np.concatenate([row["seq_lens"] for row in rows]))
    return {
        "input_ids": tokens[:, :-1].reshape(1, -1),
        "target_ids": tokens[:, 1:].reshape(1, -1),
        "loss_mask": torch.ones((1, tokens.shape[0] * (tokens.shape[1] - 1)), dtype=torch.bool),
        "position_ids": torch.cat([torch.arange(length) for length in seq_lens.tolist()]).unsqueeze(0),
        "seq_lens": seq_lens,
        "mm_kwargs": None,
        "mm_token_type_ids": None,
    }


class TrainData:
    """The training data: windows of the token pool (or of fake tokens), their dataloader and the count of windows
    trained on. Iterating yields this rank's micro-batches; every data rank walks the same window sequence, so the
    count is global without communication. ``state_dict`` holds the dataloader position and the count, for
    checkpoints."""

    def __init__(
        self, config: PretrainDataConfig, max_steps: int, vocab_size: int, eos_token_id: int, non_dp_size: int
    ):
        if config.type == "fake":
            self.source = RandomTokens(vocab_size, config.seed)
        else:
            self.source = TokenShards(config.paths, seed=config.seed)
            check_eos(self.source, eos_token_id)
            epochs = max_steps * config.batch_size * config.seq_len / self.source.num_tokens
            get_logger().info(f"Token pool: {self.source.num_tokens:,} tokens, {epochs:.3f} planned epochs")
            if epochs > config.max_epochs:
                raise ValueError(f"Planned epochs {epochs:.3f} exceed data.max_epochs={config.max_epochs}")
        dataset = TokenDataset(
            self.source,
            config.seq_len,
            non_dp_size=non_dp_size,
            document_masking=config.type == "fake" or config.document_masking,
        )
        self.loader = StatefulDataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            collate_fn=collate_rows,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        self.seq_len = config.seq_len
        self.windows_per_micro_batch = config.micro_batch_size * get_world().world_size // non_dp_size
        self.windows = 0
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        # Created at the first micro-batch, after a potential resume: iter() forks the workers with the dataset's
        # current state.
        if self.iterator is None:
            self.iterator = iter(self.loader)
        micro_batch = next(self.iterator)
        self.windows += self.windows_per_micro_batch
        return micro_batch

    def state_dict(self) -> dict:
        return {"dataloader": self.loader.state_dict(), "windows": self.windows}

    def load_state_dict(self, state_dict: dict):
        self.loader.load_state_dict(state_dict["dataloader"])
        self.windows = state_dict["windows"]

    def metrics(self) -> dict[str, float]:
        return {
            "progress/num_samples": self.windows,
            "progress/epochs": self.windows * self.seq_len / self.source.num_tokens,
        }


def check_eos(shards: TokenShards, eos_token_id: int) -> None:
    """Every document ends with the tokenizer's EOS. A different token there means the shards were made with
    another tokenizer. Checked on the first document of every folder."""
    for file_index, document in zip(shards.first_files, shards.first_documents()):
        if document[-1] != eos_token_id:
            raise ValueError(
                f"The first document of {shards.files[file_index].parent} ends with token {document[-1]}, not the "
                f"tokenizer's EOS {eos_token_id}. Were the shards tokenized with tokenizer.name?"
            )


def starts_with_bos(shards: TokenShards, bos_token_id: int | None) -> bool:
    """Whether the documents start with the tokenizer's BOS, read off the first document of every folder. The eval
    suite formats its requests the same way."""
    starts = {
        shards.files[i].parent.as_posix(): int(doc[0]) == bos_token_id
        for i, doc in zip(shards.first_files, shards.first_documents())
    }
    if len(set(starts.values())) > 1:
        raise ValueError(f"Only some token folders start their documents with BOS: {starts}")
    return all(starts.values())
