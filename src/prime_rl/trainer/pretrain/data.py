import zlib
from pathlib import Path

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.pretrain import PretrainDataConfig, TokenSourceConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.utils.logger import get_logger

# Unit of the per-epoch shuffle of a token source. A block boundary inside a document splits it, about once
# per 256 windows of 4,096 tokens.
BLOCK_TOKENS = 1 << 20
# Unit of the window shuffle of a token source: consecutive windows, so the pieces of one long document spread
# over many steps.
WINDOW_GROUP = 4096


class TokenShards:
    """Datatrove tokenized shards in one folder, the ``num_tokens`` first tokens of the files in name order.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document. File
    boundaries are document boundaries.

    Stream position ``p`` belongs to epoch ``p // num_tokens``, and every epoch reads every token once. With a
    ``seed``, each epoch reads the tokens in blocks of ``BLOCK_TOKENS`` in a seeded random order, starting at a
    seeded random offset, so epochs differ in order and window boundaries, and part of an epoch is a sample of
    the whole source, and the windows are read shuffled within groups (``window_start``). Without one
    (held-out sets), the stream is the files in name order."""

    def __init__(self, config: TokenSourceConfig, seed: tuple[int, str] | None = None):
        self.files = sorted(config.path.glob("*.ds"))
        if not self.files:
            raise FileNotFoundError(f"No datatrove .ds files in {config.path}")
        self.dtypes = [token_dtype(file) for file in self.files]
        sizes = [file.stat().st_size // dtype.itemsize for file, dtype in zip(self.files, self.dtypes)]
        self.starts = np.cumsum([0, *sizes])
        total = int(self.starts[-1])
        if config.max_tokens is not None and config.max_tokens > total:
            raise ValueError(f"{config.path} has {total:,} tokens, fewer than max_tokens={config.max_tokens:,}")
        self.num_tokens = total if config.max_tokens is None else config.max_tokens
        self.seed = None if seed is None else [seed[0], zlib.crc32(seed[1].encode())]
        self.memmaps: dict[int, tuple[np.memmap, np.memmap]] = {}
        self.layouts: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        self.window_order: tuple[tuple[int, int] | None, np.ndarray | None] = (None, None)

    def open(self, file_index: int) -> tuple[np.memmap, np.memmap]:
        # Opened lazily, so every dataloader worker maps the files itself.
        if file_index not in self.memmaps:
            file = self.files[file_index]
            tokens = np.memmap(file, dtype=self.dtypes[file_index], mode="r")
            doc_ends = np.memmap(f"{file}.index", dtype="<u8", mode="r")
            self.memmaps[file_index] = tokens, doc_ends
        return self.memmaps[file_index]

    def layout(self, epoch: int) -> tuple[np.ndarray, np.ndarray, int]:
        """An epoch's block order, the stream offsets where its blocks start, and its start offset."""
        if epoch not in self.layouts:
            rng = np.random.default_rng([*self.seed, epoch])
            order = rng.permutation(-(-self.num_tokens // BLOCK_TOKENS))
            lengths = np.minimum(BLOCK_TOKENS, self.num_tokens - order * BLOCK_TOKENS)
            self.layouts[epoch] = order, np.cumsum([0, *lengths]), int(rng.integers(self.num_tokens))
        return self.layouts[epoch]

    def locate(self, position: int, length: int) -> tuple[int, int, int]:
        """File index, offset in the file and length of the contiguous file run at stream ``position``,
        at most ``length`` tokens."""
        epoch, position = divmod(position, self.num_tokens)
        length = min(length, self.num_tokens - position)
        if self.seed is not None:
            order, block_starts, offset = self.layout(epoch)
            position = (position + offset) % self.num_tokens
            block = int(np.searchsorted(block_starts, position, side="right")) - 1
            length = min(length, int(block_starts[block + 1]) - position)
            position = int(order[block]) * BLOCK_TOKENS + position - int(block_starts[block])
        file_index = int(np.searchsorted(self.starts, position, side="right")) - 1
        length = min(length, int(self.starts[file_index + 1]) - position)
        return file_index, position - int(self.starts[file_index]), length

    def read(self, start: int, length: int) -> tuple[np.ndarray, list[int]]:
        """Tokens ``[start, start + length)`` of the stream and the offsets in them where a document starts.
        A block cut through a document starts a new one."""
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
        """Stream position of the window to read when ``position`` tokens have been read. With a seed, each epoch's
        windows are read in a seeded random order within groups of ``WINDOW_GROUP`` consecutive windows."""
        if self.seed is None:
            return position
        epoch = position // self.num_tokens
        group = (position - epoch * self.num_tokens) // (WINDOW_GROUP * seq_len)
        group_start = epoch * self.num_tokens + group * WINDOW_GROUP * seq_len
        group_end = min((epoch + 1) * self.num_tokens, group_start + WINDOW_GROUP * seq_len)
        # The windows that start in [group_start, group_end).
        first, last = -(-group_start // seq_len), -(-group_end // seq_len)
        if self.window_order[0] != (epoch, group):
            order = np.random.default_rng([*self.seed, epoch, group, 1]).permutation(last - first)
            self.window_order = (epoch, group), order
        return (first + int(self.window_order[1][position // seq_len - first])) * seq_len

    def last_token_of_first_document(self) -> int:
        tokens, doc_ends = self.open(0)
        return int(tokens[int(doc_ends[0]) - 1])


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
    """``seq_len`` windows of token sources, each window from one source of a weighted mixture.

    Window ``w``'s source is drawn from the mixture ``weights`` with an RNG keyed by ``(seed, w)``, and it continues the source's stream where that source's last
    window ended. Every data rank walks the whole window sequence but reads only its own windows; the
    state is the window counter and the tokens read per source. Each
    window reads one token past its end for the last target. With ``num_windows`` the dataset stops
    after that many windows."""

    def __init__(
        self,
        sources: dict[str, TokenShards | RandomTokens],
        weights: dict[str, float],
        seq_len: int,
        seed: int = 0,
        non_dp_size: int = 1,
        num_windows: int | None = None,
    ):
        super().__init__(non_dp_size)
        self.sources = sources
        self.names = list(sources)
        self.weights = np.array([weights.get(name, 0.0) for name in self.names])
        self.cdf = np.cumsum(self.weights)
        self.seq_len = seq_len
        self.seed = seed
        self.num_windows = num_windows
        self.tokens = dict.fromkeys(self.names, 0)

    def epochs(self) -> dict[str, float]:
        return {name: self.tokens[name] / self.sources[name].num_tokens for name in self.names}

    def state_dict(self) -> dict:
        return {"dataset": {"step": self.step, "tokens": dict(self.tokens), "epochs": self.epochs()}}

    def load_state_dict(self, state_dict: dict):
        self.step = state_dict["dataset"]["step"]
        self.tokens.update(state_dict["dataset"]["tokens"])

    def planned_epochs(self, num_tokens: int) -> dict[str, float]:
        """Expected passes over each source's tokens when ``num_tokens`` tokens are read."""
        return {
            name: num_tokens * float(weight) / self.sources[name].num_tokens
            for name, weight in zip(self.names, self.weights)
        }

    def source(self, window: int) -> str:
        if len(self.names) == 1:
            return self.names[0]
        draw = np.random.default_rng([self.seed, window]).random() * self.cdf[-1]
        return self.names[min(int(np.searchsorted(self.cdf, draw, side="right")), len(self.names) - 1)]

    def __iter__(self):
        self._setup_world_info()
        while self.num_windows is None or self.step < self.num_windows:
            window = self.step
            name = self.source(window)
            start = self.sources[name].window_start(self.tokens[name], self.seq_len)
            self.tokens[name] += self.seq_len
            self.step += 1
            if window % self.data_world_size != self.data_rank:
                continue
            tokens, doc_starts = self.sources[name].read(start, self.seq_len + 1)
            yield {
                "tokens": tokens,
                "seq_lens": np.diff([0, *(s for s in doc_starts if s < self.seq_len), self.seq_len]),
            }


def source_tokens(dataset: TokenDataset, start: int, end: int) -> dict[str, int]:
    """Tokens each source contributes to windows ``[start, end)``. Every data rank walks the same window
    sequence, so these are global counts, computed without communication."""
    tokens = dict.fromkeys(dataset.names, 0)
    for window in range(start, end):
        tokens[dataset.source(window)] += dataset.seq_len
    return tokens


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


def setup_dataloader(dataset: TokenDataset, micro_batch_size: int, num_workers: int) -> StatefulDataLoader:
    return StatefulDataLoader(
        dataset, batch_size=micro_batch_size, collate_fn=collate_rows, num_workers=num_workers, pin_memory=True
    )


def setup_train_dataset(
    config: PretrainDataConfig, max_steps: int, vocab_size: int, eos_token_id: int, non_dp_size: int
) -> TokenDataset:
    if config.type == "fake":
        sources = {"fake": RandomTokens(vocab_size, config.seed)}
        weights = {"fake": 1.0}
    else:
        sources = {name: TokenShards(source, seed=(config.seed, name)) for name, source in config.sources.items()}
        check_eos(sources, eos_token_id)
        weights = config.weights
    dataset = TokenDataset(sources, weights, config.seq_len, seed=config.seed, non_dp_size=non_dp_size)
    if config.type == "tokens":
        epochs = dataset.planned_epochs(max_steps * config.batch_tokens)
        get_logger().info(f"Planned epochs per source: {epochs}")
        if over := {name: epoch for name, epoch in epochs.items() if epoch > config.max_epochs}:
            raise ValueError(f"Planned epochs {over} exceed data.max_epochs={config.max_epochs}")
    return dataset


def setup_val_dataset(source: TokenShards, seq_len: int, num_windows: int, non_dp_size: int) -> TokenDataset:
    """One pass over the first ``num_windows`` windows of a held-out source."""
    return TokenDataset({"val": source}, {"val": 1.0}, seq_len, non_dp_size=non_dp_size, num_windows=num_windows)


def check_eos(sources: dict[str, TokenShards], eos_token_id: int) -> None:
    """Every document ends with the tokenizer's EOS. A different token there means the shards were made with
    another tokenizer."""
    for name, source in sources.items():
        last = source.last_token_of_first_document()
        if last != eos_token_id:
            raise ValueError(
                f"The first document of source {name!r} ends with token {last}, not the tokenizer's EOS "
                f"{eos_token_id}. Were the shards tokenized with tokenizer.name?"
            )
