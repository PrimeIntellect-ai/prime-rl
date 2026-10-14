import bisect
from pathlib import Path

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.pretrain import MixturePhaseConfig, PretrainDataConfig, TokenSourceConfig
from prime_rl.trainer.sft.data import StatefulIterableDataset
from prime_rl.utils.logger import get_logger


class TokenShards:
    """Datatrove tokenized shards in one folder, read as one token stream in file-name order.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document. File
    boundaries are document boundaries. The stream wraps around after ``num_tokens``."""

    def __init__(self, config: TokenSourceConfig):
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
        self.memmaps: dict[int, tuple[np.memmap, np.memmap]] = {}

    def open(self, file_index: int) -> tuple[np.memmap, np.memmap]:
        # Opened lazily, so every dataloader worker maps the files itself.
        if file_index not in self.memmaps:
            file = self.files[file_index]
            tokens = np.memmap(file, dtype=self.dtypes[file_index], mode="r")
            doc_ends = np.memmap(f"{file}.index", dtype="<u8", mode="r")
            self.memmaps[file_index] = tokens, doc_ends
        return self.memmaps[file_index]

    def read(self, start: int, length: int) -> tuple[np.ndarray, list[int]]:
        """Tokens ``[start, start + length)`` of the stream and the offsets in them where a document starts."""
        pieces, doc_starts = [], []
        offset = 0
        while offset < length:
            if offset > 0:
                doc_starts.append(offset)
            position = (start + offset) % self.num_tokens
            file_index = int(np.searchsorted(self.starts, position, side="right")) - 1
            local = position - int(self.starts[file_index])
            size = min(length - offset, int(self.starts[file_index + 1]) - position, self.num_tokens - position)
            tokens, doc_ends = self.open(file_index)
            pieces.append(tokens[local : local + size])
            first, last = np.searchsorted(doc_ends, [local, local + size], side="right")
            doc_starts.extend(offset + int(end) - local for end in doc_ends[first:last] if end < local + size)
            offset += size
        return np.concatenate(pieces).astype(np.int64), doc_starts

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


class TokenDataset(StatefulIterableDataset):
    """``seq_len`` windows of token sources, each window from one source of a weighted mixture.

    Window ``w`` belongs to step ``w // batch_size``; its source is drawn from that step's phase weights
    with an RNG keyed by ``(seed, w)``, and it continues the source's stream where that source's last
    window ended. Every data rank walks the whole window sequence but reads only its own windows, so the
    state (the window counter and the tokens read per source) does not depend on the world size. Each
    window reads one token past its end for the last target. With ``num_windows`` the dataset stops
    after that many windows."""

    def __init__(
        self,
        sources: dict[str, TokenShards | RandomTokens],
        phases: list[MixturePhaseConfig],
        seq_len: int,
        batch_size: int,
        max_steps: int,
        seed: int = 0,
        non_dp_size: int = 1,
        num_windows: int | None = None,
    ):
        super().__init__(non_dp_size)
        self.sources = sources
        self.names = list(sources)
        self.phase_starts = [round(phase.start * max_steps) for phase in phases]
        self.phase_cdfs = [np.cumsum([phase.weights.get(name, 0.0) for name in self.names]) for phase in phases]
        self.seq_len = seq_len
        self.batch_size = batch_size
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

    def planned_epochs(self, max_steps: int) -> dict[str, float]:
        """Expected passes over each source's tokens in ``max_steps`` steps."""
        ends = [*self.phase_starts[1:], max_steps]
        step_tokens = self.batch_size * self.seq_len
        tokens = sum(
            (end - start) * step_tokens * np.diff(cdf, prepend=0.0)
            for start, end, cdf in zip(self.phase_starts, ends, self.phase_cdfs)
        )
        return {name: float(tokens[i]) / self.sources[name].num_tokens for i, name in enumerate(self.names)}

    def source(self, window: int) -> str:
        if len(self.names) == 1:
            return self.names[0]
        phase = bisect.bisect_right(self.phase_starts, window // self.batch_size) - 1
        cdf = self.phase_cdfs[phase]
        draw = np.random.default_rng([self.seed, window]).random() * cdf[-1]
        return self.names[min(int(np.searchsorted(cdf, draw, side="right")), len(self.names) - 1)]

    def __iter__(self):
        self._setup_world_info()
        while self.num_windows is None or self.step < self.num_windows:
            window = self.step
            name = self.source(window)
            start = self.tokens[name]
            self.tokens[name] += self.seq_len
            self.step += 1
            if window % self.data_world_size != self.data_rank:
                continue
            tokens, doc_starts = self.sources[name].read(start, self.seq_len + 1)
            yield {
                "tokens": tokens,
                "seq_lens": np.diff([0, *(s for s in doc_starts if s < self.seq_len), self.seq_len]),
            }


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
        phases = [MixturePhaseConfig(weights={"fake": 1.0})]
    else:
        sources = {name: TokenShards(source) for name, source in config.sources.items()}
        check_eos(sources, eos_token_id)
        phases = config.phases
    dataset = TokenDataset(
        sources, phases, config.seq_len, config.batch_size, max_steps, seed=config.seed, non_dp_size=non_dp_size
    )
    if config.type == "tokens":
        epochs = dataset.planned_epochs(max_steps)
        get_logger().info(f"Planned epochs per source: {epochs}")
        if over := {name: epoch for name, epoch in epochs.items() if epoch > config.max_epochs}:
            raise ValueError(f"Planned epochs {over} exceed data.max_epochs={config.max_epochs}")
    return dataset


def setup_val_dataset(source: TokenShards, seq_len: int, num_windows: int, non_dp_size: int) -> TokenDataset:
    """One pass over the first ``num_windows`` windows of a held-out source."""
    phases = [MixturePhaseConfig(weights={"val": 1.0})]
    return TokenDataset({"val": source}, phases, seq_len, 1, 1, non_dp_size=non_dp_size, num_windows=num_windows)


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
