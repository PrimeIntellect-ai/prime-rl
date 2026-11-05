"""Pretraining data: datatrove tokenized shards read as one token pool and cut into windows of ``seq_len`` tokens."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.supervised import DatatroveDataConfig
from prime_rl.trainer.data import StatefulIterableDataset
from prime_rl.utils.logger import get_logger

MAX_OPEN_FILES = 256


def mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64's output function, increment included (Steele, Lea and Flood, "Fast splittable pseudorandom number
    generators", 2014; constants from Vigna's reference implementation): a bijective hash of uint64 values."""
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def permute(indices: np.ndarray, n: int, key: int) -> np.ndarray:
    """A pseudo-random permutation of ``range(n)``, chosen by ``key``, at ``indices``: a 4-round Feistel network on
    the smallest even number of bits that holds ``n``, applied again to values of ``n`` or more until they fall below
    (cycle walking; Black and Rogaway, "Ciphers with arbitrary finite domains", 2002).

    A stored permutation (``rng.permutation(n)``) costs 8 bytes per window in every dataloader worker: fine at 10B
    tokens, tens of GB at 10T. This one builds no table, and a window's position is a pure function of
    (seed, epoch, index), which is what makes resuming exact."""
    half = max(1, ((n - 1).bit_length() + 1) // 2)
    mask = np.uint64((1 << half) - 1)
    round_keys = mix(np.uint64(key) + np.arange(4, dtype=np.uint64))
    values = np.asarray(indices, dtype=np.uint64).copy()
    walking = np.ones(len(values), dtype=bool)
    while walking.any():
        left, right = values[walking] >> np.uint64(half), values[walking] & mask
        for round_key in round_keys:
            left, right = right, left ^ (mix(right ^ round_key) & mask)
        values[walking] = (left << np.uint64(half)) | right
        walking[walking] = values[walking] >= n
    return values.astype(np.int64)


def token_dtype(file: Path) -> np.dtype:
    first_line = Path(f"{file}.metadata").read_text().splitlines()[0]
    return np.dtype(f"<u{int(first_line.rsplit('|', 1)[1])}")


class TokenShards:
    """Datatrove tokenized shards in one or more folders, all their files (in folder order, then file name order)
    read as one cyclic token stream.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document. File boundaries are
    document boundaries.

    Each epoch reads all ``num_tokens // seq_len`` windows of the stream once, in a random order and from a random
    offset drawn from ``seed`` and the epoch, so epochs differ in order and window boundaries. The last
    ``num_tokens % seq_len`` tokens of an epoch are skipped."""

    def __init__(self, paths: list[Path], seed: int = 0):
        self.files = []
        for path in paths:
            files = sorted(path.glob("*.ds"))
            if not files:
                raise FileNotFoundError(f"No datatrove .ds files in {path}")
            self.files += files
        self.dtypes = [token_dtype(file) for file in self.files]
        sizes = [file.stat().st_size // dtype.itemsize for file, dtype in zip(self.files, self.dtypes)]
        self.starts = np.cumsum([0, *sizes])
        self.num_tokens = int(self.starts[-1])
        self.seed = seed
        self.memmaps: OrderedDict[int, tuple[np.memmap, np.memmap]] = OrderedDict()

    def open(self, file_index: int) -> tuple[np.memmap, np.memmap]:
        # Opened lazily, so every dataloader worker maps the files itself. At most MAX_OPEN_FILES stay open (each
        # holds two file descriptors), the least recently used are closed.
        if file_index in self.memmaps:
            self.memmaps.move_to_end(file_index)
        else:
            file = self.files[file_index]
            tokens = np.memmap(file, dtype=self.dtypes[file_index], mode="r")
            doc_ends = np.memmap(f"{file}.index", dtype="<u8", mode="r")
            self.memmaps[file_index] = tokens, doc_ends
            if len(self.memmaps) > MAX_OPEN_FILES:
                self.memmaps.popitem(last=False)
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
            first, last = np.searchsorted(doc_ends, np.array([local, local + size], dtype=np.uint64), side="right")
            doc_starts.extend(offset + int(end) - local for end in doc_ends[first:last] if end < local + size)
            offset += size
        return np.concatenate(pieces).astype(np.int64), doc_starts

    def window_start(self, position: int, seq_len: int) -> int:
        """Stream offset of the window to read when ``position`` tokens have been read: the epoch reads its
        ``num_tokens // seq_len`` windows in a pseudo-random order and from a random offset, both drawn from ``seed``
        and the epoch. Every window is computed on its own, so neither time nor memory grow with the pool."""
        num_windows = self.num_tokens // seq_len
        epoch, window = divmod(position // seq_len, num_windows)
        key = mix(mix(np.array([self.seed], dtype=np.uint64)) + np.uint64(epoch))
        offset = int(mix(key ^ np.uint64(1))[0] % np.uint64(self.num_tokens))
        order = int(permute(np.array([window]), num_windows, int(key[0]))[0])
        return (offset + order * seq_len) % self.num_tokens


class TokenDataset(StatefulIterableDataset):
    """``seq_len`` windows of a token pool, window ``w`` read from ``shards.window_start(w * seq_len)``. With
    ``document_masking``, a window has one segment per document (``seq_lens``) and positions restart at every
    document; without, it is one segment. Every data rank walks the whole window sequence but reads only its own
    windows, so ``step`` counts the windows of all ranks and ``epoch`` is the fraction of the pool read. With
    ``max_epochs`` or ``num_samples`` (windows over all ranks), every rank stops at the same window."""

    def __init__(
        self,
        shards: TokenShards,
        seq_len: int,
        document_masking: bool = True,
        non_dp_size: int = 1,
        max_epochs: int | None = None,
        num_samples: int | None = None,
    ):
        super().__init__(non_dp_size)
        if shards.num_tokens < seq_len:
            raise ValueError(f"The token pool has {shards.num_tokens} tokens, fewer than seq_len ({seq_len})")
        self.shards = shards
        self.seq_len = seq_len
        self.document_masking = document_masking
        limits = [] if num_samples is None else [num_samples]
        if max_epochs is not None:
            limits.append(max_epochs * (shards.num_tokens // seq_len))
        self.max_step = min(limits, default=None)

    def state_dict(self) -> dict:
        # A checkpoint follows a whole optimizer step, after which every data rank has read the same number of
        # windows. Rounding ``step`` up to whole rounds over the data ranks gives every rank the same position, so a
        # run resumed on another number of GPUs continues the window sequence exactly.
        rounds = -(-self.step // self.data_world_size)
        return {"step": rounds * self.data_world_size, "epoch": self.epoch}

    def __iter__(self):
        self._setup_world_info()
        while self.max_step is None or self.step < self.max_step:
            window = self.step
            self.step += 1
            self.epoch = self.step * self.seq_len / self.shards.num_tokens
            if window % self.data_world_size != self.data_rank:
                continue
            start = self.shards.window_start(window * self.seq_len, self.seq_len)
            tokens, doc_starts = self.shards.read(start, self.seq_len + 1)
            doc_starts = [s for s in doc_starts if s < self.seq_len] if self.document_masking else []
            # An empty document repeats a start; its zero-length segment would repeat an attention offset.
            seq_lens = [length for length in np.diff([0, *doc_starts, self.seq_len]).tolist() if length > 0]
            ids = tokens.tolist()
            yield {
                "input_ids": ids[:-1],
                "target_ids": ids[1:],
                "position_ids": [position for length in seq_lens for position in range(length)],
                "loss_mask": [True] * self.seq_len,
                "seq_lens": seq_lens,
                "mm_kwargs": None,
                "mm_token_type_ids": None,
            }


def warn_if_no_document_separator(shards: TokenShards, tokenizer: PreTrainedTokenizer) -> None:
    """Warn when the first non-empty document of a folder neither ends with a special token nor starts with BOS:
    prime-rl reads documents as they are, so shards tokenized without a separator run their documents together."""
    first_documents = {}
    for index, file in enumerate(shards.files):
        # Empty files and empty documents carry no separator to check.
        if file.parent in first_documents or shards.starts[index + 1] == shards.starts[index]:
            continue
        tokens, doc_ends = shards.open(index)
        begin = 0
        for end in doc_ends:
            if end > begin:
                break
            begin = end
        first_documents[file.parent] = tokens[int(begin) : int(end)]
    special_ids = set(tokenizer.all_special_ids)
    for folder, document in first_documents.items():
        if int(document[-1]) in special_ids or int(document[0]) == tokenizer.bos_token_id:
            continue
        get_logger().warning(
            f"The documents in {folder} have no separator: the first one ends with token {int(document[-1])} "
            f"({tokenizer.decode([int(document[-1])])!r}), not a special token, and does not start with BOS.\n"
            "The documents run together, so the model will not learn where a document ends and will not learn to "
            "stop.\n"
            "Tokenize again with datatrove's DocumentTokenizer and set eos_token to the tokenizer's end-of-text "
            "token (e.g. <|endoftext|> for Qwen)."
        )


def setup_token_dataset(
    config: DatatroveDataConfig,
    non_dp_size: int = 1,
    max_epochs: int | None = None,
    num_samples: int | None = None,
) -> TokenDataset:
    return TokenDataset(
        TokenShards(config.paths, seed=config.seed),
        seq_len=config.seq_len,
        document_masking=config.document_masking,
        non_dp_size=non_dp_size,
        max_epochs=max_epochs,
        num_samples=num_samples,
    )
