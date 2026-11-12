"""Pretraining data: datatrove tokenized shards read as one token pool and cut into windows of ``seq_len`` tokens,
and for validation every held-out document scored on its own."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.pretrain import DatatroveDataConfig
from prime_rl.trainer.data import StatefulIterableDataset
from prime_rl.utils.logger import get_logger
from prime_rl.utils.sequence import repeat_to_length

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
    windows, so ``step`` counts the windows of all ranks and ``epoch`` is the fraction of the pool read."""

    def __init__(
        self,
        shards: TokenShards,
        seq_len: int,
        document_masking: bool = True,
        non_dp_size: int = 1,
        eod_mask_token: int | None = None,
    ):
        super().__init__(non_dp_size)
        if shards.num_tokens < seq_len:
            raise ValueError(f"The token pool has {shards.num_tokens} tokens, fewer than seq_len ({seq_len})")
        self.shards = shards
        self.seq_len = seq_len
        self.document_masking = document_masking
        self.eod_mask_token = eod_mask_token

    def state_dict(self) -> dict:
        # A checkpoint follows a whole optimizer step, after which every data rank has read the same number of
        # windows. Rounding ``step`` up to whole rounds over the data ranks gives every rank the same position, so a
        # run resumed on another number of GPUs continues the window sequence exactly.
        rounds = -(-self.step // self.data_world_size)
        return {"step": rounds * self.data_world_size, "epoch": self.epoch}

    def __iter__(self):
        self._setup_world_info()
        while True:
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
                "loss_mask": eod_loss_mask(ids[:-1], [True] * self.seq_len, self.eod_mask_token),
                "seq_lens": seq_lens,
                "mm_kwargs": None,
                "mm_token_type_ids": None,
            }


def eod_loss_mask(input_ids: list[int], loss_mask: list[bool], eod_token: int | None) -> list[bool]:
    """``loss_mask`` with every position whose input is ``eod_token`` off; unchanged when ``eod_token`` is None."""
    if eod_token is None:
        return loss_mask
    return [keep and token != eod_token for keep, token in zip(loss_mask, input_ids)]


Window = tuple[int, int, int]
"""File index and the token range ``[start, end)`` of a window in the file."""


def document_windows(shards: TokenShards, seq_len: int, num_documents: int | None = None) -> list[Window]:
    """Every document of ``shards`` as stored (or the first ``num_documents``), cut into non-overlapping windows of
    ``seq_len`` scored tokens. A window holds up to ``seq_len + 1`` tokens and scores all but its first. The next
    window starts at the last token of the one before and sees none of the tokens before it."""
    windows, documents = [], 0
    for file_index in range(len(shards.files)):
        if shards.starts[file_index + 1] == shards.starts[file_index]:
            continue
        _, doc_ends = shards.open(file_index)
        begin = 0
        for end in doc_ends.tolist():
            if documents == num_documents:
                return windows
            documents += 1
            for window in range(begin, end - 1, seq_len):
                windows.append((file_index, window, min(window + seq_len + 1, end)))
            begin = end
    return windows


def pack_windows(lengths: list[int], token_budget: int) -> list[list[int]]:
    """Window indices grouped into rows of at most ``token_budget`` tokens, longest first."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    rows, row, total = [], [], 0
    for index in order:
        if row and total + lengths[index] > token_budget:
            rows.append(row)
            row, total = [], 0
        row.append(index)
        total += lengths[index]
    if row:
        rows.append(row)
    return rows


class DocumentDataset(StatefulIterableDataset):
    """Rows of ``row_len`` tokens that score every document of held-out shards on its own (`document_windows`). The
    windows are packed longest first, one segment per window with positions from 0; the rest of a row is unscored
    padding. Rows are dealt round-robin over the data ranks, and every rank yields the same number of rows, because
    under FSDP every forward is a collective: a rank short of a row repeats the first one, unscored. Full rows pass
    through the micro-batch packing unchanged."""

    def __init__(
        self, shards: TokenShards, seq_len: int, row_len: int, non_dp_size: int = 1, num_documents: int | None = None
    ):
        super().__init__(non_dp_size)
        self.shards = shards
        self.row_len = row_len
        self.windows = document_windows(shards, seq_len, num_documents)
        if not self.windows:
            raise ValueError("The held-out shards have no document of two or more tokens")
        self.rows = pack_windows([end - start - 1 for _, start, end in self.windows], row_len)

    def sample(self, row: list[int], scored: bool) -> dict:
        input_ids, target_ids, seq_lens = [], [], []
        for index in row:
            file_index, start, end = self.windows[index]
            ids = self.shards.open(file_index)[0][start:end].astype(np.int64).tolist()
            input_ids += ids[:-1]
            target_ids += ids[1:]
            seq_lens.append(len(ids) - 1)
        loss_mask = [scored] * len(input_ids)
        padding = self.row_len - len(input_ids)
        if padding > 0:
            # The row's own tokens, as in CatDataset: identical pad tokens would all route to the same MoE experts.
            input_ids += repeat_to_length(input_ids, padding)
            target_ids += [0] * padding
            loss_mask += [False] * padding
            seq_lens.append(padding)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": [position for length in seq_lens for position in range(length)],
            "loss_mask": loss_mask,
            "seq_lens": seq_lens,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }

    def __iter__(self):
        self._setup_world_info()
        mine = self.rows[self.data_rank :: self.data_world_size]
        for i in range(-(-len(self.rows) // self.data_world_size)):
            yield self.sample(mine[i], scored=True) if i < len(mine) else self.sample(self.rows[0], scored=False)


# Enough documents per folder to catch shards whose documents end differently, at two token reads each.
SEPARATOR_CHECK_DOCUMENTS = 16


def document_edges(shards: TokenShards, n: int) -> dict[Path, tuple[list[int], list[int]]]:
    """The first and the last tokens of the first ``n`` non-empty documents of each folder, read through the index."""
    edges: dict[Path, tuple[list[int], list[int]]] = {}
    for index, file in enumerate(shards.files):
        firsts, lasts = edges.setdefault(file.parent, ([], []))
        # Empty files have no documents.
        if len(lasts) >= n or shards.starts[index + 1] == shards.starts[index]:
            continue
        tokens, doc_ends = shards.open(index)
        lengths = np.diff(doc_ends, prepend=0)
        # Without empty documents: an empty first document ends at 0, and 0 - 1 would wrap around to the last token.
        ends = doc_ends[lengths > 0][: n - len(lasts)].astype(np.int64)
        firsts += tokens[ends - lengths[lengths > 0][: len(ends)].astype(np.int64)].tolist()
        lasts += tokens[ends - 1].tolist()
    return {folder: (firsts, lasts) for folder, (firsts, lasts) in edges.items() if lasts}


def special_token_ids(tokenizer: PreTrainedTokenizer) -> set[int]:
    """The tokenizer's special tokens, added ones included: the trainer sets pad to EOS, which drops Qwen's
    ``<|endoftext|>`` from ``all_special_ids``."""
    added = {token for token, added_token in tokenizer.added_tokens_decoder.items() if added_token.special}
    return set(tokenizer.all_special_ids) | added


def detect_separator(shards: TokenShards, tokenizer: PreTrainedTokenizer) -> int | None:
    """The special token all checked documents end with, the separator to mask. None when they start with BOS
    instead: the position that predicts a document's first token then has BOS as input, so there is nothing to mask.
    Anything else fails."""
    edges = document_edges(shards, SEPARATOR_CHECK_DOCUMENTS)
    ends = {token for _, lasts in edges.values() for token in lasts}
    if len(ends) == 1 and ends <= special_token_ids(tokenizer):
        return ends.pop()
    if all(set(firsts) == {tokenizer.bos_token_id} for firsts, _ in edges.values()):
        get_logger().info("The documents start with BOS and end without a separator: eod_mask_loss masks nothing")
        return None
    found = "; ".join(
        f"{folder}: " + ", ".join(f"{tokenizer.decode([token])!r} ({token})" for token in dict.fromkeys(lasts))
        for folder, (_, lasts) in edges.items()
    )
    raise ValueError(
        f"eod_mask_loss: the documents must end with one special token, the separator, but they end with {found}; "
        "tokenize with a separator or set data.eod_mask_loss = false"
    )


def warn_if_no_document_separator(shards: TokenShards, tokenizer: PreTrainedTokenizer) -> None:
    """Warn when the first non-empty document of a folder neither ends with a special token nor starts with BOS:
    prime-rl reads documents as they are, so shards tokenized without a separator run their documents together."""
    special = special_token_ids(tokenizer)
    for folder, ([first], [last]) in document_edges(shards, 1).items():
        if last in special or first == tokenizer.bos_token_id:
            continue
        get_logger().warning(
            f"The documents in {folder} have no separator: the first one ends with token {last} "
            f"({tokenizer.decode([last])!r}), not a special token, and does not start with BOS.\n"
            "The documents run together, so the model will not learn where a document ends and will not learn to "
            "stop.\n"
            "Tokenize again with datatrove's DocumentTokenizer and set eos_token to the tokenizer's end-of-text "
            "token (e.g. <|endoftext|> for Qwen)."
        )


def setup_token_dataset(
    config: DatatroveDataConfig, non_dp_size: int = 1, eod_mask_token: int | None = None
) -> TokenDataset:
    return TokenDataset(
        TokenShards(config.paths, seed=config.seed),
        seq_len=config.seq_len,
        document_masking=config.document_masking,
        non_dp_size=non_dp_size,
        eod_mask_token=eod_mask_token,
    )


def setup_document_dataset(
    config: DatatroveDataConfig, seq_len: int, non_dp_size: int = 1, num_documents: int | None = None
) -> DocumentDataset:
    return DocumentDataset(
        TokenShards(config.paths),
        seq_len=seq_len,
        row_len=config.seq_len * config.micro_batch_size,
        non_dp_size=non_dp_size,
        num_documents=num_documents,
    )


def length_label(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


# The 256 characters byte-level BPE tokenizers (GPT-2, Llama 3, Qwen, DeepSeek) write the bytes as.
BYTE_CHARS = {chr(c) for c in [*range(33, 127), *range(161, 173), *range(174, 256), *range(256, 324)]}


def token_byte_lengths(tokenizer: PreTrainedTokenizer) -> np.ndarray:
    """The UTF-8 bytes of every token id; special and added tokens count 0. Exact for byte-level BPE tokenizers."""
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)
    lengths = np.zeros(len(tokenizer), dtype=np.int64)
    for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
        if token is None or token_id in special:
            continue
        if set(token) <= BYTE_CHARS:
            lengths[token_id] = len(token)
        else:
            lengths[token_id] = len(tokenizer.convert_tokens_to_string([token]).encode())
    return lengths
