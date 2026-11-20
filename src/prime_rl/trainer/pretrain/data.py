"""Pretraining data: units of the documents in datatrove tokenized shards, sampled from one or more sources into rows of
``seq_len`` tokens."""

import hashlib
import os
import resource
import socket
import tempfile
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch.distributed as dist
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.pretrain import PretrainDataConfig
from prime_rl.trainer.data import SKIP_DATALOADER, StatefulIterableDataset, due, mixture_changed
from prime_rl.utils.bin_packing import fill_rows
from prime_rl.utils.logger import get_logger

# Every step looks up documents in files of every source, so files stay open: up to the soft limit on open files (one
# descriptor each, some left for the rest), and at most MAX_OPEN_FILES (one memory map each; Linux allows 65530 per
# process by default). The least recently used are closed.
MAX_OPEN_FILES = 32768
# The dtype of the document indexes' end offsets.
INDEX = np.dtype("<u8")
# The memory maps of a process's token and index files, by path, shared by all its TokenShards and opened on first use.
OPEN_FILES: OrderedDict[str, np.ndarray] = OrderedDict()


def memmap(path: str, dtype: np.dtype) -> np.ndarray:
    if path in OPEN_FILES:
        OPEN_FILES.move_to_end(path)
    else:
        # A plain array over the map: indexing it costs a fraction of indexing an np.memmap.
        OPEN_FILES[path] = np.memmap(path, dtype=dtype, mode="r").view(np.ndarray)
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if len(OPEN_FILES) > min(MAX_OPEN_FILES, max(64, soft_limit - 1024)):
            OPEN_FILES.popitem(last=False)
    return OPEN_FILES[path]


# The cursor's lists of a count per source.
SOURCE_COUNTS = ("units", "tokens", "mix_start", "pieces", "piece_tokens")
# Keys the order that subsets take units in apart from the order of the passes: any constant other than 0 does.
SUBSET_KEY = 0xD1B54A32D192ED03


def save(path: Path, array: np.ndarray) -> None:
    """Written whole, then renamed: ranks that write the same file at once never read a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    with os.fdopen(descriptor, "wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def raise_open_files_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64's output function, increment included (Steele, Lea and Flood, "Fast splittable pseudorandom number
    generators", 2014; constants from Vigna's reference implementation): a bijective hash of uint64 values."""
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def permute(indices: np.ndarray, n: np.ndarray | int, key: np.ndarray | int) -> np.ndarray:
    """A pseudo-random permutation of ``range(n)``, chosen by ``key``, at ``indices`` (``n`` and ``key`` may differ per
    index): a 4-round Feistel network on the smallest even number of bits that holds ``n``, applied again to values of
    ``n`` or more until they fall below (cycle walking; Black and Rogaway, "Ciphers with arbitrary finite domains", 2002).

    A stored permutation (``rng.permutation(n)``) costs 8 bytes per unit in every dataloader worker: fine at 10B tokens,
    tens of GB at 10T. This one builds no table, and a unit's position is a pure function of (key, index)."""
    values = np.asarray(indices, dtype=np.uint64).copy()
    n = np.broadcast_to(np.asarray(n, dtype=np.uint64), values.shape)
    # frexp's exponent of a positive integer below 2**53 is its bit length.
    half = np.maximum(1, (np.frexp((n - np.uint64(1)).astype(np.float64))[1] + 1) // 2).astype(np.uint64)
    mask = (np.uint64(1) << half) - np.uint64(1)
    round_keys = mix(np.asarray(key, dtype=np.uint64)[..., None] + np.arange(4, dtype=np.uint64))
    round_keys = np.broadcast_to(round_keys, (*values.shape, 4))
    walking, current = np.arange(len(values)), values
    while len(walking):
        left, right = current >> half, current & mask
        for round_key in round_keys.T:
            left, right = right, left ^ (mix(right ^ round_key) & mask)
        current = (left << half) | right
        done = current < n
        values[walking[done]] = current[done]
        walking, current, n, half, mask, round_keys = (x[~done] for x in (walking, current, n, half, mask, round_keys))
    return values.astype(np.int64)


def token_dtype(file: Path) -> np.dtype:
    first_line = Path(f"{file}.metadata").read_text().splitlines()[0]
    return np.dtype(f"<u{int(first_line.rsplit('|', 1)[1])}")


class TokenShards:
    """Datatrove tokenized shards in one or more folders, all their non-empty files in folder order, then file name
    order.

    ``<name>.ds`` holds the token ids (little-endian, 2 or 4 bytes as recorded on the first line of
    ``<name>.ds.metadata``) and ``<name>.ds.index`` the uint64 end offset of every document. File boundaries are
    document boundaries. The documents are numbered in file order (``doc_offsets``: the number of the first document of
    every file)."""

    def __init__(self, paths: list[Path]):
        self.paths, self.files = paths, []
        for path in paths:
            files = sorted(path.glob("*.ds"))
            if not files:
                raise FileNotFoundError(f"No datatrove .ds files in {path}")
            # Without empty files, which hold no tokens (and np.memmap can't map).
            self.files += [file for file in files if file.stat().st_size]
        self.dtypes = [token_dtype(file) for file in self.files]
        sizes = [file.stat().st_size // dtype.itemsize for file, dtype in zip(self.files, self.dtypes)]
        self.index_stats = [os.stat(f"{file}.index") for file in self.files]
        self.starts = np.cumsum([0, *sizes])
        self.num_tokens = int(self.starts[-1])
        self.doc_offsets = np.cumsum([0, *(stat.st_size // 8 for stat in self.index_stats)])
        self.num_documents = int(self.doc_offsets[-1])

    @classmethod
    def join(cls, parts: list["TokenShards"]) -> "TokenShards":
        """The files of ``parts`` as one TokenShards, without reading the folders again."""
        joined = cls.__new__(cls)
        for name in ("paths", "files", "dtypes", "index_stats"):
            setattr(joined, name, [item for part in parts for item in getattr(part, name)])
        joined.starts = np.cumsum([0, *(n for part in parts for n in np.diff(part.starts))])
        joined.num_tokens = int(joined.starts[-1])
        joined.doc_offsets = np.cumsum([0, *(n for part in parts for n in np.diff(part.doc_offsets))])
        joined.num_documents = int(joined.doc_offsets[-1])
        return joined

    def tokens(self, file_index: int) -> np.ndarray:
        return memmap(str(self.files[file_index]), self.dtypes[file_index])

    def index(self, file_index: int) -> np.ndarray:
        """The file's document ends."""
        return memmap(f"{self.files[file_index]}.index", INDEX)

    def long_documents(self, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """The documents longer than ``seq_len`` (their numbers) and their pieces of ``seq_len`` tokens beyond the
        first, read from every file's document index."""
        found = []
        for file_index, file in enumerate(self.files):
            # Read once, so not kept open.
            doc_ends = np.memmap(f"{file}.index", dtype="<u8", mode="r")
            # In chunks of 2^24 documents, so the lengths never take more than 128 MB.
            for begin in range(0, len(doc_ends), 1 << 24):
                ends = doc_ends[begin : begin + (1 << 24)].astype(np.int64)
                lengths = np.diff(ends, prepend=int(doc_ends[begin - 1]) if begin else 0)
                long = np.flatnonzero(lengths > seq_len)
                found.append(np.stack([self.doc_offsets[file_index] + begin + long, (lengths[long] - 1) // seq_len]))
        documents, pieces = np.concatenate([np.empty((2, 0), dtype=np.int64), *found], axis=1)
        return documents, pieces


class Step(NamedTuple):
    """A step's units in the order they were drawn, the cut one cut (``units``: a unit's source, file, begin, end,
    doc_begin and doc_end, the tokens ``[begin, end)`` of the file in its document ``[doc_begin, doc_end)``), and its
    rows: every row's pieces (``pieces`` ``[row_starts[r], row_starts[r + 1])``), each a unit, an offset in it and a
    length."""

    units: np.ndarray
    pieces: np.ndarray
    row_starts: np.ndarray

    def row_segments(self, row: int) -> np.ndarray:
        """The row's pieces as units, one per line."""
        unit, offset, length = self.pieces[self.row_starts[row] : self.row_starts[row + 1]].T
        segments = self.units[unit]
        segments[:, 2] += offset
        segments[:, 3] = segments[:, 2] + length
        return segments


class TokenSampler:
    """Plans every step's ``batch_size`` rows of ``seq_len`` tokens from a cursor of a few counters (`start`, `plan`).

    A unit is a document or a ``seq_len`` piece of a longer one. A source reads its units (or, with ``repetitions``, a
    fixed seeded subset) in passes, each in a keyed order (`permute`). Units are taken in order of due time, their
    source's tokens since the mixture last changed over its weight (weighted fair queuing); the unit that crosses the
    step's end is cut, and `fill_rows` packs the step into full rows, each in a seeded order. A resumed run can change
    the mixture (`resume`). ``cache_dir`` keeps the long-document table and the subsets."""

    def __init__(
        self,
        names: list[str],
        sources: list[TokenShards],
        weights: list[float],
        repetitions: list[float | None],
        batch_size: int,
        seq_len: int,
        seed: int = 0,
        max_steps: int | None = None,
        cache_dir: Path | None = None,
    ):
        self.names, self.sources = names, sources
        self.batch_size, self.seq_len, self.seed = batch_size, seq_len, seed
        # The files of all sources, numbered in source order, and every source's first.
        self.shards = TokenShards.join(sources)
        self.first_files = np.cumsum([0, *(len(shards.files) for shards in sources)])[:-1]
        self.num_documents = np.array([shards.num_documents for shards in sources], dtype=np.int64)
        self.num_tokens = np.array([shards.num_tokens for shards in sources], dtype=np.int64)
        # The number of every source's first document, and of every file's, in the documents of all sources.
        self.source_documents = np.cumsum([0, *self.num_documents])
        self.file_documents = self.shards.doc_offsets[:-1]
        self.long_documents, self.pieces_before = self.long_document_table(cache_dir)
        # The further pieces of all sources before every source's first.
        self.source_pieces = self.pieces_before[np.searchsorted(self.long_documents, self.source_documents)]
        self.num_units = self.num_documents + np.diff(self.source_pieces)
        self.keys = mix(np.uint64(seed) + np.array([zlib.crc32(name.encode()) for name in names], dtype=np.uint64))
        self.subset_keys = mix(self.keys ^ np.uint64(SUBSET_KEY))
        self.offered = (None, None)
        self.weights = np.array(weights) / sum(weights)
        # Every source's tokens over the run.
        self.run_tokens = None if max_steps is None else max_steps * batch_size * seq_len * self.weights
        # The units and tokens of every source's (sub)set.
        self.subsets = np.array([times is not None for times in repetitions])
        self.set_units, self.set_tokens = self.num_units.copy(), self.num_tokens.copy()
        for source in np.flatnonzero(self.subsets):
            if self.run_tokens[source] > 0:
                tokens = self.run_tokens[source] / repetitions[source]
                self.set_units[source], self.set_tokens[source] = self.subset(source, tokens, cache_dir)

    def long_document_table(self, cache_dir: Path | None) -> tuple[np.ndarray, np.ndarray]:
        """The long documents of all sources (`TokenShards.long_documents`), by their numbers in the documents of all
        sources, and the further pieces of all sources before each one's (and in all, at the end). With ``cache_dir``,
        the table is kept there under a hash of ``seq_len`` and every index file's path, size and modification time:
        a later start (or rank) maps it instead of reading the indexes again, and changed folders get a new one."""
        if cache_dir is not None:
            cache = cache_dir / f"long-documents-{self.cache_key(range(len(self.sources)))}.npy"
            if cache.exists():
                table = np.load(cache, mmap_mode="r")
                return table[0, :-1], table[1]
        tables = [shards.long_documents(self.seq_len) for shards in self.sources]
        documents = np.concatenate([start + docs for start, (docs, _) in zip(self.source_documents, tables)])
        pieces_before = np.cumsum([0, *np.concatenate([pieces for _, pieces in tables])])
        if cache_dir is not None:
            save(cache, np.stack([np.r_[documents, 0], pieces_before]))
        return documents, pieces_before

    def cache_key(self, sources, *extra) -> str:
        """A hash of ``seq_len``, every index file's path, size and modification time of ``sources``, and ``extra``."""
        stats = [
            (str(file.absolute()), stat.st_size, stat.st_mtime_ns)
            for source in sources
            for file, stat in zip(self.sources[source].files, self.sources[source].index_stats)
        ]
        return hashlib.sha256(repr([self.seq_len, stats, *extra]).encode()).hexdigest()[:24]

    def subset(self, source: int, tokens: float, cache_dir: Path | None) -> tuple[int, int]:
        """The units and tokens of the source's subset of ``tokens`` tokens: the first units of an order drawn from
        ``seed`` and the source's name, until they hold ``tokens`` (one unit more at most)."""
        name = self.names[source]
        if tokens > self.num_tokens[source]:
            raise ValueError(
                f"Source {name}: its share x the run's tokens / repetitions needs {tokens:,.0f} unique tokens but has "
                f"{self.num_tokens[source]:,}; raise repetitions, lower the weight, or train fewer steps"
            )
        if tokens < self.seq_len:
            raise ValueError(
                f"Source {name}: its subset would hold {tokens:,.1f} tokens, fewer than seq_len ({self.seq_len}); "
                "lower repetitions, raise the weight, or train more steps"
            )
        if cache_dir is not None:
            cache = cache_dir / f"subset-{self.cache_key([source], self.seed, name, round(tokens))}.npy"
            if cache.exists():
                units, set_tokens = np.load(cache).tolist()
                return units, set_tokens
        units = set_tokens = 0
        while set_tokens < tokens:
            # In chunks of 2^20 units of the order.
            order = np.arange(units, min(units + (1 << 20), self.num_units[source]))
            chunk = permute(order, self.num_units[source], self.subset_keys[source])
            cumulative = set_tokens + np.cumsum(self.unit_lengths(np.full_like(chunk, source), chunk))
            taken = min(int(np.searchsorted(cumulative, tokens)) + 1, len(order))
            units, set_tokens = units + taken, int(cumulative[taken - 1])
        if cache_dir is not None:
            save(cache, np.array([units, set_tokens]))
        return units, set_tokens

    def unit_lengths(self, sources: np.ndarray, units: np.ndarray) -> np.ndarray:
        """The tokens of units ``units`` of sources ``sources``, read from the document indexes file by file."""
        files, local, pieces = self.documents(sources, units)
        lengths = np.empty(len(units), dtype=np.int64)
        order = np.argsort(files, kind="stable")
        bounds = np.flatnonzero(np.diff(files[order])) + 1
        for chosen in np.split(order, bounds):
            doc_ends = self.shards.index(files[chosen[0]])
            ends = doc_ends[local[chosen]].astype(np.int64)
            begins = np.where(local[chosen] > 0, doc_ends[np.maximum(local[chosen] - 1, 0)].astype(np.int64), 0)
            lengths[chosen] = np.minimum(ends - begins - pieces[chosen] * self.seq_len, self.seq_len)
        return lengths

    def start(self) -> dict:
        return {key: [0] * len(self.sources) for key in SOURCE_COUNTS} | {"rows": 0, "rest": None}

    def documents(self, sources: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The file, document in the file and piece of units ``units`` of sources ``sources``."""
        documents = self.source_documents[sources] + units
        pieces = np.zeros_like(units)
        further = units >= self.num_documents[sources]
        piece = self.source_pieces[sources[further]] + units[further] - self.num_documents[sources[further]]
        long = np.searchsorted(self.pieces_before, piece, side="right") - 1
        documents[further] = self.long_documents[long]
        pieces[further] = 1 + piece - self.pieces_before[long]
        files = np.searchsorted(self.file_documents, documents, side="right") - 1
        return files, documents - self.file_documents[files], pieces

    def locate(self, sources: np.ndarray, reads: np.ndarray) -> np.ndarray:
        """The unit that every read ``reads`` (counted over all passes) of sources ``sources`` reads, one per line (as
        `Step.units`)."""
        passes, index = np.divmod(reads, self.set_units[sources])
        units = permute(index, self.set_units[sources], mix(self.keys[sources] + passes.astype(np.uint64)))
        subset = self.subsets[sources]
        if subset.any():
            units[subset] = permute(units[subset], self.num_units[sources[subset]], self.subset_keys[sources[subset]])
        files, local, pieces = self.documents(sources, units)
        files, local = files.tolist(), local.tolist()
        indexes = {file: self.shards.index(file) for file in set(files)}
        doc_ends = np.array([indexes[file][doc] for file, doc in zip(files, local)], dtype=np.int64)
        doc_begins = np.array([indexes[file][doc - 1] if doc else 0 for file, doc in zip(files, local)], np.int64)
        begins = doc_begins + pieces * self.seq_len
        ends = np.minimum(begins + self.seq_len, doc_ends)
        return np.stack([sources, files, begins, ends, doc_begins, doc_ends], axis=1)

    def plan(self, cursor: dict) -> tuple["Step", dict]:
        """The step that starts at ``cursor``, and the cursor at the start of the next step."""
        rows, weights = cursor["rows"], self.weights
        reads, tokens, mix_start = (np.array(cursor[key]) for key in ("units", "tokens", "mix_start"))
        # The cursor numbers the rest's file within its source.
        rest = np.array(cursor["rest"] or [], dtype=np.int64).reshape(-1, 6)
        rest[:, 1] += self.first_files[rest[:, 0]]
        np.add.at(tokens, rest[:, 0], rest[:, 3] - rest[:, 2])
        budget = self.batch_size * self.seq_len - int((rest[:, 3] - rest[:, 2]).sum())
        # The sources' next units by due time: a unit's source's tokens before it since the mixture last changed,
        # divided by its weight (ties to the first source); sources without weight have none. Every source offers about
        # twice its share of the step, and 4 more units for sources whose share is a unit or two; twice as many if the
        # step would take all of them. These numbers only set how often a step has to locate more units, not which units
        # it takes.
        sources = np.flatnonzero(weights > 0)
        wanted = np.ceil(2.0 * weights[sources] * budget * self.set_units[sources] / self.set_tokens[sources])
        wanted = wanted.astype(np.int64) + 4
        # The units offered but not taken in the plan before, if this one reads on from it.
        units_before, offered = self.offered
        if units_before != cursor["units"]:
            offered = np.empty((0, 6), dtype=np.int64)
        while True:
            have = np.bincount(offered[:, 0], minlength=len(weights))[sources]
            new = np.maximum(wanted - have, 0)
            firsts = np.repeat(reads[sources] + have, new)
            within = np.arange(new.sum()) - np.repeat(np.cumsum(new) - new, new)
            offered = np.concatenate([offered, self.locate(np.repeat(sources, new), firsts + within)])
            offered = offered[np.argsort(offered[:, 0], kind="stable")]
            owners, lengths = offered[:, 0], offered[:, 3] - offered[:, 2]
            counts = have + new
            before = np.cumsum(lengths) - lengths
            before -= np.repeat(before[np.cumsum(counts) - counts], counts)
            order = np.argsort(due(tokens[owners] + before, mix_start[owners], weights[owners]), kind="stable")
            ends = np.cumsum(lengths[order])
            taken = order[: np.searchsorted(ends, budget) + 1]
            exhausted = np.bincount(owners[taken], minlength=len(weights))[sources] == counts
            if not exhausted.any():
                break
            wanted = np.where(exhausted, 2 * counts, counts)
        reads += np.bincount(owners[taken], minlength=len(weights))
        self.offered = (reads.tolist(), np.delete(offered, taken, axis=0))
        # The last unit is cut at the step's end; its rest starts the next step. Empty units take no place.
        chosen = offered[taken]
        cut = int(ends[len(taken) - 1]) - budget
        next_rest = chosen[-1].copy()
        next_rest[2] = next_rest[3] - cut
        next_rest[1] -= self.first_files[next_rest[0]]
        chosen[-1, 3] -= cut
        chosen = chosen[chosen[:, 3] > chosen[:, 2]]
        np.add.at(tokens, chosen[:, 0], chosen[:, 3] - chosen[:, 2])
        units = np.concatenate([rest, chosen])
        pieces = fill_rows(units[:, 3] - units[:, 2], self.batch_size, self.seq_len)
        # The pieces of every row in a random order.
        shuffled = pieces[np.lexsort((np.random.default_rng([self.seed, rows, 1]).random(len(pieces)), pieces[:, 0]))]
        row_starts = np.searchsorted(shuffled[:, 0], np.arange(self.batch_size + 1))
        # Every source's pieces in the step's rows and their tokens, for the step's metrics.
        piece_sources = units[pieces[:, 1], 0]
        cursor = {
            "rows": rows + self.batch_size,
            "units": reads.tolist(),
            "tokens": tokens.tolist(),
            "mix_start": cursor["mix_start"],
            "rest": next_rest.tolist() if cut else None,
            "pieces": np.bincount(piece_sources, minlength=len(weights)).tolist(),
            "piece_tokens": np.bincount(piece_sources, pieces[:, 3], minlength=len(weights)).astype(int).tolist(),
        }
        return Step(units, shuffled[:, 1:], row_starts), cursor

    def progress(self, cursor: dict) -> dict[str, float]:
        """Every source's tokens read, the unique tokens among them (at most its set's), their repetitions, and the
        mean length of its pieces in the rows of the step before ``cursor``; and the same over all sources."""
        unique = [min(tokens, limit) for tokens, limit in zip(cursor["tokens"], self.set_tokens.tolist())]
        metrics = {}
        for name, tokens, unique_tokens, count, piece_tokens in zip(
            self.names, cursor["tokens"], unique, cursor["pieces"], cursor["piece_tokens"]
        ):
            metrics[f"progress/{name}/num_tokens"] = tokens
            metrics[f"progress/{name}/unique_tokens"] = unique_tokens
            metrics[f"progress/{name}/repetitions"] = tokens / unique_tokens if unique_tokens else 0.0
            if count:
                metrics[f"progress/{name}/mean_segment_length"] = piece_tokens / count
        metrics["progress/unique_tokens"] = sum(unique)
        metrics["progress/repetitions"] = sum(cursor["tokens"]) / sum(unique) if sum(unique) else 0.0
        if sum(cursor["pieces"]):
            metrics["progress/mean_segment_length"] = sum(cursor["piece_tokens"]) / sum(cursor["pieces"])
        return metrics

    def inputs(self) -> dict:
        """What the plan depends on, for the ranks to compare and a resumed run to match (`mixture_changed`): every
        source's weight, its files with their tokens and documents, and the tokens of its subset (None without
        repetitions)."""
        files = [
            hashlib.sha256(
                repr([[f.name for f in s.files], np.diff(s.starts).tolist(), np.diff(s.doc_offsets).tolist()]).encode()
            ).hexdigest()[:24]
            for s in self.sources
        ]
        subsets = [int(tokens) if subset else None for tokens, subset in zip(self.set_tokens, self.subsets)]
        return {
            "seq_len": self.seq_len,
            "seed": self.seed,
            "sources": {
                name: {"weight": weight, "files": digest, "subset": subset}
                for name, weight, digest, subset in zip(self.names, self.weights.tolist(), files, subsets)
            },
        }

    def resume(self, cursor: dict, saved: dict, changed: bool) -> dict:
        """The cursor of a checkpoint whose sources were ``saved`` (`inputs`), for these sources, matched by name: a
        kept source reads on where it stopped, in the same order; an added one starts at its beginning; a removed one
        drops out, the rest of its unit cut at the step's end too. If the mixture ``changed``, the due times count from
        here."""
        names = list(saved)
        counts = {key: dict(zip(names, cursor[key])) for key in SOURCE_COUNTS}
        resumed = {key: [counts[key].get(name, 0) for name in self.names] for key in SOURCE_COUNTS}
        if changed:
            resumed["mix_start"] = list(resumed["tokens"])
        rest = cursor["rest"]
        kept = rest is not None and names[rest[0]] in self.names
        resumed["rest"] = [self.names.index(names[rest[0]]), *rest[1:]] if kept else None
        resumed["rows"] = cursor["rows"]
        return resumed


class TokenDataset(StatefulIterableDataset):
    """The rows of the sampler's steps: every rank plans every step, and data rank ``r`` of ``R`` yields rows ``r``,
    ``r + R``, ... of each. The state is the cursor at the start of the current step and the rows of it yielded; at a
    step boundary it is the same on every rank, so a resumed run reads the same rows on any number of GPUs.
    ``eod_mask_token`` (set by the trainer) has no loss where it is the input."""

    def __init__(
        self,
        sampler: TokenSampler,
        document_masking: bool = True,
        num_workers: int = 1,
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        super().__init__(dp_rank, dp_size)
        self.sampler = sampler
        self.document_masking = document_masking
        self.eod_mask_token = None
        self.cursor, self.rows_done = sampler.start(), 0
        # A worker's state holds the rows it yielded, so the number of workers must stay the same.
        self.inputs = sampler.inputs() | {"num_workers": num_workers}

    def state_dict(self) -> dict:
        epoch = sum(self.cursor["tokens"]) / self.sampler.shards.num_tokens
        return {
            "step": self.cursor["rows"],
            "epoch": epoch,
            "cursor": self.cursor,
            "rows_done": self.rows_done,
            "plan": self.inputs,
        }

    def load_state_dict(self, state_dict: dict):
        if "cursor" not in state_dict:
            raise ValueError(
                f"The checkpoint's data state is not the source sampler's, so the run can't resume it. {SKIP_DATALOADER}"
            )
        if "mix_start" not in state_dict["cursor"]:
            raise ValueError(
                "The checkpoint's data state is from before the sources could change on resume, so the run can't "
                f"resume it. {SKIP_DATALOADER}"
            )
        plan = state_dict["plan"]
        if changed := [key for key in ("seq_len", "seed", "num_workers") if plan.get(key) != self.inputs[key]]:
            raise ValueError(
                "The data settings differ from the checkpoint's: "
                + "; ".join(f"{key} {plan.get(key)!r} -> {self.inputs[key]!r}" for key in changed)
                + f". Resume with the same ones. {SKIP_DATALOADER}"
            )
        # Every dataloader worker of every rank loads the state; data rank 0 logs a change.
        self._setup_world_info()
        changed = mixture_changed(plan["sources"], self.inputs["sources"], log=self.data_rank == 0)
        self.cursor = self.sampler.resume(state_dict["cursor"], plan["sources"], changed)
        self.rows_done = state_dict["rows_done"]

    def sample(self, row: np.ndarray) -> dict:
        """The sample of a row's segments (`Step.row_segments`)."""
        _, files, begins, ends, doc_begins, doc_ends = row.T
        lengths = ends - begins
        # Every segment's tokens, then its document's next token if the document continues.
        more = ends < doc_ends
        maps = [self.sampler.shards.tokens(file) for file in files.tolist()]
        read = zip(maps, begins.tolist(), (ends + more).tolist())
        tokens = np.concatenate([file_tokens[begin:end] for file_tokens, begin, end in read], dtype=np.int64)
        nexts = np.cumsum(lengths + more)[more] - 1
        input_ids = np.delete(tokens, nexts)
        target_ids = np.append(input_ids[1:], 0)
        lasts = np.cumsum(lengths) - 1
        target_ids[lasts[more]] = tokens[nexts]
        # At a document's end a segment predicts the next segment's first token if that starts a document, else nothing.
        unscored = lasts[~more & np.append(begins[1:] != doc_begins[1:], True)]
        target_ids[unscored] = 0
        loss_mask = np.ones(len(input_ids), dtype=bool)
        loss_mask[unscored] = False
        if self.eod_mask_token is not None:
            loss_mask &= input_ids != self.eod_mask_token
        seq_lens = lengths.tolist() if self.document_masking else [self.sampler.seq_len]
        positions = np.arange(self.sampler.seq_len) - np.repeat(np.cumsum(seq_lens) - seq_lens, seq_lens)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": positions,
            "loss_mask": loss_mask,
            "seq_lens": seq_lens,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }

    def __iter__(self):
        self._setup_world_info()
        batch_size = self.sampler.batch_size
        while True:
            step, after = self.sampler.plan(self.cursor)
            mine = range(self.data_rank, batch_size, self.data_world_size)
            for i in range(self.rows_done, len(mine)):
                sample = self.sample(step.row_segments(mine[i]))
                if i == len(mine) - 1:
                    self.cursor, self.rows_done = after, 0
                else:
                    self.rows_done = i + 1
                yield sample


# Enough documents per folder to catch shards whose documents end differently, at two token reads each.
SEPARATOR_CHECK_DOCUMENTS = 16


def document_edges(shards: TokenShards, n: int) -> dict[Path, tuple[list[int], list[int]]]:
    """The first and the last tokens of the first ``n`` non-empty documents of each folder, read through the index."""
    edges: dict[Path, tuple[list[int], list[int]]] = {}
    for index, file in enumerate(shards.files):
        firsts, lasts = edges.setdefault(file.parent, ([], []))
        if len(lasts) >= n:
            continue
        tokens, doc_ends = shards.tokens(index), shards.index(index)
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
    config: PretrainDataConfig,
    *,
    dp_rank: int,
    dp_size: int,
    cache_dir: Path | None = None,
    max_steps: int | None = None,
) -> TokenDataset:
    raise_open_files_limit()
    sources = [TokenShards([source.path]) for source in config.source]
    for source, shards in zip(config.source, sources):
        if shards.num_tokens == 0:
            raise ValueError(f"Source {source.name} ({source.path}) has no tokens")
    # The sources' weights, or else their natural shares.
    weights = [s.weight if s.weight is not None else shards.num_tokens for s, shards in zip(config.source, sources)]
    repetitions = [source.repetitions for source in config.source]
    sampler = TokenSampler(
        [source.name for source in config.source],
        sources,
        weights,
        repetitions,
        config.batch_size,
        config.seq_len,
        config.seed,
        max_steps,
        cache_dir,
    )
    for index, (name, shards) in enumerate(zip(sampler.names, sources)):
        set_tokens = int(sampler.set_tokens[index])
        times = sampler.run_tokens[index] / set_tokens if sampler.run_tokens is not None else None
        get_logger().info(
            f"Source {name}: share {sampler.weights[index]:.4g}, {shards.num_tokens:,} tokens, "
            f"{set_tokens:,} used, repetitions {f'{times:.3g}' if times is not None else '(no max_steps)'}"
            f"{' (given)' if repetitions[index] is not None else ''}"
        )
    return TokenDataset(sampler, config.document_masking, config.num_workers, dp_rank=dp_rank, dp_size=dp_size)


def check_ranks_see_same_data(sampler: TokenSampler) -> None:
    """Raise unless every rank of the default group sees the same plan inputs (`TokenSampler.inputs`) and folders: a
    node that sees other files, tokens or documents would plan other steps and train on rows that overlap or miss the
    others'."""
    view = repr([sampler.inputs(), [str(path) for path in sampler.shards.paths]])
    seen = [None] * dist.get_world_size()
    dist.all_gather_object(seen, (hashlib.sha256(view.encode()).hexdigest(), socket.gethostname()))
    views = {}
    for rank, (digest, host) in enumerate(seen):
        views.setdefault(digest, []).append(f"{rank} ({host})")
    if len(views) > 1:
        raise RuntimeError(
            "Ranks see different data (files, tokens or documents of the sources) and would plan different steps. "
            f"Each group sees one view: {' | '.join('ranks ' + ', '.join(ranks) for ranks in views.values())}. "
            "Likely a folder is still being written, or a node has a stale NFS view of it."
        )
