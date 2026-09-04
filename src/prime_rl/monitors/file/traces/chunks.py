"""Chunked JSONL — an append-only stream split into numbered files, sealed as it grows.

    stream/00000.jsonl.zst   sealed: zstd in seekable frames, so a reader still lands on one line
    stream/00001.jsonl       live: plain text, so it can be tailed

A line never spans chunks. Once a chunk reaches its size the writer moves on and seals it
in the background: the compressed twin is written beside it and the plain file removed
only once the twin is complete, so a reader that misses the plain file finds the sealed
one. ``zstd -dcf stream/* | jq`` reads a whole stream, sealed and live alike.
"""

import shutil
import threading
from pathlib import Path
from typing import BinaryIO

import pyzstd

LEVEL = 3
"""zstd level: ~7x on episode records at hundreds of MB/s, so sealing keeps up with a run."""

FRAME_BYTES = 4 << 20
"""Independent frames in a sealed chunk: a seek decompresses one of these, not the chunk."""


def chunk_path(directory: Path, number: int) -> Path:
    return directory / f"{number:05d}.jsonl"


def chunk_numbers(directory: Path) -> dict[int, bool]:
    """Chunk number -> whether its plain file is present (a sealed chunk has none)."""
    numbers: dict[int, bool] = {}
    for path in directory.glob("*.jsonl*"):
        number = int(path.name.split(".", 1)[0])
        numbers[number] = numbers.get(number, False) or path.suffix == ".jsonl"
    return numbers


def open_chunk(directory: Path, number: int) -> BinaryIO:
    """A chunk for seeking reads, whether still plain or already sealed."""
    path = chunk_path(directory, number)
    try:
        return open(path, "rb")
    except FileNotFoundError:
        return pyzstd.SeekableZstdFile(path.with_name(path.name + ".zst"), "rb")


def seal(path: Path) -> None:
    """Compress a finished chunk beside itself, then drop the plain file."""
    target = path.with_name(path.name + ".zst")
    partial = target.with_name(target.name + ".tmp")
    with (
        open(path, "rb") as src,
        pyzstd.SeekableZstdFile(partial, "wb", level_or_option=LEVEL, max_frame_content_size=FRAME_BYTES) as dst,
    ):
        shutil.copyfileobj(src, dst, 1 << 20)
    partial.replace(target)
    path.unlink()


class ChunkedJsonl:
    """Appends lines to the live chunk of a stream and reports where each one landed."""

    def __init__(self, directory: Path, max_bytes: int, compress: bool) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.max_bytes = max_bytes
        self.compress = compress
        self._sealing: list[threading.Thread] = []
        numbers = chunk_numbers(directory)
        last = max(numbers, default=-1)
        # A relaunch appends to the live chunk it finds and starts a new one after a sealed
        # last chunk; anything else still plain was left mid-seal by a crash.
        self.number = last if numbers.get(last) else last + 1
        for number, plain in numbers.items():
            if plain and number != self.number and compress:
                self._seal(number)
        self.file = open(chunk_path(directory, self.number), "ab")
        self.size = self.file.tell()

    def append(self, line: bytes) -> tuple[int, int]:
        """Write one line; returns ``(chunk, offset)``, the address a reader seeks to."""
        if self.size and self.size + len(line) > self.max_bytes:
            self._roll()
        offset = self.size
        self.file.write(line)
        self.size += len(line)
        return self.number, offset

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        """Seal the live chunk too, so a finished run keeps nothing plain."""
        self.file.close()
        if not self.size:
            chunk_path(self.directory, self.number).unlink()
        elif self.compress:
            self._seal(self.number)
        for thread in self._sealing:
            thread.join()

    def _roll(self) -> None:
        self.file.close()
        if self.compress:
            self._seal(self.number)
        self.number += 1
        self.file = open(chunk_path(self.directory, self.number), "ab")
        self.size = 0

    def _seal(self, number: int) -> None:
        # not a daemon: a process that exits mid-seal waits for it rather than leaving a torn twin
        thread = threading.Thread(target=seal, args=(chunk_path(self.directory, number),))
        thread.start()
        self._sealing.append(thread)
