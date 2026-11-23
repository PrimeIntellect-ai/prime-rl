"""Per-tensor sha256 of safetensors files, using only the standard library."""

import hashlib
import json
import struct
import sys
from pathlib import Path

CHUNK_BYTES = 64 * 1024 * 1024


def hash_safetensors_file(path: Path) -> dict[str, dict]:
    tensors = {}
    with path.open("rb") as file:
        (header_len,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            file.seek(data_start + begin)
            digest = hashlib.sha256()
            remaining = end - begin
            while remaining:
                chunk = file.read(min(CHUNK_BYTES, remaining))
                digest.update(chunk)
                remaining -= len(chunk)
            tensors[name] = {"dtype": info["dtype"], "shape": info["shape"], "sha256": digest.hexdigest()}
    return tensors


def hash_safetensors_dir(directory: Path) -> dict[str, dict]:
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        for name, info in hash_safetensors_file(path).items():
            if name in tensors:
                raise ValueError(f"{name} appears in more than one file of {directory}")
            tensors[name] = info
    return tensors


if __name__ == "__main__":
    json.dump(hash_safetensors_dir(Path(sys.argv[1])), sys.stdout, indent=1, sort_keys=True)
