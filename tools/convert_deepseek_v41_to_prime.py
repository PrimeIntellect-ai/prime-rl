"""Convert the published DeepSeek-V4.1 checkpoint (fp8 / fp4) into a bf16 PrimeRL-format checkpoint.

The trainer can convert a checkpoint itself, but it loads the whole state dict on one rank first,
and V4.1 Flash is ~1.5 TB in bf16 (two ~98B-parameter engram tables alone are ~400 GB). This
streams instead: one decoder layer at a time, dequantized on the GPU and written straight to its
own safetensors file, with each engram table dequantized and written in row chunks so it never
sits in host memory whole. Dequantization is exact: every fp8 / fp4 value times its power-of-two
scale is representable in bf16.

The output directory is a drop-in `model.name`: it gets the PrimeRL-named weights, an index,
`config.json` without `quantization_config`, and the tokenizer files the engram hash reads.

Usage (from the prime-rl repo, ideally one process per GPU):
    uv run python tools/convert_deepseek_v41_to_prime.py <snapshot_dir> <output_dir> [--worker K --num-workers N]

Run every worker, then once more with `--finalize` to write the index and copy the metadata files.
"""

import argparse
import json
import shutil
import struct
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
from prime_rl.trainer.models.deepseek_v4.dequantize import dequantize_weight
from prime_rl.trainer.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
from prime_rl.trainer.models.deepseek_v41.converting_deepseek_v41 import conversion_chain

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64"}
ENGRAM_CHUNK_ROWS = 1 << 23
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


class SafetensorsWriter:
    """Writes one safetensors file whose tensors are produced chunk by chunk."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[tuple[str, torch.dtype, tuple[int, ...], Iterator[torch.Tensor]]] = []

    def add(self, name: str, dtype: torch.dtype, shape: tuple[int, ...], chunks: Iterator[torch.Tensor]) -> None:
        self.entries.append((name, dtype, shape, chunks))

    def write(self) -> None:
        header, offset = {}, 0
        for name, dtype, shape, _ in self.entries:
            nbytes = torch.Size(shape).numel() * torch.empty(0, dtype=dtype).element_size()
            header[name] = {
                "dtype": DTYPE_NAMES[dtype],
                "shape": list(shape),
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
        header_bytes = json.dumps(header).encode()
        header_bytes += b" " * (-len(header_bytes) % 8)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for name, dtype, _, chunks in self.entries:
                written = 0
                for chunk in chunks:
                    data = chunk.to(dtype).contiguous().cpu().view(torch.uint8).numpy()
                    f.write(memoryview(data))
                    written += data.nbytes
                expected = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
                assert written == expected, f"{name}: wrote {written} bytes, expected {expected}"
        tmp.rename(self.path)


class Source:
    def __init__(self, snapshot: Path):
        self.snapshot = snapshot
        self.weight_map: dict[str, str] = json.loads((snapshot / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self._handles = {}

    def _handle(self, key: str):
        file = self.weight_map[key]
        if file not in self._handles:
            self._handles[file] = safe_open(self.snapshot / file, framework="pt", device="cpu")
        return self._handles[file]

    def get(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def get_slice(self, key: str):
        return self._handle(key).get_slice(key)


def dequantized(source: Source, key: str, device: torch.device) -> torch.Tensor:
    """`key` as bf16 (or its stored dtype when unquantized), dequantized on `device`."""
    tensor = source.get(key)
    scale_key = key.removesuffix(".weight") + ".scale"
    if not key.endswith(".weight") or scale_key not in source.weight_map:
        return tensor
    return dequantize_weight(tensor.to(device), source.get(scale_key).to(device)).cpu()


def engram_chunks(source: Source, key: str, device: torch.device) -> Iterator[torch.Tensor]:
    weight, scale = source.get_slice(key), source.get_slice(key.removesuffix(".weight") + ".scale")
    rows = weight.get_shape()[0]
    for start in range(0, rows, ENGRAM_CHUNK_ROWS):
        stop = min(rows, start + ENGRAM_CHUNK_ROWS)
        yield dequantize_weight(weight[start:stop].to(device), scale[start:stop].to(device))


def convert_unit(source: Source, config, keys: list[str], out_path: Path, device: torch.device) -> list[str]:
    """Convert the published keys of one unit (a layer, or everything outside the layers)."""
    writer = SafetensorsWriter(out_path)
    engram_keys = [k for k in keys if ".engram.embed.weight" in k]
    regular = {}
    for key in keys:
        if key.endswith(".scale") or key in engram_keys:
            continue
        regular[key] = dequantized(source, key, device)
    converted = apply_hf_to_prime(regular, conversion_chain(config))
    for key in engram_keys:
        prime_key = next(iter(apply_hf_to_prime({key: torch.empty(0)}, conversion_chain(config))))
        rows, dim = source.get_slice(key).get_shape()
        writer.add(prime_key, torch.bfloat16, (rows, dim), engram_chunks(source, key, device))
    for name, tensor in converted.items():
        writer.add(
            name, tensor.dtype if tensor.dtype in DTYPE_NAMES else torch.bfloat16, tuple(tensor.shape), iter([tensor])
        )
    names = [name for name, *_ in writer.entries]
    writer.write()
    return names


def unit_of(key: str, config) -> str | None:
    """Which output file a published key lands in, or None when the key is dropped."""
    if key.startswith(("mtp.", "vision.", "aligner.", "image_")):
        return None
    if key.startswith("layers."):
        layer_idx = int(key.split(".")[1])
        return f"layer{layer_idx:02d}" if layer_idx < config.text_config.num_hidden_layers else None
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    raw_config = json.loads((args.snapshot_dir / "config.json").read_text())
    config = DeepseekV41Config.model_validate(raw_config)
    source = Source(args.snapshot_dir)
    units: dict[str, list[str]] = defaultdict(list)
    for key in source.weight_map:
        unit = unit_of(key, config)
        if unit is not None:
            units[unit].append(key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        weight_map = {}
        for unit in units:
            with safe_open(args.output_dir / f"model-{unit}.safetensors", framework="pt") as f:
                weight_map.update({name: f"model-{unit}.safetensors" for name in f.keys()})
        (args.output_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2))
        raw_config.pop("quantization_config", None)
        (args.output_dir / "config.json").write_text(json.dumps(raw_config, indent=2))
        for name in METADATA_FILES[1:]:
            if (args.snapshot_dir / name).exists():
                shutil.copy(args.snapshot_dir / name, args.output_dir / name)
        print(f"wrote index with {len(weight_map)} tensors to {args.output_dir}")
        return

    device = torch.device(f"cuda:{args.worker % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    for i, unit in enumerate(sorted(units)):
        out_path = args.output_dir / f"model-{unit}.safetensors"
        if i % args.num_workers != args.worker or out_path.exists():
            continue
        names = convert_unit(source, config.text_config, units[unit], out_path, device)
        print(f"[worker {args.worker}] {unit}: {len(names)} tensors", flush=True)


if __name__ == "__main__":
    main()
