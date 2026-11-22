"""Convert an Inkling 2 Small MXFP8 release checkpoint to bf16.

The release stores the routed-expert weights as F8_E4M3 with per-1x32-block
ue8m0 scales (``*_weight_scale``). A weight block dequantizes exactly to

    bf16 = fp8 * 2 ** (scale_byte - 127)

because an e4m3 value has a 4-bit significand, so scaling by a power of two is
exact in bf16. Scale tensors are dropped and every other tensor is copied
byte-for-byte.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_mxfp8_to_bf16.py <release_dir> [<output_dir>]

Writes to ``<release_dir>-BF16`` by default. To convert shards in parallel, run
one process per shard with ``--shard`` and then once with ``--finalize``.
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np

DTYPE_ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
SCALE_SUFFIX = "_weight_scale"


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, header_len


def is_expert_scale(name: str, header: dict) -> bool:
    return name.endswith(SCALE_SUFFIX) and header[name]["dtype"] == "U8" and ".mlp.experts." in name


def is_expert_weight(name: str, info: dict) -> bool:
    return ".mlp.experts." in name and info["dtype"].startswith("F8_")


def build_e4m3_lut() -> np.ndarray:
    """The bf16 bit pattern of every e4m3 byte at scale exponent 127.

    A bf16 bit pattern is sign | exponent | mantissa, so adding
    ``(scale_byte - 127) << 7`` multiplies it by ``2 ** (scale_byte - 127)``
    while the result stays normal and finite.
    """
    lut = np.zeros(256, dtype=np.uint16)
    for byte in range(256):
        sign = byte >> 7
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            # NaN encoding; convert_expert_weight rejects it before using the table.
            lut[byte] = 0x7FC0 if sign == 0 else 0xFFC0
            continue
        if exponent == 0:
            value = mantissa * 2.0**-9
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        if sign:
            value = -value
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        # float32 -> bf16, round to nearest even; exact for e4m3 values.
        lsb = (bits >> 16) & 1
        lut[byte] = ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF
    return lut


E4M3_LUT = build_e4m3_lut()


def convert_expert_weight(weight: np.ndarray, scales: np.ndarray, out_file) -> int:
    """Dequantize uint8 ``weight [E, R, K]`` with uint8 ``scales [E, R, K // 32]`` to bf16."""
    experts, rows, width = weight.shape
    blocks = scales.shape[2]
    if width != blocks * 32:
        raise ValueError(f"weight shape {weight.shape} does not match scale shape {scales.shape}")
    if scales.min() < 10 or scales.max() > 246:
        raise ValueError(f"scale byte range [{scales.min()}, {scales.max()}] outside [10, 246]")
    for expert in range(experts):
        codes = weight[expert]
        if (codes & 0x7F == 0x7F).any():
            raise ValueError("NaN code in the fp8 expert weight")
        exponent = (scales[expert].astype(np.uint16) - np.uint16(127)) << 7
        values = E4M3_LUT[codes]
        values.reshape(rows, blocks, 32).__iadd__(exponent[:, :, None])
        # Subnormals and zeros cannot take the exponent add: restore their bf16 bit patterns.
        zeros = (codes & 0x7F) == 0
        if zeros.any():
            values[zeros] = (codes[zeros].astype(np.uint16) & np.uint16(0x80)) << 8
        out_file.write(memoryview(values).cast("B"))
    return experts


def convert_shard(input_dir: Path, output_dir: Path, shard: str) -> int:
    src_path = input_dir / shard
    dst_path = output_dir / shard
    header, header_len = read_header(src_path)
    metadata = header.pop("__metadata__", {"format": "pt"})
    data_start = 8 + header_len

    entries = []
    for name, info in header.items():
        if is_expert_scale(name, header):
            continue
        dtype = info["dtype"]
        converted = is_expert_weight(name, info)
        if not converted and dtype.startswith("F8_"):
            raise ValueError(f"unexpected quantized non-expert tensor {name}")
        n_bytes = int(np.prod(info["shape"], dtype=np.int64)) * DTYPE_ITEMSIZE[dtype]
        entries.append((name, info, "BF16" if converted else dtype, n_bytes * 2 if converted else n_bytes))

    out_header = {}
    offset = 0
    for name, info, out_dtype, out_bytes in entries:
        out_header[name] = {"dtype": out_dtype, "shape": info["shape"], "data_offsets": [offset, offset + out_bytes]}
        offset += out_bytes
    out_header["__metadata__"] = metadata
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    expected_size = data_start + max(info["data_offsets"][1] for info in header.values())
    if src_path.stat().st_size != expected_size:
        raise ValueError(f"{src_path} is truncated: {src_path.stat().st_size} != {expected_size}")

    source = np.memmap(src_path, dtype=np.uint8, mode="r")
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    n_experts = 0
    with open(tmp_path, "wb", buffering=8 << 20) as out_file:
        out_file.write(struct.pack("<Q", len(header_bytes)))
        out_file.write(header_bytes)
        for name, info, _, _ in entries:
            start = data_start + info["data_offsets"][0]
            end = data_start + info["data_offsets"][1]
            if is_expert_weight(name, info):
                scale_info = header[name[: -len("_weight")] + SCALE_SUFFIX]
                scale_start = data_start + scale_info["data_offsets"][0]
                scale_end = data_start + scale_info["data_offsets"][1]
                weight = source[start:end].reshape(info["shape"])
                scales = source[scale_start:scale_end].reshape(scale_info["shape"])
                n_experts += convert_expert_weight(weight, scales, out_file)
            else:
                out_file.write(memoryview(source[start:end]))
    os.replace(tmp_path, dst_path)
    print(f"[convert] {shard}: {len(entries)} tensors, {n_experts} experts, {dst_path.stat().st_size / 1e9:.2f} GB")
    return n_experts


def converted_shard_header(source: Path, converted: Path) -> dict:
    original, _ = read_header(source)
    header, header_len = read_header(converted)
    original.pop("__metadata__", None)
    header.pop("__metadata__", None)
    expected = {name for name in original if not is_expert_scale(name, original)}
    if header.keys() != expected:
        raise ValueError(f"{converted} has unexpected or missing tensors")
    offset = 0
    for name, info in sorted(header.items(), key=lambda entry: entry[1]["data_offsets"][0]):
        src = original[name]
        dtype = "BF16" if is_expert_weight(name, src) else src["dtype"]
        end = offset + int(np.prod(src["shape"], dtype=np.int64)) * DTYPE_ITEMSIZE[dtype]
        if info != {"dtype": dtype, "shape": src["shape"], "data_offsets": [offset, end]}:
            raise ValueError(f"{converted} has invalid metadata for {name}")
        offset = end
    if converted.stat().st_size != 8 + header_len + offset:
        raise ValueError(f"{converted} has an incorrect file size")
    return header


def finalize(input_dir: Path, output_dir: Path) -> None:
    old_index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(old_index["weight_map"].values()))
    weight_map = {}
    total_size = 0
    for shard in shards:
        header = converted_shard_header(input_dir / shard, output_dir / shard)
        total_size += max(info["data_offsets"][1] for info in header.values())
        for name in header:
            if name in weight_map:
                raise ValueError(f"tensor {name} appears in two shards")
            weight_map[name] = shard
    for name in old_index["weight_map"]:
        if name not in weight_map and not name.endswith(SCALE_SUFFIX):
            raise ValueError(f"missing tensor {name} in the converted shards")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config = json.loads((input_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    # transformers treats a config without a version as written by the release and
    # may re-attach the fp8 quantization config; mark the output as a plain bf16 export.
    config["transformers_version"] = "5.23.0"
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    converted = {"model.safetensors.index.json", "config.json", *shards}
    for path in input_dir.iterdir():
        if path.is_file() and path.name not in converted:
            shutil.copyfile(path, output_dir / path.name)
    print(f"[finalize] {len(weight_map)} tensors, total_size {total_size / 1e9:.1f} GB -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 Small MXFP8 release directory")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-BF16")
    parser.add_argument("--shard", help="convert only this shard file (for parallel jobs)")
    parser.add_argument("--finalize", action="store_true", help="write the index, config and side files only")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir.with_name(input_dir.name + "-BF16")).resolve()
    if output_dir == input_dir:
        parser.error("input and output directories must be different")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.finalize:
        index = json.loads((input_dir / "model.safetensors.index.json").read_text())
        shards = [args.shard] if args.shard else sorted(set(index["weight_map"].values()))
        for shard in shards:
            if (output_dir / shard).exists():
                converted_shard_header(input_dir / shard, output_dir / shard)
                print(f"[skip] {shard}")
                continue
            convert_shard(input_dir, output_dir, shard)
    if not args.shard:
        finalize(input_dir, output_dir)


if __name__ == "__main__":
    main()
