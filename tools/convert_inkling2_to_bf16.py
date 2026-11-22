"""Convert an Inkling 2 release checkpoint (NVFP4 / MXFP8 routed experts) to bf16.

The releases store the routed-expert weights (``*.mlp.experts.{w13,w2}_weight``) quantized, in a
format chosen per tensor; every other tensor is plain bf16 or fp32.

MXFP8 (F8_E4M3 weight): every 32 inputs share a ue8m0 scale (``<name>_scale``), and

    bf16 = fp8 * 2 ** (scale_byte - 127)

exactly, because an e4m3 value has a 4-bit significand.

NVFP4 (U8 weight): two E2M1 codes per byte along the input dim, low nibble first. Every 16 inputs
share an E4M3 scale (``<name>_scale``) and every expert a factor
``<name>_amax[e] / <name>_outer_scale_denominator``:

    bf16 = round(e2m1 * block_scale * expert_factor)

computed in fp32 like the serving engines.

The scale tensors are dropped and every other tensor is copied byte-for-byte.
``hf_quant_config.json`` (the release's per-tensor precision metadata) is rewritten to list the
experts as bf16.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_to_bf16.py <release_dir> [<output_dir>]

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
WEIGHT_SUFFIX = "_weight"
SCALE_SUFFIX = "_weight_scale"
AMAX_SUFFIX = "_weight_amax"
DENOMINATOR_SUFFIX = "_weight_outer_scale_denominator"
SIDECAR_SUFFIXES = (SCALE_SUFFIX, AMAX_SUFFIX, DENOMINATOR_SUFFIX)
# Inputs that share a block scale.
BLOCK = {"mxfp8": 32, "nvfp4": 16}
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
PRECISION_METADATA = "hf_quant_config.json"


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, header_len


def is_expert_sidecar(name: str) -> bool:
    return ".mlp.experts." in name and name.endswith(SIDECAR_SUFFIXES)


def expert_format(name: str, info: dict) -> str | None:
    """``"mxfp8"`` or ``"nvfp4"`` for a quantized expert weight, None for any other tensor."""
    if ".mlp.experts." not in name or not name.endswith(WEIGHT_SUFFIX):
        return None
    return {"F8_E4M3": "mxfp8", "U8": "nvfp4"}.get(info["dtype"])


def converted_info(name: str, info: dict) -> tuple[str, list[int]]:
    """The dtype and shape a tensor has in the output."""
    fmt = expert_format(name, info)
    if fmt is None:
        return info["dtype"], info["shape"]
    # An NVFP4 byte packs two inputs.
    return "BF16", [*info["shape"][:-1], info["shape"][-1] * (2 if fmt == "nvfp4" else 1)]


def e4m3_value(byte: int) -> float:
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    value = mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if byte >> 7 else value


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns, round to nearest even."""
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def build_e4m3_lut() -> np.ndarray:
    """The bf16 bit pattern of every e4m3 byte at scale exponent 127.

    A bf16 bit pattern is sign | exponent | mantissa, so adding
    ``(scale_byte - 127) << 7`` multiplies it by ``2 ** (scale_byte - 127)``
    while the result stays normal and finite.
    """
    # The NaN encodings (0x7F, 0xFF) are rejected before the table is used.
    values = np.array([e4m3_value(byte) for byte in range(256)], dtype=np.float32)
    return f32_to_bf16_bits(values)


E4M3_LUT = build_e4m3_lut()
E4M3_VALUES = np.array([e4m3_value(byte) for byte in range(256)], dtype=np.float32)


def convert_mxfp8_weight(weight: np.ndarray, scales: np.ndarray, out_file) -> None:
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


def convert_nvfp4_weight(
    packed: np.ndarray, scales: np.ndarray, amax: np.ndarray, denominator: np.ndarray, out_file
) -> None:
    """Dequantize uint8 ``packed [E, R, K // 2]`` with e4m3 ``scales [E, R, K // 16]``, fp32 ``amax [E]`` and
    fp32 ``denominator [1]`` to bf16 ``[E, R, K]``."""
    experts, rows, half_width = packed.shape
    blocks = scales.shape[2]
    if half_width != blocks * 8:
        raise ValueError(f"weight shape {packed.shape} does not match scale shape {scales.shape}")
    if amax.shape != (experts,) or denominator.shape != (1,):
        raise ValueError(f"unexpected amax {amax.shape} or outer scale denominator {denominator.shape} shape")
    if not np.isfinite(amax).all() or (amax < 0).any() or not np.isfinite(denominator).all() or denominator[0] <= 0:
        raise ValueError("amax must be finite and non-negative and the outer scale denominator finite and positive")
    factors = amax.astype(np.float32) / denominator.astype(np.float32)[0]
    for expert in range(experts):
        codes = packed[expert]
        scale_codes = scales[expert]
        if (scale_codes & 0x7F == 0x7F).any():
            raise ValueError("NaN code in the NVFP4 block scales")
        # The low nibble holds the even input.
        values = np.stack((E2M1[codes & 0xF], E2M1[codes >> 4]), axis=-1).reshape(rows, blocks, 16)
        values *= E4M3_VALUES[scale_codes][:, :, None]
        values *= factors[expert]
        if not np.isfinite(values).all():
            raise ValueError("non-finite dequantized NVFP4 expert weight")
        out_file.write(memoryview(f32_to_bf16_bits(values.reshape(rows, -1))).cast("B"))


class Release:
    """The safetensors shards of a release: an expert weight's scales can live in another shard."""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.index = json.loads((input_dir / "model.safetensors.index.json").read_text())
        self.shards = sorted(set(self.index["weight_map"].values()))
        self.headers: dict[str, dict] = {}
        self.data_starts: dict[str, int] = {}
        self.metadata: dict[str, dict] = {}
        self.location: dict[str, str] = {}
        self._sources: dict[str, np.memmap] = {}
        for shard in self.shards:
            header, header_len = read_header(input_dir / shard)
            self.metadata[shard] = header.pop("__metadata__", {"format": "pt"})
            self.headers[shard] = header
            self.data_starts[shard] = 8 + header_len
            for name in header:
                if name in self.location:
                    raise ValueError(f"tensor {name} appears in two shards")
                self.location[name] = shard

    def info(self, name: str) -> dict:
        return self.headers[self.location[name]][name]

    def read(self, name: str) -> np.ndarray:
        """The raw bytes of a tensor, shaped like the tensor for one-byte dtypes."""
        shard = self.location[name]
        if shard not in self._sources:
            path = self.input_dir / shard
            expected = self.data_starts[shard] + max(
                (info["data_offsets"][1] for info in self.headers[shard].values()), default=0
            )
            if path.stat().st_size != expected:
                raise ValueError(f"{path} is truncated: {path.stat().st_size} != {expected}")
            self._sources[shard] = np.memmap(path, dtype=np.uint8, mode="r")
        info = self.headers[shard][name]
        start, end = (self.data_starts[shard] + offset for offset in info["data_offsets"])
        data = self._sources[shard][start:end]
        return data.reshape(info["shape"]) if DTYPE_ITEMSIZE[info["dtype"]] == 1 else data

    def read_f32(self, name: str) -> np.ndarray:
        info = self.info(name)
        if info["dtype"] != "F32":
            raise ValueError(f"{name} is {info['dtype']}, expected F32")
        return np.frombuffer(self.read(name), dtype=np.float32).reshape(info["shape"])

    def expert_formats(self) -> dict[str, str]:
        """The quantization format of every quantized expert weight."""
        formats = {}
        for name in self.location:
            info = self.info(name)
            fmt = expert_format(name, info)
            if fmt is not None:
                formats[name] = fmt
            elif not is_expert_sidecar(name) and (
                info["dtype"].startswith("F8_") or (info["dtype"] == "U8" and ".mlp.experts." in name)
            ):
                raise ValueError(f"unexpected quantized tensor {name}")
        return formats

    def convert_expert(self, name: str, out_file) -> None:
        stem = name[: -len(WEIGHT_SUFFIX)]
        scales = self.read(stem + SCALE_SUFFIX)
        if expert_format(name, self.info(name)) == "mxfp8":
            if self.info(stem + SCALE_SUFFIX)["dtype"] != "U8":
                raise ValueError(f"{stem + SCALE_SUFFIX} must hold ue8m0 scales")
            convert_mxfp8_weight(self.read(name), scales, out_file)
        else:
            if self.info(stem + SCALE_SUFFIX)["dtype"] != "F8_E4M3":
                raise ValueError(f"{stem + SCALE_SUFFIX} must hold e4m3 scales")
            convert_nvfp4_weight(
                self.read(name),
                scales,
                self.read_f32(stem + AMAX_SUFFIX),
                self.read_f32(stem + DENOMINATOR_SUFFIX),
                out_file,
            )

    def converted_header(self, shard: str) -> dict:
        """The safetensors header of a converted shard, without its metadata."""
        header, offset = {}, 0
        for name, info in self.headers[shard].items():
            if is_expert_sidecar(name):
                continue
            dtype, shape = converted_info(name, info)
            end = offset + int(np.prod(shape, dtype=np.int64)) * DTYPE_ITEMSIZE[dtype]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
            offset = end
        return header

    def convert_shard(self, output_dir: Path, shard: str) -> None:
        dst_path = output_dir / shard
        out_header = self.converted_header(shard)
        header_bytes = json.dumps(out_header | {"__metadata__": self.metadata[shard]}, separators=(",", ":")).encode()
        header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        n_experts = 0
        with open(tmp_path, "wb", buffering=8 << 20) as out_file:
            out_file.write(struct.pack("<Q", len(header_bytes)))
            out_file.write(header_bytes)
            for name in out_header:
                info = self.headers[shard][name]
                if expert_format(name, info) is not None:
                    self.convert_expert(name, out_file)
                    n_experts += info["shape"][0]
                else:
                    out_file.write(memoryview(self.read(name).reshape(-1)))
        os.replace(tmp_path, dst_path)
        print(
            f"[convert] {shard}: {len(out_header)} tensors, {n_experts} experts, {dst_path.stat().st_size / 1e9:.2f} GB"
        )

    def check_converted_shard(self, converted: Path, shard: str) -> dict:
        """Raises unless ``converted`` is a complete conversion of ``shard``; returns its header."""
        header, header_len = read_header(converted)
        header.pop("__metadata__", None)
        expected = self.converted_header(shard)
        if header.keys() != expected.keys():
            raise ValueError(f"{converted} has unexpected or missing tensors")
        for name, info in expected.items():
            if header[name] != info:
                raise ValueError(f"{converted} has invalid metadata for {name}")
        size = max((info["data_offsets"][1] for info in expected.values()), default=0)
        if converted.stat().st_size != 8 + header_len + size:
            raise ValueError(f"{converted} has an incorrect file size")
        return header


def list_experts_as_bf16(precision_metadata: dict, formats: dict[str, str]) -> dict:
    """``hf_quant_config.json`` with the tensors of its NVFP4 / MXFP8 groups moved to one bf16 group."""
    groups, experts = [], []
    listed = {fmt: set() for fmt in BLOCK}
    for group in precision_metadata["model"]["parameter_groups"]:
        if group["compute_dtype"] in BLOCK:
            listed[group["compute_dtype"]].update(group["tensors"])
            experts.extend(group["tensors"])
        else:
            groups.append(group)
    for fmt, names in listed.items():
        if names != {name for name, tensor_fmt in formats.items() if tensor_fmt == fmt}:
            raise ValueError(f"{PRECISION_METADATA} and the tensor dtypes disagree on the {fmt} expert weights")
    groups.append(
        {
            "compute_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "required_activation_quantization": None,
            "tensors": sorted(experts),
            "weight_block_shape": [],
            "weight_transform": "cast",
        }
    )
    return {**precision_metadata, "model": {**precision_metadata["model"], "parameter_groups": groups}}


def finalize(release: Release, output_dir: Path) -> None:
    input_dir = release.input_dir
    weight_map = {}
    total_size = 0
    for shard in release.shards:
        header = release.check_converted_shard(output_dir / shard, shard)
        total_size += max((info["data_offsets"][1] for info in header.values()), default=0)
        for name in header:
            weight_map[name] = shard
    for name in release.index["weight_map"]:
        if name not in weight_map and not is_expert_sidecar(name):
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

    converted = {"model.safetensors.index.json", "config.json", *release.shards}
    if (input_dir / PRECISION_METADATA).exists():
        precision_metadata = json.loads((input_dir / PRECISION_METADATA).read_text())
        precision_metadata = list_experts_as_bf16(precision_metadata, release.expert_formats())
        (output_dir / PRECISION_METADATA).write_text(json.dumps(precision_metadata, indent=2) + "\n")
        converted.add(PRECISION_METADATA)
    # Side files, such as the tokenizer and the DSpark drafter (mtp.safetensors), are copied unchanged.
    for path in input_dir.iterdir():
        if path.is_file() and path.name not in converted:
            shutil.copyfile(path, output_dir / path.name)
    print(f"[finalize] {len(weight_map)} tensors, total_size {total_size / 1e9:.1f} GB -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 release directory (NVFP4 or MXFP8 experts)")
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <input_dir>-BF16")
    parser.add_argument("--shard", help="convert only this shard file (for parallel jobs)")
    parser.add_argument("--finalize", action="store_true", help="write the index, config and side files only")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir.with_name(input_dir.name + "-BF16")).resolve()
    if output_dir == input_dir:
        parser.error("input and output directories must be different")
    output_dir.mkdir(parents=True, exist_ok=True)
    release = Release(input_dir)
    release.expert_formats()
    if not args.finalize:
        for shard in [args.shard] if args.shard else release.shards:
            if (output_dir / shard).exists():
                release.check_converted_shard(output_dir / shard, shard)
                print(f"[skip] {shard}")
                continue
            release.convert_shard(output_dir, shard)
    if not args.shard:
        finalize(release, output_dir)


if __name__ == "__main__":
    main()
