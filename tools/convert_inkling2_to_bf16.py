"""Dequantize an Inkling 2 release checkpoint (NVFP4 / MXFP8 routed experts) to bf16.

The releases store the routed experts, ``model.llm.layers.N.mlp.experts.{w13,w2}_weight``, in NVFP4 or
MXFP8, chosen per tensor; every other tensor is plain bf16 or fp32. The trainer reads bf16 experts, so this
writes the same checkpoint with every expert weight dequantized and its scale tensors dropped.

Convert ``thinkingmachines/Inkling-2-Small-MXFP8``: an MXFP8 weight is a bf16 number, so that conversion is
exact. An NVFP4 weight is not: it is rounded to bf16, to within one step of the nearest number.

MXFP8 (E4M3 weight): every 32 inputs share the power-of-two scale ``2 ** (s - 127)``, with ``s`` the uint8 in
``<name>_scale``. A NaN code, or a scale for which some code of its block would not be a bf16 number, stops
the conversion. A block of zeros is zero at any scale.

NVFP4 (uint8 weight): two E2M1 codes per byte along the input dim, low nibble first. Bit 3 of a code is the
sign and bits 0-2 index (0, 0.5, 1, 1.5, 2, 3, 4, 6). Every 16 inputs share an E4M3 scale (``<name>_scale``)
and every expert a factor ``<name>_amax[e] / <name>_outer_scale_denominator``.

So that the output is read as bf16:

* ``hf_quant_config.json`` (the release's per-tensor precision metadata) lists the experts as bf16. vLLM
  builds a layer's experts from it, and serves bf16 experts on GPUs without the Blackwell kernels of the
  quantized ones.
* ``config.json`` loses ``quantization_config``, which vLLM would read before that file and which tells
  transformers that the experts are FP8, and gains ``transformers_version``: transformers 5.23 reads a
  ``config.json`` with neither key as the MXFP8 release.

Every other file is copied, subdirectories included. ``bf16_conversion.json`` records the source revision,
this tool's commit and what a rerun has to match.

Before the first write, the index, every weight file's header and size, the two JSON files and every scale
tensor are checked; ``--check`` stops there. NaN codes are found when their weight is converted.

Each file is written under a temporary name, synced to disk and renamed, and ``model.safetensors.index.json``
is written last. A rerun into the same output directory continues: it keeps the weight files that are
complete and hold the tensors it would write, and stops at one that does not (delete that file). It
continues only for the same input (the index and the sizes of the weight files) and the same version of
this file. One conversion at a time: ``bf16_conversion.lock`` in the output directory names the process that
holds it, and a conversion that was killed leaves it behind; delete it once that process is gone.

Usage (from the prime-rl repo):
    uv run python tools/convert_inkling2_to_bf16.py <model_dir> [output_dir] [--source-revision <commit>]
        [--device cuda] [--check]

Writes to ``<model_dir>-BF16`` by default. The arithmetic runs on the CPU unless ``--device`` names a GPU.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
# 2 ** (s - 127) for every scale byte s: read from a table, so that no device computes a power.
MXFP8_SCALES = torch.tensor([2.0 ** (scale - 127) for scale in range(256)], dtype=torch.float32)
# The scale bytes for which every E4M3 code times its scale is a bf16 number. Below, the smallest codes fall
# between bf16's subnormals; above, the largest overflow.
MXFP8_SCALE_RANGE = (3, 246)
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
EXPERT_FORMATS = {"F8_E4M3": "mxfp8", "U8": "nvfp4"}
INDEX = "model.safetensors.index.json"
PRECISION_METADATA = "hf_quant_config.json"
RECORD = "bf16_conversion.json"
LOCK = "bf16_conversion.lock"
QUANTIZED_COMPUTE_DTYPES = ("nvfp4", "mxfp8")
# The first transformers release with Inkling 2.
TRANSFORMERS_VERSION = "5.23.0"
# Experts dequantized at a time: bounds the fp32 intermediate to a few GB.
EXPERT_CHUNK = 16
# What a rerun into a started output directory has to match.
RESUME_KEYS = ("input_fingerprint", "tool_sha256")


def say(message: str) -> None:
    """Progress has to show under a pipe or in a batch log, where Python buffers a plain print to the end."""
    print(message, flush=True)


def dequantize_nvfp4(
    packed: torch.Tensor, block_scales: torch.Tensor, amax: torch.Tensor, outer_scale_denominator: torch.Tensor
) -> torch.Tensor:
    """``packed [E, N, K / 2]`` with ``block_scales [E, N, K / 16]`` and ``amax [E]`` -> bf16 ``[E, N, K]``."""
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=packed.device)
    code_values = torch.cat((magnitudes, -magnitudes))
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    weight = code_values[codes].unflatten(-1, (-1, NVFP4_BLOCK)) * block_scales.float().unsqueeze(-1)
    expert_factor = (amax.float() / outer_scale_denominator.float()).view(-1, 1, 1)
    return (weight.flatten(-2) * expert_factor).to(torch.bfloat16)


def dequantize_mxfp8(weight: torch.Tensor, block_exponents: torch.Tensor) -> torch.Tensor:
    """E4M3 ``weight [E, N, K]`` with uint8 ``block_exponents [E, N, K / 32]`` -> bf16 ``[E, N, K]``.

    Exact for the inputs `check_mxfp8_scales` and `check_mxfp8_codes` accept: the product of a code and a
    power of two is an fp32 number, and a bf16 number as well. A scale outside the range belongs to a block
    of zeros, which is zero at the nearest scale in range too.
    """
    low, high = MXFP8_SCALE_RANGE
    block_scales = MXFP8_SCALES.to(weight.device)[block_exponents.clamp(low, high).long()].unsqueeze(-1)
    return (weight.float().unflatten(-1, (-1, MXFP8_BLOCK)) * block_scales).flatten(-2).to(torch.bfloat16)


def check_mxfp8_scales(name: str, weight: torch.Tensor, block_exponents: torch.Tensor) -> None:
    low, high = MXFP8_SCALE_RANGE
    outside = (block_exponents < low) | (block_exponents > high)
    if not outside.any():
        return
    # Only these blocks' codes are read.
    codes = weight.view(torch.uint8).unflatten(-1, (-1, MXFP8_BLOCK))[outside]
    nonzero = (codes & 0x7F).any(dim=-1)
    if nonzero.any():
        scales = block_exponents[outside][nonzero]
        raise ValueError(
            f"{name}_scale holds {int(nonzero.sum())} scale bytes from {int(scales.min())} to {int(scales.max())} "
            f"on blocks that are not zero; the bf16 result is exact from {low} to {high} only"
        )


def check_mxfp8_codes(name: str, weight: torch.Tensor) -> None:
    if ((weight.view(torch.uint8) & 0x7F) == 0x7F).any():
        raise ValueError(f"{name} holds NaN codes")


def check_device_arithmetic(device: str) -> None:
    """Every E4M3 code at every scale in range has to dequantize on ``device`` to the bits it gives on the CPU."""
    codes = torch.tensor([code for code in range(256) if code & 0x7F != 0x7F] + [0, 0], dtype=torch.uint8)
    low, high = MXFP8_SCALE_RANGE
    weight = codes.view(torch.float8_e4m3fn).expand(1, high + 1 - low, -1).contiguous()
    block_exponents = torch.arange(low, high + 1, dtype=torch.uint8).view(1, -1, 1).repeat(1, 1, 256 // MXFP8_BLOCK)
    on_cpu = dequantize_mxfp8(weight, block_exponents)
    on_device = dequantize_mxfp8(weight.to(device), block_exponents.to(device)).cpu()
    if not torch.equal(on_device.view(torch.int16), on_cpu.view(torch.int16)):
        raise RuntimeError(f"MXFP8 dequantization on {device} differs from the CPU's")


def is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def is_expert_scale(name: str) -> bool:
    return is_expert_tensor(name) and name.endswith(SCALE_SUFFIXES)


def read_header(path: Path) -> dict[str, dict]:
    """A safetensors file's tensors: name -> dtype, shape and byte range. Fails unless the file has exactly
    the bytes its header describes."""
    size = path.stat().st_size
    with path.open("rb") as file:
        header_bytes = int.from_bytes(file.read(8), "little")
        if not 0 < header_bytes <= size - 8:
            raise ValueError(f"{path} is not a complete safetensors file")
        header = json.loads(file.read(header_bytes))
    header.pop("__metadata__", None)
    described = 8 + header_bytes + max((tensor["data_offsets"][1] for tensor in header.values()), default=0)
    if size != described:
        raise ValueError(f"{path} has {size} bytes and its header describes {described}: incomplete or damaged")
    return header


def list_experts_as_bf16(precision_metadata: dict) -> tuple[dict, set[str]]:
    """``hf_quant_config.json`` with the tensors of its NVFP4 / MXFP8 groups moved to one bf16 group, and
    those tensors."""
    groups = []
    experts: list[str] = []
    for group in precision_metadata["model"]["parameter_groups"]:
        if group["compute_dtype"] in QUANTIZED_COMPUTE_DTYPES:
            experts.extend(group["tensors"])
        else:
            groups.append(group)
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
    rewritten = {**precision_metadata, "model": {**precision_metadata["model"], "parameter_groups": groups}}
    return rewritten, set(experts)


@dataclass
class Release:
    """What the conversion reads, after `check_release`."""

    path: Path
    tensor_to_shard: dict[str, str]
    headers: dict[str, dict[str, dict]]
    # The expert weights to dequantize, and their format.
    formats: dict[str, str]
    config: dict
    precision_metadata: dict
    # The files to copy as they are, relative to `path`.
    other_files: list[Path]

    def get_tensor(self, name: str) -> torch.Tensor:
        """A view of the file's mapping, which goes when the tensor does: no file stays mapped once its
        tensors are written. Scales can live in a different shard than their weight."""
        with safe_open(self.path / self.tensor_to_shard[name], framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)

    def converted(self, shard_name: str) -> dict[str, tuple[str, list[int]]]:
        """The dtype and shape of every tensor the conversion writes to this file."""
        tensors = {}
        for name, tensor in self.headers[shard_name].items():
            if is_expert_scale(name):
                continue
            last = tensor["shape"][-1:] if self.formats.get(name) != "nvfp4" else [2 * tensor["shape"][-1]]
            dequantized = name in self.formats
            tensors[name] = ("BF16" if dequantized else tensor["dtype"], tensor["shape"][:-1] + last)
        return tensors


def check_release(input_dir: Path) -> Release:
    """Everything about the input that can be checked from its JSON files, headers and file sizes."""
    tensor_to_shard: dict[str, str] = json.loads((input_dir / INDEX).read_text())["weight_map"]
    config = json.loads((input_dir / "config.json").read_text())
    precision_metadata, listed = list_experts_as_bf16(json.loads((input_dir / PRECISION_METADATA).read_text()))
    shard_names = sorted(set(tensor_to_shard.values()))
    headers = {name: read_header(input_dir / name) for name in shard_names}

    in_headers = {name: shard for shard, header in headers.items() for name in header}
    if in_headers != tensor_to_shard:
        differing = sorted(name for name in in_headers.keys() | tensor_to_shard.keys())
        differing = [name for name in differing if in_headers.get(name) != tensor_to_shard.get(name)]
        raise ValueError(f"The index and the weight files disagree on {len(differing)} tensors: {differing[:3]}")

    formats = {}
    for name, shard in tensor_to_shard.items():
        weight = headers[shard][name]
        if not is_expert_tensor(name) or is_expert_scale(name) or weight["dtype"] not in EXPERT_FORMATS:
            continue
        formats[name] = EXPERT_FORMATS[weight["dtype"]]
        suffixes = SCALE_SUFFIXES if formats[name] == "nvfp4" else SCALE_SUFFIXES[:1]
        missing = [name + suffix for suffix in suffixes if name + suffix not in tensor_to_shard]
        if missing:
            raise ValueError(f"The checkpoint lacks {missing}")
        scale = headers[tensor_to_shard[name + "_scale"]][name + "_scale"]
        # An NVFP4 byte holds two inputs.
        scale_dtype, per_scale = ("U8", MXFP8_BLOCK) if formats[name] == "mxfp8" else ("F8_E4M3", NVFP4_BLOCK // 2)
        *leading, last = weight["shape"]
        if last % per_scale or (scale["dtype"], scale["shape"]) != (scale_dtype, [*leading, last // per_scale]):
            raise ValueError(
                f"{name}_scale is {scale['dtype']} {scale['shape']}, which does not fit {name} {weight['shape']}"
            )
    if not formats:
        raise ValueError(f"No quantized expert weights found in {input_dir} - is this an Inkling 2 release?")
    if listed != formats.keys():
        raise ValueError(f"{PRECISION_METADATA} lists other quantized tensors than the weight files hold")

    other_files = []
    skipped = {INDEX, "config.json", PRECISION_METADATA, *shard_names}
    for directory, subdirectories, names in os.walk(input_dir, followlinks=True):
        if Path(directory) == input_dir:
            # The Hub client's record of a download, and a git clone's second copy of every file.
            subdirectories[:] = [name for name in subdirectories if name not in (".cache", ".git")]
        for name in names:
            path = Path(directory, name)
            if not path.is_file():
                raise ValueError(f"{path} is not a file that can be copied")
            if path.relative_to(input_dir).as_posix() not in skipped:
                other_files.append(path.relative_to(input_dir))
    return Release(input_dir, tensor_to_shard, headers, formats, config, precision_metadata, sorted(other_files))


def check_scales(release: Release) -> None:
    """Reads every scale tensor: a few percent of the checkpoint, against all of it for the conversion."""
    smallest, largest = 255, 0
    for name, expert_format in sorted(release.formats.items()):
        scales = release.get_tensor(name + "_scale")
        if expert_format == "mxfp8":
            check_mxfp8_scales(name, release.get_tensor(name), scales)
            smallest, largest = min(smallest, int(scales.min())), max(largest, int(scales.max()))
        else:
            check_mxfp8_codes(name + "_scale", scales)
            factors = release.get_tensor(name + "_amax") / release.get_tensor(name + "_outer_scale_denominator")
            if not factors.isfinite().all():
                raise ValueError(f"{name}_amax and {name}_outer_scale_denominator give factors that are not finite")
    formats = dict(Counter(release.formats.values()))
    scale_range = f", MXFP8 scale bytes from {smallest} to {largest}" if "mxfp8" in formats else ""
    say(f"Checked {len(release.headers)} weight files and the scales of {formats} expert weights{scale_range}")


def source_of(input_dir: Path) -> tuple[str, str | None]:
    """The checkpoint's name and revision. A snapshot directory of the Hub cache,
    ``models--<org>--<name>/snapshots/<commit>``, gives both."""
    if input_dir.parent.name == "snapshots" and input_dir.parent.parent.name.startswith("models--"):
        return input_dir.parent.parent.name.removeprefix("models--").replace("--", "/"), input_dir.name
    return input_dir.name, None


def tool_commit() -> str | None:
    """The commit of the checkout this file is in, marked ``-dirty`` when tracked files differ from it."""
    git = ["git", "-C", str(Path(__file__).resolve().parent)]
    if shutil.which("git") is None:
        return None
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        return None
    changes = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if changes.stdout.strip() else "")


def new_record(release: Release, source_revision: str | None) -> dict:
    source, snapshot_revision = source_of(release.path)
    fingerprint = hashlib.sha256((release.path / INDEX).read_bytes())
    for shard_name in sorted(release.headers):
        fingerprint.update(f"{shard_name} {(release.path / shard_name).stat().st_size}\n".encode())
    return {
        "source": source,
        "source_revision": source_revision or snapshot_revision,
        "input_fingerprint": fingerprint.hexdigest(),
        "tool": "tools/convert_inkling2_to_bf16.py",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tool_commit": tool_commit(),
        "expert_weights": dict(Counter(release.formats.values())),
        "exact": "nvfp4" not in release.formats.values(),
    }


@contextmanager
def locked(output_dir: Path) -> Iterator[None]:
    """One conversion into an output directory at a time."""
    lock = output_dir / LOCK
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{lock} exists: a conversion into this directory is running, or one was killed "
            f"({lock.read_text().strip()}). Delete the file once that process is gone, and run again."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(f"process {os.getpid()} on {socket.gethostname()}\n")
        yield
    finally:
        lock.unlink()


def sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)


def partial_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".partial")


def finish(partial: Path) -> None:
    """Give a completed file its name once it is on disk, with the mode of a newly created file
    (safetensors writes 0600 and does not sync)."""
    umask = os.umask(0)
    os.umask(umask)
    partial.chmod(0o666 & ~umask)
    sync(partial)
    partial.replace(partial.with_suffix(""))
    sync(partial.parent)


def write_json(path: Path, content: dict, **kwargs) -> None:
    partial = partial_path(path)
    partial.write_text(json.dumps(content, indent=2, **kwargs) + "\n")
    finish(partial)


def convert(input_dir: Path, output_dir: Path, source_revision: str | None = None, device: str = "cpu") -> None:
    release = check_release(input_dir)
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError(f"{output_dir} is inside the checkpoint it would be converted from")
    check_scales(release)
    if device != "cpu":
        check_device_arithmetic(device)
    record = new_record(release, source_revision)

    output_dir.mkdir(parents=True, exist_ok=True)
    with locked(output_dir):
        if (output_dir / RECORD).exists():
            started = json.loads((output_dir / RECORD).read_text())
            differing = [key for key in RESUME_KEYS if started.get(key) != record[key]]
            if len({started["source_revision"], record["source_revision"]} - {None}) > 1:
                differing.append("source_revision")
            if differing:
                raise ValueError(
                    f"{output_dir} holds a conversion that differs from this one in {differing} (see {RECORD}): "
                    "another input, another revision, or another version of this file. Convert into an empty "
                    "directory."
                )
            record = {**started, "source_revision": started["source_revision"] or record["source_revision"]}
        elif any(path.name != LOCK for path in output_dir.iterdir()):
            raise ValueError(f"{output_dir} is not empty and has no {RECORD}. Convert into an empty directory.")
        write_json(output_dir / RECORD, record)

        # With the lock held, a temporary file is what a stopped run left: safetensors writes to `.tmp*`.
        for leftover in [*output_dir.glob(".tmp*"), *output_dir.rglob("*.partial")]:
            leftover.unlink()

        if not record["exact"]:
            print(
                f"WARNING: {record['expert_weights']['nvfp4']} of {len(release.formats)} expert weights are NVFP4. "
                "They are not bf16 numbers, so this conversion rounds them, and they are a coarser copy of the "
                "weights than MXFP8. Convert thinkingmachines/Inkling-2-Small-MXFP8 instead: that conversion is "
                "exact.",
                file=sys.stderr,
                flush=True,
            )
        if record["source_revision"] is None:
            say(f"The source revision is unknown, so {RECORD} does not record it: pass --source-revision.")

        def dequantize(name: str, weight: torch.Tensor) -> torch.Tensor:
            scales = release.get_tensor(name + "_scale")
            chunks = []
            if release.formats[name] == "nvfp4":
                amax = release.get_tensor(name + "_amax").to(device)
                denominator = release.get_tensor(name + "_outer_scale_denominator").to(device)
                for start in range(0, weight.shape[0], EXPERT_CHUNK):
                    experts = slice(start, start + EXPERT_CHUNK)
                    chunk = dequantize_nvfp4(
                        weight[experts].to(device), scales[experts].to(device), amax[experts], denominator
                    )
                    if not chunk.isfinite().all():
                        raise ValueError(f"{name} dequantizes to values that are not finite")
                    chunks.append(chunk.cpu())
            else:
                for start in range(0, weight.shape[0], EXPERT_CHUNK):
                    experts = slice(start, start + EXPERT_CHUNK)
                    check_mxfp8_codes(name, weight[experts])
                    chunks.append(dequantize_mxfp8(weight[experts].to(device), scales[experts].to(device)).cpu())
            return torch.cat(chunks)

        shard_names = sorted(release.headers)
        for position, shard_name in enumerate(shard_names, start=1):
            if (output_dir / shard_name).exists():
                written = read_header(output_dir / shard_name)
                if {name: (tensor["dtype"], tensor["shape"]) for name, tensor in written.items()} != release.converted(
                    shard_name
                ):
                    raise ValueError(
                        f"{output_dir / shard_name} does not hold the tensors this conversion writes. "
                        "Delete it and run again."
                    )
                say(f"Kept {shard_name} ({position} of {len(shard_names)})")
                continue
            out_shard: dict[str, torch.Tensor] = {}
            for name in sorted(release.converted(shard_name)):
                tensor = release.get_tensor(name)
                out_shard[name] = dequantize(name, tensor) if name in release.formats else tensor
            partial = partial_path(output_dir / shard_name)
            save_file(out_shard, partial, metadata={"format": "pt"})
            finish(partial)
            say(f"Wrote {shard_name} ({position} of {len(shard_names)})")

        for relative in release.other_files:
            # Everything else is copied as is, the speculator's `mtp.safetensors` included.
            source, copy = input_dir / relative, output_dir / relative
            if not copy.exists() or copy.stat().st_size != source.stat().st_size:
                partial = partial_path(copy)
                shutil.copyfile(source, partial)
                finish(partial)

        release.config.pop("quantization_config", None)
        release.config["transformers_version"] = TRANSFORMERS_VERSION
        write_json(output_dir / "config.json", release.config)
        write_json(output_dir / PRECISION_METADATA, release.precision_metadata)

        # The index follows the files that are there, whichever run wrote them.
        weight_map: dict[str, str] = {}
        total_size = 0
        for shard_name in shard_names:
            for name, tensor in read_header(output_dir / shard_name).items():
                weight_map[name] = shard_name
                total_size += tensor["data_offsets"][1] - tensor["data_offsets"][0]
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        write_json(output_dir / INDEX, index, sort_keys=True)
    say(f"Done: {len(release.formats)} expert weights dequantized -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path, help="Inkling 2 release checkpoint dir (MXFP8 or NVFP4 experts)")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="default: <input_dir>-BF16; required for a Hub cache snapshot, where that would be in the cache",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Hub commit of the checkpoint, to record in the output (default: the one in a Hub cache snapshot path)",
    )
    parser.add_argument("--device", default="cpu", help="where the arithmetic runs; a GPU has to give the CPU's bits")
    parser.add_argument("--check", action="store_true", help="check the input, scales included, and write nothing")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.check:
        check_scales(check_release(input_dir))
        return
    if args.output_dir is None and source_of(input_dir)[1] is not None:
        parser.error("name an output directory: the default would be inside the Hub cache")
    output_dir = args.output_dir or input_dir.with_name(input_dir.name + "-BF16")
    convert(input_dir, output_dir, args.source_revision, args.device)


if __name__ == "__main__":
    main()
