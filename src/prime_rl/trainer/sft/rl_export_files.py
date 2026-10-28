"""File-level part of the GeneBench RL -> Hugging Face export (see hf_export.py): writing one base safetensors file
with the trained tensors swapped in, and checking it. Needs only torch and safetensors (no distributed setup)."""

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

# The untrained multi-token-prediction head: absent from the DCP checkpoint, copied from the base.
UNTRAINED_PREFIX = "mtp."


def is_trained(key: str) -> bool:
    return not key.startswith(UNTRAINED_PREFIX)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def assign_files(sizes: dict[str, int], num_ranks: int) -> list[list[str]]:
    """Spread files over ranks by size, largest first onto the least-loaded rank (deterministic)."""
    loads = [0] * num_ranks
    assigned: list[list[str]] = [[] for _ in range(num_ranks)]
    for name in sorted(sizes, key=lambda name: (-sizes[name], name)):
        rank = min(range(num_ranks), key=lambda r: (loads[r], r))
        assigned[rank].append(name)
        loads[rank] += sizes[name]
    return assigned


def write_file(src: Path, dst: Path, trained: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, float], int]:
    """Write ``src`` to ``dst`` with its trained keys swapped in (cast to the source dtype); the rest is copied.

    Consumes the swapped-in entries of ``trained``. Returns each swapped tensor's relative L2 change from the base,
    as exported and in the trainer's fp32, and the number of copied tensors.
    """
    tensors: dict[str, Tensor] = {}
    exported: dict[str, float] = {}
    fp32: dict[str, float] = {}
    copied = 0
    with safe_open(src, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            base = f.get_tensor(key)
            if not is_trained(key):
                tensors[key] = base
                copied += 1
                continue
            new = trained.pop(key)
            if not torch.isfinite(new).all():
                raise ValueError(f"{key}: non-finite values")
            if new.shape != base.shape:
                raise ValueError(f"{key}: trained shape {tuple(new.shape)} != base shape {tuple(base.shape)}")
            base_f = base.float()
            base_norm = base_f.norm().item()
            delta = (new - base_f).norm().item()
            fp32[key] = delta / base_norm if base_norm > 0 else delta
            new = new.to(base.dtype)
            delta = (new.float() - base_f).norm().item()
            exported[key] = delta / base_norm if base_norm > 0 else delta
            tensors[key] = new
    save_file(tensors, dst, metadata=metadata)
    return exported, fp32, copied


def check_layout(src: Path, dst: Path) -> None:
    """``dst`` holds exactly ``src``'s keys with the same dtypes and shapes."""
    with safe_open(src, framework="pt") as a, safe_open(dst, framework="pt") as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError(f"{dst.name}: key set differs from the base")
        for key in a.keys():
            sa, sb = a.get_slice(key), b.get_slice(key)
            if (sa.get_dtype(), sa.get_shape()) != (sb.get_dtype(), sb.get_shape()):
                raise ValueError(
                    f"{dst.name}:{key}: {sb.get_dtype()} {sb.get_shape()} != {sa.get_dtype()} {sa.get_shape()}"
                )


def write_my_files(
    base_dir: Path, my_files: list[str], out_dir: Path, trained: dict[str, Tensor]
) -> tuple[dict[str, float], dict[str, float], int, dict[str, str]]:
    exported: dict[str, float] = {}
    fp32: dict[str, float] = {}
    copied = 0
    digests: dict[str, str] = {}
    for name in my_files:
        e, f, c = write_file(base_dir / name, out_dir / name, trained)
        exported.update(e)
        fp32.update(f)
        copied += c
        check_layout(base_dir / name, out_dir / name)
        digests[name] = sha256_file(out_dir / name)
    if trained:
        raise ValueError(f"converted keys not in this rank's base files: {sorted(trained)[:5]}")
    return exported, fp32, copied, digests
