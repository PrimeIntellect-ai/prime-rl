import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

SPARSE_UPDATE_FORMAT = "prime-rl-sparse-update"
SPARSE_UPDATE_VERSION = 1
SPARSE_UPDATE_MANIFEST_NAME = "sparse_update_manifest.json"
SPARSE_UPDATE_PATCH_NAME = "sparse_update_patch.safetensors"


@dataclass(frozen=True)
class SparseUpdateStats:
    total_tensors: int
    patched_tensors: int
    total_numel: int
    changed_numel: int
    patch_bytes: int

    @property
    def sparsity(self) -> float:
        if self.total_numel == 0:
            return 1.0
        return 1.0 - (self.changed_numel / self.total_numel)


def is_sparse_update_dir(path: Path) -> bool:
    return (path / SPARSE_UPDATE_MANIFEST_NAME).exists()


def to_compute_tensor(tensor: Tensor, compute_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    tensor = tensor.detach()
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=compute_dtype)
    return tensor.to(device="cpu", copy=True).contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in sparse update patch: {name}")
    return dtype


def save_sparse_update(
    previous_state: Mapping[str, Tensor],
    current_state: Mapping[str, Tensor],
    save_dir: Path,
    *,
    step: int,
    base_step: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> SparseUpdateStats:
    """Save changed compute-dtype values between two dense state dicts."""
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_tensors: dict[str, Tensor] = {}
    tensor_entries: list[dict] = []
    total_numel = 0
    changed_numel = 0

    for tensor_idx, (name, current) in enumerate(current_state.items()):
        current_compute = to_compute_tensor(current, compute_dtype)
        previous = previous_state.get(name)
        total_numel += current_compute.numel()

        if previous is None or tuple(previous.shape) != tuple(current_compute.shape):
            indices = torch.arange(current_compute.numel(), dtype=torch.int64)
        else:
            previous_compute = previous.to(dtype=current_compute.dtype, device="cpu").contiguous()
            changed = current_compute.reshape(-1).ne(previous_compute.reshape(-1))
            indices = torch.nonzero(changed, as_tuple=False).reshape(-1).to(torch.int64)

        if indices.numel() == 0:
            continue

        values = current_compute.reshape(-1).index_select(0, indices).contiguous()
        indices_key = f"tensor_{tensor_idx}.indices"
        values_key = f"tensor_{tensor_idx}.values"
        patch_tensors[indices_key] = indices
        patch_tensors[values_key] = values
        changed_numel += indices.numel()
        tensor_entries.append(
            {
                "name": name,
                "shape": list(current_compute.shape),
                "dtype": _dtype_name(current_compute.dtype),
                "indices": indices_key,
                "values": values_key,
                "numel": current_compute.numel(),
                "changed_numel": indices.numel(),
            }
        )

    if patch_tensors:
        save_file(patch_tensors, save_dir / SPARSE_UPDATE_PATCH_NAME, metadata={"format": "pt"})

    patch_file = save_dir / SPARSE_UPDATE_PATCH_NAME
    patch_bytes = patch_file.stat().st_size if patch_file.exists() else 0
    stats = SparseUpdateStats(
        total_tensors=len(current_state),
        patched_tensors=len(tensor_entries),
        total_numel=total_numel,
        changed_numel=changed_numel,
        patch_bytes=patch_bytes,
    )
    manifest = {
        "format": SPARSE_UPDATE_FORMAT,
        "version": SPARSE_UPDATE_VERSION,
        "step": step,
        "base_step": base_step,
        "compute_dtype": _dtype_name(compute_dtype),
        "patch_file": SPARSE_UPDATE_PATCH_NAME if patch_tensors else None,
        "tensors": tensor_entries,
        "stats": {
            "total_tensors": stats.total_tensors,
            "patched_tensors": stats.patched_tensors,
            "total_numel": stats.total_numel,
            "changed_numel": stats.changed_numel,
            "sparsity": stats.sparsity,
            "patch_bytes": stats.patch_bytes,
        },
    }
    with open(save_dir / SPARSE_UPDATE_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return stats


def load_sparse_update_manifest(patch_dir: Path) -> dict:
    with open(patch_dir / SPARSE_UPDATE_MANIFEST_NAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format") != SPARSE_UPDATE_FORMAT:
        raise ValueError(f"Unsupported sparse update patch format: {manifest.get('format')}")
    if manifest.get("version") != SPARSE_UPDATE_VERSION:
        raise ValueError(f"Unsupported sparse update patch version: {manifest.get('version')}")
    return manifest


def apply_sparse_update(
    state_dict: dict[str, Tensor], patch_dir: Path, *, expected_base_step: int | None = None
) -> dict:
    """Apply a sparse value update to a dense CPU state dict in-place."""
    manifest = load_sparse_update_manifest(patch_dir)
    if expected_base_step is not None and manifest["base_step"] != expected_base_step:
        raise ValueError(f"Sparse update base step mismatch: cache={expected_base_step} patch={manifest['base_step']}")

    patch_file = manifest.get("patch_file")
    patch_tensors = load_file(patch_dir / patch_file, device="cpu") if patch_file else {}

    for entry in manifest["tensors"]:
        name = entry["name"]
        if name not in state_dict:
            raise KeyError(f"Sparse update refers to tensor not present in receiver cache: {name}")

        expected_shape = tuple(entry["shape"])
        tensor = state_dict[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Sparse update shape mismatch for {name}: cache={tuple(tensor.shape)} patch={expected_shape}"
            )

        dtype = _dtype_from_name(entry["dtype"])
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)

        indices = patch_tensors[entry["indices"]].to(dtype=torch.long)
        values = patch_tensors[entry["values"]].to(dtype=dtype)
        tensor.reshape(-1).index_copy_(0, indices, values)
        state_dict[name] = tensor.contiguous()

    return manifest
