import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import torch
from safetensors.torch import load, load_file, save, save_file
from torch import Tensor

if TYPE_CHECKING:
    from torch.nn import Module

SPARSE_UPDATE_FORMAT = "prime-rl-sparse-update"
SPARSE_UPDATE_VERSION = 1
SPARSE_UPDATE_MANIFEST_NAME = "sparse_update_manifest.json"
SPARSE_UPDATE_PATCH_NAME = "sparse_update_patch.safetensors"

# zstd frame magic number (RFC 8478)
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass(frozen=True)
class SparseUpdateStats:
    total_tensors: int
    patched_tensors: int
    total_numel: int
    changed_numel: int
    patch_bytes: int
    diff_s: float = 0.0
    save_s: float = 0.0

    @property
    def sparsity(self) -> float:
        if self.total_numel == 0:
            return 1.0
        return 1.0 - (self.changed_numel / self.total_numel)


def is_sparse_update_dir(path: Path) -> bool:
    return (path / SPARSE_UPDATE_MANIFEST_NAME).exists()


def to_compute_tensor(
    tensor: Tensor, compute_dtype: torch.dtype = torch.bfloat16, *, device: str | torch.device = "cpu"
) -> Tensor:
    tensor = tensor.detach()
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=compute_dtype)
    return tensor.to(device=device, copy=True).contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported tensor dtype in sparse update patch: {name}")
    return dtype


def _idx_dtype(numel: int) -> torch.dtype:
    """Pick the narrowest integer dtype that can address ``numel`` elements."""
    return torch.int32 if numel < 2**31 else torch.int64


def _checksum(indices: Tensor, values: Tensor) -> int:
    """Compute a XOR hash over indices and values for bit-corruption detection."""
    return torch.hash_tensor(indices.contiguous()).item() ^ torch.hash_tensor(values.contiguous()).item()


def _save_patch(patch_tensors: dict[str, Tensor], patch_path: Path, *, compress: bool) -> None:
    """Write sparse patch tensors to disk, optionally zstd-compressed."""
    if not patch_tensors:
        return
    if compress:
        import pyzstd

        blob = save(patch_tensors, metadata={"format": "pt"})
        blob = pyzstd.compress(blob)
        tmp_path = patch_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            f.write(blob)
        tmp_path.replace(patch_path)
    else:
        save_file(patch_tensors, patch_path, metadata={"format": "pt"})


def _load_patch_tensors(patch_path: Path, *, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Load sparse patch tensors, transparently decompressing zstd if detected."""
    with open(patch_path, "rb") as f:
        magic = f.read(4)
    if magic == ZSTD_MAGIC:
        import pyzstd

        with open(patch_path, "rb") as f:
            blob = pyzstd.decompress(f.read())
        tensors = load(blob)
        if str(device) != "cpu":
            tensors = {k: v.to(device) for k, v in tensors.items()}
        return tensors
    return load_file(patch_path, device=str(device))


def save_sparse_update(
    previous_state: dict[str, Tensor],
    current_state: Mapping[str, Tensor],
    save_dir: Path,
    *,
    step: int,
    base_step: int,
    compute_dtype: torch.dtype | None = torch.bfloat16,
    device: str | torch.device = "cpu",
    compress: bool = False,
) -> SparseUpdateStats:
    """Save changed values between two dense state dicts.

    When ``compute_dtype`` is set, tensors are cast to that dtype before diffing
    (e.g. BF16 to skip changes invisible after quantization). When ``None``, the
    native dtype of each tensor is preserved — used for kernel-format patches
    where quantization has already been applied.

    When ``device`` is not ``"cpu"``, each current tensor is moved to the GPU,
    the diff and nonzero are performed there, and only the sparse indices and
    values are brought back to CPU. The ``previous_state`` baseline stays on CPU
    and is refreshed in-place after each tensor is processed.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = str(device) != "cpu"
    patch_tensors: dict[str, Tensor] = {}
    tensor_entries: list[dict] = []
    total_numel = 0
    changed_numel = 0

    diff_start = time.perf_counter()

    for tensor_idx, (name, current) in enumerate(current_state.items()):
        if use_gpu:
            current_compute = current.detach().to(device=device)
            if compute_dtype is not None and current_compute.is_floating_point():
                current_compute = current_compute.to(dtype=compute_dtype)
            current_compute = current_compute.contiguous()
        elif compute_dtype is not None:
            current_compute = to_compute_tensor(current, compute_dtype)
        else:
            current_compute = current.detach().to(device="cpu", copy=True).contiguous()

        previous = previous_state.get(name)
        numel = current_compute.numel()
        total_numel += numel
        idx_dtype = _idx_dtype(numel)

        if previous is None or tuple(previous.shape) != tuple(current_compute.shape):
            indices = torch.arange(numel, dtype=idx_dtype, device=current_compute.device)
        elif use_gpu:
            prev_gpu = previous.to(device=device, dtype=current_compute.dtype, non_blocking=True).contiguous()
            changed = current_compute.reshape(-1).ne(prev_gpu.reshape(-1))
            indices = changed.nonzero(as_tuple=False).reshape(-1).to(idx_dtype)
        else:
            previous_compute = previous.to(dtype=current_compute.dtype, device="cpu").contiguous()
            changed = current_compute.reshape(-1).ne(previous_compute.reshape(-1))
            indices = torch.nonzero(changed, as_tuple=False).reshape(-1).to(idx_dtype)

        if indices.numel() == 0:
            if use_gpu and previous is not None and tuple(previous.shape) == tuple(current_compute.shape):
                previous.copy_(current_compute.cpu())
            elif use_gpu:
                previous_state[name] = current_compute.cpu()
            continue

        if use_gpu:
            values = current_compute.reshape(-1).index_select(0, indices.to(torch.long)).contiguous()
            indices = indices.cpu()
            values = values.cpu()
        else:
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
                "numel": numel,
                "changed_numel": indices.numel(),
                "checksum": _checksum(indices, values),
            }
        )

        # Refresh the baseline so the next call diffs against this version.
        if use_gpu:
            current_cpu = current_compute.cpu()
            if previous is not None and tuple(previous.shape) == tuple(current_cpu.shape):
                previous.copy_(current_cpu)
            else:
                previous_state[name] = current_cpu

    diff_s = time.perf_counter() - diff_start

    save_start = time.perf_counter()
    _save_patch(patch_tensors, save_dir / SPARSE_UPDATE_PATCH_NAME, compress=compress)
    save_s = time.perf_counter() - save_start

    patch_file = save_dir / SPARSE_UPDATE_PATCH_NAME
    patch_bytes = patch_file.stat().st_size if patch_file.exists() else 0
    stats = SparseUpdateStats(
        total_tensors=len(current_state),
        patched_tensors=len(tensor_entries),
        total_numel=total_numel,
        changed_numel=changed_numel,
        patch_bytes=patch_bytes,
        diff_s=diff_s,
        save_s=save_s,
    )
    manifest = {
        "format": SPARSE_UPDATE_FORMAT,
        "version": SPARSE_UPDATE_VERSION,
        "step": step,
        "base_step": base_step,
        "compute_dtype": _dtype_name(compute_dtype) if compute_dtype is not None else None,
        "compressed": compress if patch_tensors else False,
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


def _verify_checksum(entry: dict, indices: Tensor, values: Tensor) -> None:
    """Verify the per-tensor checksum if the manifest entry carries one."""
    expected = entry.get("checksum")
    if expected is None:
        return
    actual = _checksum(indices, values)
    if actual != expected:
        raise ValueError(f"Sparse update checksum mismatch for {entry['name']}: expected {expected}, got {actual}")


def apply_sparse_update(
    state_dict: dict[str, Tensor], patch_dir: Path, *, expected_base_step: int | None = None
) -> dict:
    """Apply a sparse value update to a dense CPU state dict in-place."""
    manifest = load_sparse_update_manifest(patch_dir)
    if expected_base_step is not None and manifest["base_step"] != expected_base_step:
        raise ValueError(f"Sparse update base step mismatch: cache={expected_base_step} patch={manifest['base_step']}")

    patch_file = manifest.get("patch_file")
    patch_tensors = _load_patch_tensors(patch_dir / patch_file, device="cpu") if patch_file else {}

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
        _verify_checksum(entry, indices, values)

        tensor.reshape(-1).index_copy_(0, indices, values)
        state_dict[name] = tensor.contiguous()

    return manifest


@torch.no_grad()
def apply_sparse_update_to_params(
    model: "Module",
    patch_dir: Path,
    *,
    expected_base_step: int | None = None,
    device: str | torch.device = "cuda",
) -> dict:
    """Apply a sparse value update directly to GPU model parameters in-place.

    Unlike ``apply_sparse_update`` which operates on a dense CPU state dict,
    this function loads patch tensors directly onto the target device and
    scatters changed values into the model's named parameters via ``index_copy_``.
    No CPU cache is required.
    """

    manifest = load_sparse_update_manifest(patch_dir)
    if expected_base_step is not None and manifest["base_step"] != expected_base_step:
        raise ValueError(f"Sparse update base step mismatch: cache={expected_base_step} patch={manifest['base_step']}")

    patch_file = manifest.get("patch_file")
    patch_tensors = _load_patch_tensors(patch_dir / patch_file, device=str(device)) if patch_file else {}

    params = dict(model.named_parameters())
    for entry in manifest["tensors"]:
        name = entry["name"]
        if name not in params:
            raise KeyError(f"Sparse update refers to parameter not present in model: {name}")

        param = params[name]
        expected_shape = tuple(entry["shape"])
        if tuple(param.shape) != expected_shape:
            raise ValueError(
                f"Sparse update shape mismatch for {name}: param={tuple(param.shape)} patch={expected_shape}"
            )

        dtype = _dtype_from_name(entry["dtype"])
        if param.dtype != dtype:
            raise ValueError(f"Sparse update dtype mismatch for {name}: param={param.dtype} patch={dtype}")

        indices = patch_tensors[entry["indices"]].to(dtype=torch.long, device=device)
        values = patch_tensors[entry["values"]].to(dtype=dtype, device=device)

        _verify_checksum(entry, indices, values)

        param.data.reshape(-1).index_copy_(0, indices, values)

    return manifest
