"""Experimental sparse filesystem policy updates in HF checkpoint coordinates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.optim import Optimizer

FORMAT = "prime-rl-sparse-hf-v1"


@dataclass(frozen=True)
class TensorPatch:
    name: str
    global_shape: tuple[int, ...]
    dtype: str
    indices_key: str
    values_key: str
    changed: int


@dataclass(frozen=True)
class RankPatch:
    rank: int
    base_step: int
    target_step: int
    file: str
    tensors: tuple[TensorPatch, ...]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _global_indices(
    local_indices: torch.Tensor, local_shape: tuple[int, ...], global_shape: tuple[int, ...], offset: tuple[int, ...]
) -> torch.Tensor:
    coordinates = torch.unravel_index(local_indices, local_shape)
    global_coordinates = tuple(coordinate + start for coordinate, start in zip(coordinates, offset))
    strides = [math.prod(global_shape[index + 1 :]) for index in range(len(global_shape))]
    result = torch.zeros_like(local_indices, dtype=torch.int64)
    for coordinate, stride in zip(global_coordinates, strides):
        result += coordinate * stride
    return result


def _local_tensor(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    if not isinstance(value, DTensor):
        return value.detach(), tuple(value.shape), (0,) * value.ndim, 0, 1
    if any(isinstance(placement, Partial) for placement in value.placements):
        raise ValueError("Partial DTensors cannot be sparsely published")
    shape, offset = compute_local_shape_and_global_offset(value.shape, value.device_mesh, value.placements)
    coordinate = value.device_mesh.get_coordinate()
    replica_dims = [index for index, placement in enumerate(value.placements) if isinstance(placement, Replicate)]
    replica_index, replica_count = 0, 1
    for dimension in replica_dims:
        replica_index = replica_index * value.device_mesh.shape[dimension] + coordinate[dimension]
        replica_count *= value.device_mesh.shape[dimension]
    local = value.to_local().detach()
    if tuple(local.shape) != tuple(shape):
        local = local[tuple(slice(size) for size in shape)]
    return local, tuple(value.shape), tuple(offset), replica_index, replica_count


class SparseUpdateWriter:
    """Keep one BF16 local-shard baseline and emit sorted global indices/new values."""

    def __init__(self, root: Path, *, rank: int, serving_dtype: torch.dtype = torch.bfloat16) -> None:
        self.root, self.rank, self.serving_dtype = root, rank, serving_dtype
        self._baseline: dict[str, torch.Tensor] = {}
        self._initialized = False
        self.base_step = 0

    def initialize(self, state_dict: Mapping[str, torch.Tensor], *, base_step: int = 0) -> None:
        self._baseline = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                local, _, _, replica_index, _ = _local_tensor(value)
                if replica_index == 0:
                    self._baseline[name] = local.to("cpu", dtype=self.serving_dtype, copy=True).contiguous()
        self.base_step = base_step
        self._initialized = True

    def write(self, state_dict: Mapping[str, torch.Tensor], *, target_step: int) -> RankPatch:
        if not self._initialized:
            raise RuntimeError("initialize sparse baseline before optimizer.step")
        directory = self.root / f"step_{target_step}"
        directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, torch.Tensor] = {}
        entries: list[TensorPatch] = []
        for tensor_index, (name, value) in enumerate(state_dict.items()):
            if name not in self._baseline:
                continue
            local, global_shape, offset, replica_index, _ = _local_tensor(value)
            if replica_index != 0:
                continue
            current = local.to("cpu", dtype=self.serving_dtype, copy=True).contiguous()
            baseline = self._baseline[name]
            if tuple(current.shape) != tuple(baseline.shape):
                raise ValueError(f"local shape changed for {name}")
            changed_local = current.view(-1).ne(baseline.view(-1)).nonzero().view(-1)
            if not changed_local.numel():
                baseline.copy_(current)
                continue
            indices = _global_indices(changed_local, tuple(current.shape), global_shape, offset)
            if math.prod(global_shape) < 2**31:
                indices = indices.to(torch.int32)
            indices, order = indices.sort()
            values = current.view(-1).index_select(0, changed_local).index_select(0, order).contiguous()
            indices_key, values_key = f"t{tensor_index}_indices", f"t{tensor_index}_values"
            payload[indices_key], payload[values_key] = indices, values
            entries.append(
                TensorPatch(
                    name,
                    global_shape,
                    str(self.serving_dtype).removeprefix("torch."),
                    indices_key,
                    values_key,
                    indices.numel(),
                )
            )
            baseline.copy_(current)
        file = f"rank_{self.rank}.safetensors"
        if payload:
            save_file(payload, directory / file)
        patch = RankPatch(self.rank, self.base_step, target_step, file, tuple(entries))
        _atomic_json(directory / f"rank_{self.rank}.json", asdict(patch))
        self.base_step = target_step
        return patch


def commit_sparse_update(root: Path, *, target_step: int, base_step: int, world_size: int) -> Path:
    directory = root / f"step_{target_step}"
    ranks = []
    for rank in range(world_size):
        data = json.loads((directory / f"rank_{rank}.json").read_text())
        if data["base_step"] != base_step or data["target_step"] != target_step:
            raise ValueError("sparse patch chain mismatch")
        ranks.append(data)
    changed = sum(tensor["changed"] for rank in ranks for tensor in rank["tensors"])
    payload_bytes = sum((directory / rank["file"]).stat().st_size for rank in ranks if rank["tensors"])
    manifest = {
        "format": FORMAT,
        "base_step": base_step,
        "target_step": target_step,
        "changed": changed,
        "payload_bytes": payload_bytes,
        "ranks": ranks,
    }
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = directory / "sparse_manifest.json"
    _atomic_json(path, manifest)
    (directory / "STABLE").touch()
    return path


def apply_sparse_update(state_dict: dict[str, torch.Tensor], directory: Path, *, expected_base_step: int) -> int:
    manifest = json.loads((directory / "sparse_manifest.json").read_text())
    checksum = manifest.pop("sha256", None)
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if checksum != actual:
        raise ValueError("sparse patch manifest checksum mismatch")
    if manifest["format"] != FORMAT or manifest["base_step"] != expected_base_step:
        raise ValueError("sparse patch base mismatch")
    for rank in manifest["ranks"]:
        tensors = load_file(directory / rank["file"]) if rank["tensors"] else {}
        for entry in rank["tensors"]:
            target = state_dict[entry["name"]]
            if tuple(target.shape) != tuple(entry["global_shape"]):
                raise ValueError(f"shape mismatch for {entry['name']}")
            indices = tensors[entry["indices_key"]].long()
            if indices.numel() and (indices[0] < 0 or indices[-1] >= target.numel()):
                raise ValueError(f"index out of bounds for {entry['name']}")
            if indices.numel() > 1 and not bool(torch.all(indices[1:] > indices[:-1])):
                raise ValueError(f"indices are not strictly sorted for {entry['name']}")
            values = tensors[entry["values_key"]].to(target.dtype)
            target.view(-1).index_copy_(0, indices, values)
    return manifest["target_step"]


class OptimizerSparseUpdateHook:
    def __init__(
        self,
        optimizer: Optimizer | Any,
        writer: SparseUpdateWriter,
        state_dict: Callable[[], Mapping[str, torch.Tensor]],
        step: Callable[[], int],
    ) -> None:
        self.writer, self.state_dict, self.step = writer, state_dict, step
        self.handle = getattr(optimizer, "base_optimizer", optimizer).register_step_post_hook(self._after_step)

    def _after_step(self, optimizer, args, kwargs) -> None:
        self.writer.write(self.state_dict(), target_step=self.step())

    def remove(self) -> None:
        self.handle.remove()
