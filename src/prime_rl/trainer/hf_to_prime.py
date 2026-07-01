import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem
from torch.futures import Future

_LAYER_KEY_RE = re.compile(r"^(?:model(?:\.language_model)?|backbone)\.layers\.(?P<layer>\d+)\.")


@dataclass(frozen=True)
class _ConvertedStorageInfo:
    group_idx: int


@dataclass(frozen=True)
class _SourceChunk:
    key: str
    chunk: ChunkStorageMetadata
    tensor_metadata: TensorStorageMetadata


class HFToPrimeStorageReader(HuggingFaceStorageReader):
    """Stream HF safetensors through a model's layer conversion code without writing a converted snapshot."""

    def __init__(
        self,
        path: str,
        convert_layer_to_prime: Callable[[dict[str, Tensor], int], dict[str, Tensor]],
        thread_count: int = 1,
    ) -> None:
        super().__init__(path=path, thread_count=thread_count)
        self.convert_layer_to_prime = convert_layer_to_prime
        self.source_metadata: Metadata | None = None
        self.source_storage_data: dict[MetadataIndex, object] = {}
        self.source_groups: dict[int, list[str]] = {}
        self.target_groups: dict[str, int] = {}

    def read_metadata(self) -> Metadata:
        source_metadata = super().read_metadata()
        self.source_metadata = source_metadata
        self.source_storage_data = source_metadata.storage_data
        self.source_groups = self._group_source_keys(source_metadata)

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _ConvertedStorageInfo] = {}

        for group_idx, source_keys in self.source_groups.items():
            with FakeTensorMode():
                fake_state_dict = self._make_fake_state_dict(source_keys, source_metadata)
                fake_state_dict = self.convert_layer_to_prime(fake_state_dict, group_idx)

            for key, tensor in fake_state_dict.items():
                if not isinstance(tensor, Tensor):
                    continue
                offsets = torch.Size([0] * tensor.ndim)
                state_dict_metadata[key] = TensorStorageMetadata(
                    properties=TensorProperties.create_from_tensor(tensor),
                    size=tensor.size(),
                    chunks=[ChunkStorageMetadata(offsets=offsets, sizes=tensor.size())],
                )
                storage_data[MetadataIndex(key, offsets)] = _ConvertedStorageInfo(group_idx)
                self.target_groups[key] = group_idx

        return Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
            storage_meta=StorageMeta(load_id=self.load_id),
        )

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        read_items_by_group: dict[int, list[ReadItem]] = {}
        for read_item in plan.items:
            group_idx = self.target_groups[read_item.storage_index.fqn]
            read_items_by_group.setdefault(group_idx, []).append(read_item)

        for group_idx, read_items in read_items_by_group.items():
            state_dict = self._load_source_group(group_idx)
            state_dict = self.convert_layer_to_prime(state_dict, group_idx)
            for read_item in read_items:
                self._copy_read_item(state_dict, read_item, planner)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    def converted_state_dicts(self) -> Iterator[tuple[int, dict[str, Tensor]]]:
        if self.source_metadata is None:
            self.read_metadata()

        for group_idx in sorted(self.source_groups):
            state_dict = self._load_source_group(group_idx)
            yield group_idx, self.convert_layer_to_prime(state_dict, group_idx)

    def _group_source_keys(self, metadata: Metadata) -> dict[int, list[str]]:
        groups: dict[int, list[str]] = {}
        for key, tensor_metadata in metadata.state_dict_metadata.items():
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                continue
            group_idx = self._group_idx(key)
            groups.setdefault(group_idx, []).append(key)
        return groups

    def _make_fake_state_dict(self, source_keys: list[str], metadata: Metadata) -> dict[str, Tensor]:
        state_dict = {}
        for key in source_keys:
            tensor_metadata = cast(TensorStorageMetadata, metadata.state_dict_metadata[key])
            state_dict[key] = torch.empty(tensor_metadata.size, dtype=tensor_metadata.properties.dtype)
        return state_dict

    def _load_source_group(self, group_idx: int) -> dict[str, Tensor]:
        from safetensors import safe_open

        if self.source_metadata is None:
            raise AssertionError("source metadata must be initialized before reading data")

        chunks_by_file: dict[str, list[_SourceChunk]] = {}
        for key in self.source_groups[group_idx]:
            tensor_metadata = self.source_metadata.state_dict_metadata[key]
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                continue
            for chunk in tensor_metadata.chunks:
                storage_info = self.source_storage_data[MetadataIndex(key, chunk.offsets)]
                chunks_by_file.setdefault(storage_info.relative_path, []).append(
                    _SourceChunk(key=key, chunk=chunk, tensor_metadata=tensor_metadata)
                )

        state_dict: dict[str, Tensor] = {}
        for file_name, source_chunks in chunks_by_file.items():
            with safe_open(filename=file_name, framework="pt", device="cpu") as f:
                for source_chunk in source_chunks:
                    self._set_source_chunk(state_dict, source_chunk, f.get_tensor(source_chunk.key))
        return state_dict

    @staticmethod
    def _set_source_chunk(state_dict: dict[str, Tensor], source_chunk: _SourceChunk, tensor: Tensor) -> None:
        key = source_chunk.key
        chunk = source_chunk.chunk
        tensor_metadata = source_chunk.tensor_metadata
        if chunk.offsets == torch.Size([0] * len(chunk.offsets)) and chunk.sizes == tensor_metadata.size:
            state_dict[key] = tensor
            return
        if key not in state_dict:
            state_dict[key] = torch.empty(tensor_metadata.size, dtype=tensor_metadata.properties.dtype)
        slices = tuple(slice(offset, offset + length) for offset, length in zip(chunk.offsets, chunk.sizes))
        state_dict[key][slices].copy_(tensor)

    def _copy_read_item(self, state_dict: dict[str, Tensor], read_item: ReadItem, planner: LoadPlanner) -> None:
        tensor = state_dict[read_item.storage_index.fqn]
        slices = tuple(
            slice(offset, offset + length) for offset, length in zip(read_item.storage_offsets, read_item.lengths)
        )
        tensor = tensor[slices]
        target_tensor = planner.resolve_tensor(read_item).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {read_item.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            )

        target_tensor.copy_(tensor)
        planner.commit_tensor(read_item, target_tensor)

    @staticmethod
    def _group_idx(key: str) -> int:
        match = _LAYER_KEY_RE.match(key)
        if match is None:
            return -1
        return int(match.group("layer"))


def materialize_hf_to_prime(
    path: Path,
    output_path: Path,
    convert_layer_to_prime: Callable[[dict[str, Tensor], int], dict[str, Tensor]],
) -> None:
    from safetensors.torch import save_file

    if output_path.exists():
        return

    reader = HFToPrimeStorageReader(path=path.as_posix(), convert_layer_to_prime=convert_layer_to_prime)
    reader.read_metadata()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))

    try:
        group_indices = sorted(reader.source_groups)
        num_shards = len(group_indices)
        total_size = 0
        weight_map: dict[str, str] = {}
        for shard_idx, (_group_idx, state_dict) in enumerate(reader.converted_state_dicts(), start=1):
            tensors = {
                key: tensor.detach().cpu().contiguous()
                for key, tensor in state_dict.items()
                if isinstance(tensor, Tensor)
            }
            if not tensors:
                continue
            shard_name = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            save_file(tensors, tmp_path / shard_name, metadata={"format": "pt"})
            for key, tensor in tensors.items():
                weight_map[key] = shard_name
                total_size += tensor.numel() * tensor.element_size()
            del tensors
            del state_dict
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
        tmp_path.rename(output_path)
    except FileExistsError:
        shutil.rmtree(tmp_path)
        if not output_path.exists():
            raise
    except Exception:
        shutil.rmtree(tmp_path)
        raise


__all__ = ["HFToPrimeStorageReader", "materialize_hf_to_prime"]
