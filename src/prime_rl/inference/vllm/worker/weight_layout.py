from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import Module, Parameter


@dataclass(frozen=True)
class LogicalSparseUpdate:
    name: str
    shape: tuple[int, ...]
    indices: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class LocalSparseUpdate:
    name: str
    indices: torch.Tensor
    values: torch.Tensor


class ModelLayoutAdapter(Protocol):
    def to_local(self, update: LogicalSparseUpdate) -> LocalSparseUpdate: ...


@dataclass(frozen=True)
class _LayoutMapping:
    target_name: str
    kind: str
    shard_id: str | int | None = None


_FUSED_SUFFIX_MAPPINGS = (
    (".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", "qkv", "q"),
    (".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", "qkv", "k"),
    (".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", "qkv", "v"),
    (".self_attn.q_proj.bias", ".self_attn.qkv_proj.bias", "qkv", "q"),
    (".self_attn.k_proj.bias", ".self_attn.qkv_proj.bias", "qkv", "k"),
    (".self_attn.v_proj.bias", ".self_attn.qkv_proj.bias", "qkv", "v"),
    (".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", "merged_column", 0),
    (".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", "merged_column", 1),
    (".mlp.gate_proj.bias", ".mlp.gate_up_proj.bias", "merged_column", 0),
    (".mlp.up_proj.bias", ".mlp.gate_up_proj.bias", "merged_column", 1),
)


class DenseVllmModelLayoutAdapter:
    """Map logical Hugging Face sparse updates to unquantized vLLM dense-model shards."""

    def __init__(self, model: Module):
        self._params = dict(model.named_parameters())
        self._modules = dict(model.named_modules())

    def to_local(self, update: LogicalSparseUpdate) -> LocalSparseUpdate:
        mapping = self._mapping(update.name)
        param = self._params[mapping.target_name]
        module = self._parameter_module(mapping.target_name)
        self._validate_supported_parameter(mapping.target_name, param, module)
        self._validate_update(update)

        if mapping.kind == "qkv":
            indices, mask = self._project_qkv(update, param, module, str(mapping.shard_id))
        elif mapping.kind == "merged_column":
            indices, mask = self._project_merged_column(update, param, module, int(mapping.shard_id))
        elif mapping.kind == "row":
            indices, mask = self._project_regular(update, param, module, "input_dim")
        elif mapping.kind == "column":
            indices, mask = self._project_regular(update, param, module, "output_dim")
        elif mapping.kind == "vocab":
            indices, mask = self._project_vocab(update, param, module)
        elif mapping.kind == "replicated":
            if tuple(param.shape) != update.shape:
                raise ValueError(
                    f"replicated parameter shape mismatch for {update.name}: "
                    f"logical={update.shape} local={tuple(param.shape)}"
                )
            indices = update.indices
            mask = torch.ones(update.indices.numel(), dtype=torch.bool)
        else:
            raise ValueError(f"unsupported sparse weight layout kind: {mapping.kind}")

        return LocalSparseUpdate(
            name=mapping.target_name,
            indices=indices,
            values=update.values.reshape(-1)[mask],
        )

    def _mapping(self, source_name: str) -> _LayoutMapping:
        if source_name in self._params:
            module = self._parameter_module(source_name)
            return _LayoutMapping(source_name, _direct_layout_kind(module, self._params[source_name]))

        for source_suffix, target_suffix, kind, shard_id in _FUSED_SUFFIX_MAPPINGS:
            if not source_name.endswith(source_suffix):
                continue
            target_name = source_name.removesuffix(source_suffix) + target_suffix
            if target_name not in self._params:
                break
            return _LayoutMapping(target_name, kind, shard_id)

        raise ValueError(f"logical delta parameter is not supported by the inference model: {source_name}")

    def _parameter_module(self, parameter_name: str) -> Module:
        module_name, separator, _parameter = parameter_name.rpartition(".")
        if not separator:
            module_name = ""
        if module_name not in self._modules:
            raise ValueError(f"cannot resolve inference module for parameter: {parameter_name}")
        return self._modules[module_name]

    @staticmethod
    def _validate_supported_parameter(name: str, param: Parameter, module: Module) -> None:
        if getattr(module, "quant_config", None) is not None:
            raise ValueError(f"quantized inference layout is not supported for sparse delta updates: {name}")
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and type(quant_method).__name__ not in {
            "UnquantizedEmbeddingMethod",
            "UnquantizedLinearMethod",
        }:
            raise ValueError(f"quantized inference layout is not supported for sparse delta updates: {name}")
        if getattr(param, "packed_dim", None) is not None or getattr(param, "use_bitsandbytes_4bit", False):
            raise ValueError(f"packed inference parameter is not supported for sparse delta updates: {name}")

    @staticmethod
    def _validate_update(update: LogicalSparseUpdate) -> None:
        if update.indices.dtype != torch.int64:
            raise ValueError("logical sparse indices must be decoded to int64")
        if update.indices.numel() != update.values.numel():
            raise ValueError(
                f"logical sparse index/value length mismatch for {update.name}: "
                f"{update.indices.numel()} vs {update.values.numel()}"
            )
        if update.indices.numel() > 1 and torch.any(update.indices[1:] <= update.indices[:-1]):
            raise ValueError(f"logical sparse indices must be strictly increasing for {update.name}")
        logical_numel = math.prod(update.shape)
        if update.indices.numel() and (
            int(update.indices[0].item()) < 0 or int(update.indices[-1].item()) >= logical_numel
        ):
            raise ValueError(f"logical sparse index out of range for {update.name}")

    @staticmethod
    def _project_regular(
        update: LogicalSparseUpdate,
        param: Parameter,
        module: Module,
        dimension_attribute: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimension = getattr(param, dimension_attribute, None)
        if dimension is None:
            raise ValueError(f"inference parameter has no {dimension_attribute}: {update.name}")
        tp_rank, tp_size = _tensor_parallel_position(module)
        if update.shape[dimension] % tp_size != 0:
            raise ValueError(
                f"logical dimension is not divisible by TP size for {update.name}: "
                f"shape={update.shape}, dim={dimension}, tp={tp_size}"
            )
        shard_size = update.shape[dimension] // tp_size
        return _project_dimension(
            update.indices,
            update.shape,
            tuple(param.shape),
            dimension=dimension,
            source_start=tp_rank * shard_size,
            source_size=shard_size,
            target_start=0,
        )

    @staticmethod
    def _project_vocab(
        update: LogicalSparseUpdate,
        param: Parameter,
        module: Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard_indices = getattr(module, "shard_indices", None)
        if shard_indices is None:
            raise ValueError(f"vocabulary-parallel module has no shard indices: {update.name}")
        start = int(shard_indices.org_vocab_start_index)
        end = int(shard_indices.org_vocab_end_index)
        return _project_dimension(
            update.indices,
            update.shape,
            tuple(param.shape),
            dimension=0,
            source_start=start,
            source_size=end - start,
            target_start=0,
        )

    @staticmethod
    def _project_merged_column(
        update: LogicalSparseUpdate,
        param: Parameter,
        module: Module,
        shard_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_sizes = tuple(int(size) for size in module.output_sizes)
        if shard_id >= len(output_sizes):
            raise ValueError(f"invalid merged-column shard id for {update.name}: {shard_id}")
        if update.shape[0] != output_sizes[shard_id]:
            raise ValueError(
                f"merged-column source shape mismatch for {update.name}: "
                f"logical={update.shape[0]} expected={output_sizes[shard_id]}"
            )
        tp_rank, tp_size = _tensor_parallel_position(module)
        if any(size % tp_size != 0 for size in output_sizes):
            raise ValueError(f"merged-column sizes are not divisible by TP size for {update.name}")
        local_sizes = tuple(size // tp_size for size in output_sizes)
        local_size = local_sizes[shard_id]
        return _project_dimension(
            update.indices,
            update.shape,
            tuple(param.shape),
            dimension=0,
            source_start=tp_rank * local_size,
            source_size=local_size,
            target_start=sum(local_sizes[:shard_id]),
        )

    @staticmethod
    def _project_qkv(
        update: LogicalSparseUpdate,
        param: Parameter,
        module: Module,
        shard_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_size = int(module.head_size)
        v_head_size = int(getattr(module, "v_head_size", head_size))
        global_sizes = {
            "q": int(module.total_num_heads) * head_size,
            "k": int(module.total_num_kv_heads) * head_size,
            "v": int(module.total_num_kv_heads) * v_head_size,
        }
        local_sizes = {
            "q": int(module.num_heads) * head_size,
            "k": int(module.num_kv_heads) * head_size,
            "v": int(module.num_kv_heads) * v_head_size,
        }
        if update.shape[0] != global_sizes[shard_id]:
            raise ValueError(
                f"QKV source shape mismatch for {update.name}: "
                f"logical={update.shape[0]} expected={global_sizes[shard_id]}"
            )

        tp_rank, _tp_size = _tensor_parallel_position(module)
        replicas = int(getattr(module, "num_kv_head_replicas", 1))
        source_rank = tp_rank if shard_id == "q" else tp_rank // replicas
        target_start = {
            "q": 0,
            "k": local_sizes["q"],
            "v": local_sizes["q"] + local_sizes["k"],
        }[shard_id]
        return _project_dimension(
            update.indices,
            update.shape,
            tuple(param.shape),
            dimension=0,
            source_start=source_rank * local_sizes[shard_id],
            source_size=local_sizes[shard_id],
            target_start=target_start,
        )


def _direct_layout_kind(module: Module, param: Parameter) -> str:
    override = getattr(module, "prime_rl_shard_kind", None)
    if override is not None:
        return str(override)
    if hasattr(module, "shard_indices") and hasattr(module, "num_embeddings_per_partition"):
        return "vocab"
    class_name = type(module).__name__
    if "RowParallelLinear" in class_name:
        return "row" if getattr(param, "input_dim", None) is not None else "replicated"
    if "ColumnParallelLinear" in class_name or "ParallelLMHead" in class_name:
        return "column" if getattr(param, "output_dim", None) is not None else "replicated"
    return "replicated"


def _tensor_parallel_position(module: Module) -> tuple[int, int]:
    tp_rank = int(getattr(module, "tp_rank", 0))
    tp_size = int(getattr(module, "tp_size", 1))
    if tp_size < 1 or tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"invalid tensor-parallel position: rank={tp_rank}, size={tp_size}")
    return tp_rank, tp_size


def _project_dimension(
    indices: torch.Tensor,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    dimension: int,
    source_start: int,
    source_size: int,
    target_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(source_shape) != len(target_shape):
        raise ValueError(f"source/target tensor ranks differ: {source_shape} vs {target_shape}")
    if dimension < 0:
        dimension += len(source_shape)
    if dimension < 0 or dimension >= len(source_shape):
        raise ValueError(f"invalid sharding dimension {dimension} for shape {source_shape}")
    for axis, (source_dim, target_dim) in enumerate(zip(source_shape, target_shape)):
        if axis != dimension and source_dim != target_dim:
            raise ValueError(f"non-sharded dimensions differ: {source_shape} vs {target_shape}")
    if source_start < 0 or source_size < 0 or source_start + source_size > source_shape[dimension]:
        raise ValueError(
            f"invalid source shard [{source_start}, {source_start + source_size}) for shape {source_shape}"
        )
    if target_start < 0 or target_start + source_size > target_shape[dimension]:
        raise ValueError(
            f"invalid target shard [{target_start}, {target_start + source_size}) for shape {target_shape}"
        )

    inner_size = math.prod(source_shape[dimension + 1 :])
    source_block = source_shape[dimension] * inner_size
    coordinates = torch.div(indices, inner_size, rounding_mode="floor") % source_shape[dimension]
    mask = (coordinates >= source_start) & (coordinates < source_start + source_size)
    selected = indices[mask]
    selected_coordinates = coordinates[mask]
    outer = torch.div(selected, source_block, rounding_mode="floor")
    inner = selected % inner_size
    local_coordinates = selected_coordinates - source_start + target_start
    target_block = target_shape[dimension] * inner_size
    local_indices = outer * target_block + local_coordinates * inner_size + inner
    return local_indices.to(torch.int64), mask
