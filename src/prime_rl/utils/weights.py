import json
import re
import warnings
from pathlib import Path
from typing import Callable, cast

import torch
import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)

from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory as safetensors."""
    logger = get_logger()
    weights_name = ADAPTER_SAFE_WEIGHTS_NAME if adapter else SAFE_WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = save_dir / SAFE_WEIGHTS_INDEX_NAME
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})


def convert_state_dict_to_hf(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a (possibly rank-partial) training-format state dict to HF hub format.

    Format detection uses the model's full key set, so a partial dict (one rank's
    slice) converts the same way as the full state dict would.
    """
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    full_keys = dict.fromkeys(resolve_fqn(model, key) for key in model.state_dict().keys())
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(full_keys):
        # PrimeRL custom model holding weights in prime format: apply the model's
        # declarative prime->HF conversion chain (renames, expert stack/unstack).
        return model.convert_to_hf(state_dict)
    else:
        # Plain transformers model: undo the key renames transformers applied when
        # it loaded the HF checkpoint.
        from transformers.core_model_loading import revert_weight_conversion

        return revert_weight_conversion(model, state_dict)


def resolve_fqn(model: nn.Module, key: str) -> str:
    """Resolve a state-dict key to the parameter's canonical fully-qualified name.

    Strips wrapper prefixes that training composes onto module paths, e.g. with
    ``torch.compile`` the key ``model._orig_mod.layers.0.self_attn.q_proj.weight``
    resolves to ``model.layers.0.self_attn.q_proj.weight`` — the name the tensor
    has in the HF checkpoint.
    """
    fqns = get_fqns(model, key)
    assert len(fqns) == 1
    return next(iter(fqns))


_LAYER_RE = re.compile(r"^(.*?\blayers\.\d+)\.")


def partition_weights(state_dict: dict[str, Tensor], world_size: int, dtype: torch.dtype) -> dict[str, int]:
    """Assign each state-dict key an owner rank, balanced by byte size.

    All keys of one decoder layer share an owner, so per-rank prime->HF conversion
    (which stacks/unstacks tensors within a layer) always sees complete layers.
    """
    key_to_unit: dict[str, str] = {}
    unit_bytes: dict[str, int] = {}
    for key, value in state_dict.items():
        match = _LAYER_RE.match(key)
        unit = match.group(1) if match else ""
        key_to_unit[key] = unit
        itemsize = dtype.itemsize if isinstance(value, DTensor) else value.element_size()
        unit_bytes[unit] = unit_bytes.get(unit, 0) + value.numel() * itemsize
    loads = [0] * world_size
    unit_owner: dict[str, int] = {}
    for unit in sorted(unit_bytes, key=lambda unit: (-unit_bytes[unit], unit)):
        owner = min(range(world_size), key=lambda rank: loads[rank])
        unit_owner[unit] = owner
        loads[owner] += unit_bytes[unit]
    return {key: unit_owner[unit] for key, unit in key_to_unit.items()}


def resolve_wire_dtype(keep_in_fp32: Callable[[str], bool] | None, key: str, default: torch.dtype) -> torch.dtype:
    """The dtype a tensor must travel to the inference engine in.

    TODO: NIXL keeps its own inline copy of this rule; unify the handling.
    """
    return torch.float32 if keep_in_fp32 is not None and keep_in_fp32(key) else default


def resolve_row_chunk(value: Tensor, start: int, stop: int, dtype: torch.dtype) -> Tensor:
    """Gather only this row interval, rather than the complete embedding table."""
    if not isinstance(value, DTensor):
        return value[start:stop].to(dtype)
    mesh = value.device_mesh
    active_dims = [
        i for i, placement in enumerate(value.placements) if mesh.size(i) > 1 and not isinstance(placement, Replicate)
    ]
    if not active_dims:
        return value.to_local()[start:stop].to(dtype)
    if len(active_dims) != 1 or value.placements[active_dims[0]] != Shard(0):
        raise ValueError("Embedding transfer requires replicated weights or a single row-sharding dimension")
    mesh_dim = active_dims[0]
    rows_per_rank = (value.shape[0] + mesh.size(mesh_dim) - 1) // mesh.size(mesh_dim)
    offset = min(rows_per_rank * mesh.get_local_rank(mesh_dim), value.shape[0])
    local = value.to_local()
    chunk = torch.zeros((stop - start, *value.shape[1:]), dtype=dtype, device=value.device)
    left, right = max(start, offset), min(stop, offset + local.shape[0])
    if left < right:
        chunk[left - start : right - start].copy_(local[left - offset : right - offset])
    # Each element has exactly one owner in this group; adding zero is exact.
    dist.all_reduce(chunk, group=mesh.get_group(mesh_dim))
    return chunk


def gather_weights_parallel(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> dict[str, Tensor]:
    """Gather distributed weights cooperatively, each rank keeping a slice on CPU.

    Ngram tables are reconstructed one HF fragment at a time, with fragments spread
    across writers. Other distributed tensors use per-tensor all-gathers, with each
    rank retaining its assigned weights on CPU. The copies are plain blocking
    transfers; no stream-ordering assumptions.

    Parameters the model reports via ``keep_in_fp32_for_weight_transfer`` skip the
    downcast. Those lists exist because the inference engine's kernels require fp32
    for them: vLLM's DeepSeek V4 hyper-connection kernels, for instance, assert
    ``fn``/``base``/``scale`` are fp32, and its loader silently casts a bf16 tensor
    back up rather than failing, so a downcast here shows up as lost mantissa in a
    Sinkhorn normalization rather than as an error.
    """
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
    from prime_rl.trainer.models.qwen3_8_flash_next.converting_qwen3_8_flash_next import NGramEmbeddingWeights

    state_dict = model.state_dict()
    embedding_ops = (
        {op.runtime_weight: op for op in model.conversion_chain(model.config) if isinstance(op, NGramEmbeddingWeights)}
        if isinstance(model, PreTrainedModelPrimeRL)
        else {}
    )
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    world = get_world()
    owners = partition_weights(state_dict, world.world_size, dtype)
    partial: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        for key, value in state_dict.items():
            op = embedding_ops.get(resolve_fqn(model, key)) if embedding_ops else None
            if op is not None:
                shard_rows = (value.shape[0] + op.shard_count - 1) // op.shard_count
                target_dtype = (
                    resolve_wire_dtype(keep_in_fp32, key, dtype) if isinstance(value, DTensor) else value.dtype
                )
                for index in range(op.shard_count):
                    start = min(index * shard_rows, value.shape[0])
                    stop = min(start + shard_rows, value.shape[0])
                    chunk = resolve_row_chunk(value, start, stop, target_dtype)
                    # Spread HF fragments across writers instead of keeping the table on one host.
                    if (owners[key] + index) % world.world_size == world.rank:
                        partial[f"{op.checkpoint_prefix}.shard_{index}.weight"] = chunk.to("cpu")
                    del chunk
                continue
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                target_dtype = resolve_wire_dtype(keep_in_fp32, key, dtype)
                value = cast(DTensor, value.to(target_dtype)).full_tensor()
            if owners[key] != world.rank:
                continue
            partial[resolve_fqn(model, key)] = value.to("cpu")
        dist.barrier()

    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in partial.keys()):
        raise ValueError("gather_weights_parallel does not support LoRA state dicts")

    return partial


def save_state_dict_parallel(state_dict: dict[str, Tensor], save_dir: Path) -> None:
    """Cooperatively save a rank-partitioned state dict as sharded safetensors.

    Each rank splits its slice into <=5GB shards and writes them under temporary
    names, shard maps are exchanged, and every rank renames its own files into the
    global numbering (non-POSIX shared filesystems cannot reliably rename another
    rank's unpublished writes). Master writes the index last, after the barrier, so
    a complete index implies complete shards.
    """
    logger = get_logger()
    world = get_world()

    # Master-only mkdir + barrier: concurrent mkdir from every rank can re-raise
    # FileExistsError on a parallel FS (EEXIST + stale is_dir()).
    if world.is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tmp_pattern = f"tmp-rank{world.rank}{{suffix}}.safetensors"
    split = split_torch_state_dict_into_shards(state_dict, filename_pattern=tmp_pattern)
    local_shards: list[tuple[str, dict[str, int]]] = []
    for tmp_name, tensor_names in split.filename_to_tensors.items():
        if not tensor_names:
            continue
        shard = {name: state_dict[name].contiguous() for name in tensor_names}
        save_file(shard, save_dir / tmp_name, metadata={"format": "pt"})
        local_shards.append((tmp_name, {name: shard[name].nbytes for name in tensor_names}))

    all_shards: list[list[tuple[str, dict[str, int]]] | None] = [None] * world.world_size
    dist.all_gather_object(all_shards, local_shards)
    flat = [(rank, tmp_name, sizes) for rank, shards in enumerate(all_shards) for tmp_name, sizes in shards]

    num_shards = len(flat)
    weight_map: dict[str, str] = {}
    for index, (rank, tmp_name, sizes) in enumerate(flat, start=1):
        if num_shards == 1:
            final_name = SAFE_WEIGHTS_NAME
        else:
            final_name = SAFE_WEIGHTS_NAME.replace(".safetensors", f"-{index:05d}-of-{num_shards:05d}.safetensors")
        for name in sizes:
            weight_map[name] = final_name
        if rank == world.rank:
            (save_dir / tmp_name).rename(save_dir / final_name)
    logger.debug(f"Saved {num_shards} weight shards across {world.world_size} ranks")

    dist.barrier()
    if world.is_master and num_shards > 1:
        total_size = sum(size for _, _, sizes in flat for size in sizes.values())
        index_json = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
            f.write(json.dumps(index_json, indent=2, sort_keys=True) + "\n")
