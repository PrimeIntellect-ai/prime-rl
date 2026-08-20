import json
import re
import warnings
from pathlib import Path
from typing import Literal, cast

import torch
import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from prime_rl.trainer.lora import (
    clean_lora_state_dict,
)
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
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
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
    if adapter:
        weights_name = ADAPTER_SAFE_WEIGHTS_NAME if save_format == "safetensors" else ADAPTER_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
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
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            save_index_file = save_dir / save_index_file
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


def convert_state_dict_to_hf(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a (possibly rank-partial) training-format state dict to HF hub format.

    Format detection uses the model's full key set, so a partial dict (one rank's
    slice) converts the same way as the full state dict would.
    """
    full_keys = dict.fromkeys(resolve_fqn(model, key) for key in model.state_dict().keys())
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(full_keys):
        return model.convert_to_hf(state_dict)
    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, state_dict)


def resolve_fqn(model: nn.Module, key: str) -> str:
    fqns = get_fqns(model, key)
    assert len(fqns) == 1
    return next(iter(fqns))


_LAYER_RE = re.compile(r"^(.*?\blayers\.\d+)\.")

# Pinned staging buffers for gather_weights_parallel, keyed by FQN. They persist
# across saves so the D2H copies never re-pin memory (pinning is slow); each rank
# holds roughly model_size / world_size bytes.
_staging_buffers: dict[str, Tensor] = {}


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


def gather_weights_parallel(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> dict[str, Tensor]:
    """Gather distributed weights cooperatively, each rank keeping a slice on CPU.

    Every rank participates in the per-tensor all-gathers (a ``full_tensor`` call is
    collective and materializes the full tensor on all ranks anyway), but instead of
    only the master keeping everything, each rank copies its owned slice into pinned
    CPU buffers with non-blocking copies that overlap subsequent gathers.
    """
    world = get_world()
    owners = partition_weights(model.state_dict(), world.world_size, dtype)
    partial: dict[str, Tensor] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()
            if owners[key] != world.rank:
                continue
            fqn = resolve_fqn(model, key)
            if value.is_cuda:
                buffer = _staging_buffers.get(fqn)
                if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                    buffer = torch.empty_like(value, device="cpu", pin_memory=True)
                    _staging_buffers[fqn] = buffer
                buffer.copy_(value, non_blocking=True)
                partial[fqn] = buffer
            else:
                partial[fqn] = value
        torch.cuda.synchronize()
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


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Gather distributed weights on CPU on master rank."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)
        torch.distributed.barrier()

    # Always clean up the state dict for HF compatibility
    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
        cpu_state = clean_lora_state_dict(cpu_state)

    return cpu_state
