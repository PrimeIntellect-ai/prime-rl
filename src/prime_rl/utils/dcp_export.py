"""Model-only, bounded CPU reads of PrimeRL DCP checkpoints."""

import itertools
import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime, apply_prime_to_hf
from prime_rl.utils.export_conversion import (
    CUSTOM_MODEL_TYPES,
    conversion_chain,
    keep_in_fp32,
    prime_schema_details,
    resolve_config,
)
from prime_rl.utils.hf_export_schema import ModelSchemaUnavailable, hf_schema_details, matches_hf_schema

_LAYER = re.compile(r"^(.*?\b(?:layers|h|blocks|block)\.\d+)\.")
_PREFIX = "app.model."


def _validate_chunks(name: str, entry: TensorStorageMetadata) -> None:
    chunks = entry.chunks
    for chunk in chunks:
        if len(chunk.offsets) != len(entry.size) or len(chunk.sizes) != len(entry.size):
            raise ValueError(f"Invalid chunk dimensions for {name}")
        if any(o < 0 or s < 0 or o + s > dim for o, s, dim in zip(chunk.offsets, chunk.sizes, entry.size)):
            raise ValueError(f"Out-of-bounds checkpoint chunk for {name}")
    if sum(math.prod(chunk.sizes) for chunk in chunks) != math.prod(entry.size):
        raise ValueError(f"Checkpoint chunks do not cover {name}")
    for left, right in itertools.combinations(chunks, 2):
        if (
            math.prod(left.sizes)
            and math.prod(right.sizes)
            and all(
                max(a, b) < min(a + sa, b + sb)
                for a, sa, b, sb in zip(left.offsets, left.sizes, right.offsets, right.sizes)
            )
        ):
            raise ValueError(f"Overlapping checkpoint chunks for {name}")


def _model_entries(reader: FileSystemReader) -> dict[str, TensorStorageMetadata]:
    metadata = reader.read_metadata()
    entries = {}
    for name, entry in metadata.state_dict_metadata.items():
        if not name.startswith(_PREFIX):
            continue
        logical = name.removeprefix(_PREFIX)
        if any(marker in logical for marker in ("lora_A", "lora_B", ".base_layer.")):
            raise ValueError("LoRA checkpoints are not supported")
        if not isinstance(entry, TensorStorageMetadata):
            raise ValueError(f"Expected a model tensor: {name}")
        if entry.properties.layout != torch.strided:
            raise ValueError(f"Unsupported checkpoint tensor layout: {name}")
        if entry.properties.dtype not in (
            torch.float64,
            torch.float32,
            torch.bfloat16,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ):
            raise ValueError(f"Unsupported checkpoint tensor dtype: {name}: {entry.properties.dtype}")
        _validate_chunks(name, entry)
        entries[name] = entry
    if not entries:
        raise ValueError("Checkpoint has no app.model tensors")
    return entries


def _validate_custom_layout(
    entries: dict[str, TensorStorageMetadata],
    shapes: dict[str, tuple[int, ...]],
    config: dict,
    model_config: dict | None = None,
) -> set[str]:
    """Validate checkpoint keys/shapes/dtypes against the custom model's meta state.

    Tied aliases are the only optional names; returns them.
    """
    schema, aliases, dtypes = prime_schema_details(config, attn_implementation=(model_config or {}).get("attn"))
    missing = schema.keys() - shapes.keys() - aliases.keys()
    unexpected = shapes.keys() - schema.keys()
    mismatched = {name for name in shapes.keys() & schema.keys() if shapes[name] != schema[name]}
    if missing or unexpected or mismatched:
        raise ValueError(
            "Checkpoint does not match the custom PrimeRL model schema: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}"
        )
    for name in shapes.keys() & schema.keys():
        actual = entries[_PREFIX + name].properties.dtype
        if dtypes[name].is_floating_point:
            if not actual.is_floating_point:
                raise ValueError(f"Quantized/nonfloating checkpoint payload for floating model tensor: {name}")
        elif actual != dtypes[name]:
            raise ValueError(f"Checkpoint dtype {actual} does not match model buffer dtype for {name}")
    return set(aliases)


def _export_layout(
    entries: dict[str, TensorStorageMetadata], config: dict, model_config: dict | None
) -> tuple[list, set[str]]:
    shapes = {name.removeprefix(_PREFIX): tuple(entry.size) for name, entry in entries.items()}
    model_type = config.get("model_type")
    # A meta model validates HF-native keys/shapes without allocating weight storage.
    try:
        schema, aliases, expected_dtypes = hf_schema_details(config, model_config)
    except ModelSchemaUnavailable:
        if model_type not in CUSTOM_MODEL_TYPES:
            raise
    except ImportError:
        if model_type not in CUSTOM_MODEL_TYPES or (model_config or {}).get("trust_remote_code", False):
            raise
    else:
        if matches_hf_schema(shapes, schema, aliases):
            for name in shapes:
                actual = entries[_PREFIX + name].properties.dtype
                if expected_dtypes[name].is_floating_point and not actual.is_floating_point:
                    raise ValueError(f"Quantized/nonfloating checkpoint payload for floating model tensor: {name}")
            return [], set(aliases) & shapes.keys()
        if model_type not in CUSTOM_MODEL_TYPES:
            missing = schema.keys() - shapes.keys() - aliases.keys()
            unexpected = shapes.keys() - schema.keys()
            mismatched = {name for name in shapes.keys() & schema.keys() if shapes[name] != schema[name]}
            raise ValueError(
                "Checkpoint does not match HF model schema: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}"
            )
    aliases = _validate_custom_layout(entries, shapes, config, model_config)
    return conversion_chain(resolve_config(config)), aliases


def _identity_canonical_names(entries: dict[str, TensorStorageMetadata], config: dict) -> dict[str, str]:
    """Resolve HF buffer and parameter names through the shared forward conversion."""
    model_type = config.get("model_type", "")
    if model_type not in CUSTOM_MODEL_TYPES:
        return {}
    state = {name.removeprefix(_PREFIX): torch.empty(entry.size, device="meta") for name, entry in entries.items()}
    original_names = {id(tensor): name for name, tensor in state.items()}
    canonical = apply_hf_to_prime(state, conversion_chain(resolve_config(config)))
    return {original_names[id(tensor)]: name for name, tensor in canonical.items() if id(tensor) in original_names}


def iter_cpu_hf_groups(
    dcp_dir: Path, config: dict, *, model_config: dict | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    """Read one structural block at a time, never optimizer state or model weights."""
    reader = FileSystemReader(dcp_dir)
    entries = _model_entries(reader)
    operations, tied_aliases = _export_layout(entries, config, model_config)
    identity_names = _identity_canonical_names(entries, config) if not operations else {}
    grouped: dict[str, list[str]] = {}
    for name in sorted(entries):
        logical = name.removeprefix(_PREFIX)
        if logical in tied_aliases:
            continue
        match = _LAYER.match(logical)
        unit = match.group(1) if match else logical
        grouped.setdefault(unit, []).append(name)
    model_type = config.get("model_type", "")
    for names in grouped.values():
        targets = {
            name: torch.empty(entries[name].size, dtype=entries[name].properties.dtype, device="cpu") for name in names
        }
        dcp_load(
            targets,
            storage_reader=reader,
            planner=DefaultLoadPlanner(
                flatten_state_dict=False, flatten_sharded_tensors=False, allow_partial_load=False
            ),
            no_dist=True,
        )
        group = {name.removeprefix(_PREFIX): tensor for name, tensor in targets.items()}
        del targets
        for name in group:
            if group[name].is_floating_point():
                canonical_name = identity_names.get(name, name)
                preserve = keep_in_fp32(model_type, canonical_name)
                dtype = torch.float32 if preserve else torch.bfloat16
                if canonical_name.endswith(".router.selection_bias"):
                    dtype = group[name].dtype
                group[name] = group[name].to(dtype)
        yield apply_prime_to_hf(group, operations)
        del group


def write_hf_shards(
    groups: Iterable[dict[str, torch.Tensor]], output_dir: Path, max_shard_size: int = 5_000_000_000
) -> None:
    """Write one conversion group and one output shard at a time."""
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = {}
    shard_bytes = 0
    weight_map = {}
    total_size = 0
    shard_number = 0

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_number
        if not shard:
            return
        shard_number += 1
        filename = f"model-{shard_number:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, filename))
        shard = {}
        shard_bytes = 0

    for group in groups:
        for name in sorted(group):
            if name in weight_map or name in shard:
                raise ValueError(f"Duplicate output tensor: {name}")
            tensor = group.pop(name)
            if tensor.device.type != "cpu":
                raise ValueError(f"Expected CPU output tensor: {name}")
            size = tensor.numel() * tensor.element_size()
            if shard and shard_bytes + size > max_shard_size:
                flush()
            # Own storage: small views must not retain earlier expert stacks.
            shard[name] = tensor.clone(memory_format=torch.contiguous_format)
            shard_bytes += size
            total_size += size
            if shard_bytes >= max_shard_size:
                flush()
            del tensor
        flush()
    if not weight_map:
        raise ValueError("No HF tensors to write")
    filenames = {}
    for number in range(1, shard_number + 1):
        original = f"model-{number:05d}.safetensors"
        final = "model.safetensors" if shard_number == 1 else f"model-{number:05d}-of-{shard_number:05d}.safetensors"
        (output_dir / original).rename(output_dir / final)
        filenames[original] = final
    if shard_number == 1:
        return
    weight_map = {name: filenames[filename] for name, filename in weight_map.items()}
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    path = output_dir / "model.safetensors.index.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
