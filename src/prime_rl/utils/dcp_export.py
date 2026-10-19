"""Model-only, bounded CPU reads of PrimeRL DCP checkpoints."""

import itertools
import json
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

from prime_rl.utils.conversion_ops import apply_prime_to_hf
from prime_rl.utils.glm_weight_conversion import conversion_chain, keep_in_fp32

CPU_MODEL_TYPES = {"glm4_moe", "glm_moe_dsa", "llama", "qwen3"}
_LAYER = re.compile(r"^(.*?\blayers\.\d+)\.")
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
            torch.float32,
            torch.bfloat16,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.bool,
        ):
            raise ValueError(f"Unsupported checkpoint tensor dtype: {name}: {entry.properties.dtype}")
        if ".qkv_proj." in logical or logical.endswith(".experts.gate_up_proj"):
            raise ValueError(f"Expected canonical unfused checkpoint keys, found {logical}")
        _validate_chunks(name, entry)
        entries[name] = entry
    if not entries:
        raise ValueError("Checkpoint has no app.model tensors")
    return entries


def validate_model_layout(entries: dict[str, TensorStorageMetadata], config: dict) -> None:
    """Require canonical core weights without constructing a training model."""
    model_type = config["model_type"]
    is_glm = model_type in {"glm4_moe", "glm_moe_dsa"}
    if is_glm:
        missing_config = {"first_k_dense_replace", "n_shared_experts", "n_routed_experts"} - config.keys()
        if missing_config:
            raise ValueError(
                f"Missing GLM layout config fields: {sorted(missing_config)}; "
                "provide an unquantized base config.json with explicit GLM layout fields"
            )
    names = {name.removeprefix(_PREFIX) for name in entries}
    layer_ids = {int(match.group(1).rsplit(".", 1)[1]) for name in names if (match := _LAYER.match(name))}
    if layer_ids != set(range(config["num_hidden_layers"])):
        raise ValueError("Checkpoint decoder layers do not match configured depth")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.get("tie_word_embeddings", False):
        required.add("lm_head.weight")
    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}"
        required.update(f"{prefix}.{norm}.weight" for norm in ("input_layernorm", "post_attention_layernorm"))
        attention = ["o_proj"]
        if model_type == "glm_moe_dsa":
            attention.extend(
                ["q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj"]
            )
        else:
            attention.extend(["q_proj", "k_proj", "v_proj"])
            if model_type == "qwen3" or (model_type == "glm4_moe" and config.get("use_qk_norm", False)):
                attention.extend(["q_norm", "k_norm"])
        required.update(f"{prefix}.self_attn.{projection}.weight" for projection in attention)
        if is_glm and i >= config["first_k_dense_replace"]:
            required.update(
                f"{prefix}.mlp.experts.{projection}" for projection in ("gate_proj", "up_proj", "down_proj")
            )
            required.update([f"{prefix}.mlp.router.gate.weight", f"{prefix}.mlp.router.selection_bias"])
            if config["n_shared_experts"] > 0:
                required.update(
                    f"{prefix}.mlp.shared_expert.{projection}.weight"
                    for projection in ("gate_proj", "up_proj", "down_proj")
                )
        else:
            required.update(f"{prefix}.mlp.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj"))
    if missing := required - names:
        raise ValueError(f"Missing model tensors: {sorted(missing)}")


def _validate_glm_group(group: dict[str, torch.Tensor], config: dict) -> None:
    for name, tensor in group.items():
        if name.endswith(".experts.gate_proj"):
            prefix = name.removesuffix("gate_proj")
            required = [prefix + projection for projection in ("gate_proj", "up_proj", "down_proj")]
            if any(key not in group for key in required):
                raise ValueError(f"Incomplete expert projections at {prefix}")
            if any(group[key].ndim != 3 for key in required):
                raise ValueError(f"Expected stacked three-dimensional experts at {prefix}")
            experts = tensor.shape[0]
            if any(group[key].shape[0] != experts for key in required):
                raise ValueError(f"Inconsistent expert count at {prefix}")
            if experts != config.get("n_routed_experts", experts):
                raise ValueError(f"Expert count differs from config at {prefix}")
            gate, up, down = (group[key] for key in required)
            if gate.shape != up.shape or gate.shape[1:] != down.shape[1:][::-1]:
                raise ValueError(f"Inconsistent expert projection shapes at {prefix}")
        elif name.endswith((".experts.up_proj", ".experts.down_proj")):
            if name.rsplit(".", 1)[0] + ".gate_proj" not in group:
                raise ValueError(f"Incomplete expert projections at {name}")


def iter_cpu_hf_groups(dcp_dir: Path, config: dict) -> Iterator[dict[str, torch.Tensor]]:
    """Read full decoder layers individually, never optimizer state or a full model."""
    model_type = config.get("model_type")
    if model_type not in CPU_MODEL_TYPES:
        raise ValueError(f"CPU export does not support model_type={model_type!r}")
    if "vision_config" in config or "text_config" in config:
        raise ValueError("CPU export currently supports text-only models")
    reader = FileSystemReader(dcp_dir)
    entries = _model_entries(reader)
    validate_model_layout(entries, config)
    grouped: dict[str, list[str]] = {}
    for name in sorted(entries):
        logical = name.removeprefix(_PREFIX)
        match = _LAYER.match(logical)
        unit = match.group(1) if match else logical
        if match and int(unit.rsplit(".", 1)[1]) >= config["num_hidden_layers"]:
            raise ValueError(f"Checkpoint layer exceeds configured depth: {logical}")
        grouped.setdefault(unit, []).append(name)
    operations = conversion_chain(SimpleNamespace(**config)) if model_type.startswith("glm") else []
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
        if model_type.startswith("glm"):
            _validate_glm_group(group, config)
        for name in list(group):
            if config.get("tie_word_embeddings", False) and name == "lm_head.weight":
                del group[name]
            elif group[name].is_floating_point():
                dtype = torch.float32 if model_type.startswith("glm") and keep_in_fp32(name) else torch.bfloat16
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
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    path = output_dir / "model.safetensors.index.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
