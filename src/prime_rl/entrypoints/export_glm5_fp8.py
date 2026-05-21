import argparse
import gc
import json
import re
import shutil
import time
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

import prime_rl._compat  # noqa: F401 - patch transitive imports before model setup
from prime_rl.configs.rl import RLConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.model import setup_model
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise
from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.weights import PYTORCH_WRAPPER_PREFIXES
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path, resolve_latest_ckpt_step

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
SAFE_WEIGHTS_NAME = "model.safetensors"
DCP_MODEL_PREFIX = "app.model."


class ModelOnlyAppState(Stateful):
    """Load only `app.model` from a PrimeRL trainer DCP checkpoint."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, _ = get_state_dict(self.model, [])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(self.model, [], model_state_dict=state_dict["model"], optim_state_dict={})


@dataclass
class ExportStats:
    tensors: int = 0
    quantized: int = 0
    bf16: int = 0
    skipped: int = 0


class IncrementalSafeTensorWriter:
    def __init__(self, save_dir: Path, max_shard_bytes: int):
        self.save_dir = save_dir
        self.max_shard_bytes = max_shard_bytes
        self.current: dict[str, Tensor] = {}
        self.current_size = 0
        self.total_size = 0
        self.shards: list[tuple[Path, list[str]]] = []
        self.tensor_to_file: dict[str, str] = {}
        self.next_tmp_idx = 1

    def add(self, name: str, tensor: Tensor) -> None:
        if name in self.tensor_to_file or name in self.current:
            raise ValueError(f"Duplicate tensor in export: {name}")

        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()
        if self.current and self.current_size + tensor_size > self.max_shard_bytes:
            self.flush()

        self.current[name] = tensor
        self.current_size += tensor_size
        self.total_size += tensor_size

    def flush(self) -> None:
        if not self.current:
            return
        tmp_name = f"model-tmp-{self.next_tmp_idx:05d}.safetensors"
        tmp_path = self.save_dir / tmp_name
        keys = list(self.current)
        save_file(self.current, tmp_path, metadata={"format": "pt"})
        self.shards.append((tmp_path, keys))
        for key in keys:
            self.tensor_to_file[key] = tmp_name
        self.current = {}
        self.current_size = 0
        self.next_tmp_idx += 1
        gc.collect()

    def finalize(self) -> None:
        self.flush()
        if not self.shards:
            raise RuntimeError("No tensors were written")

        if len(self.shards) == 1:
            tmp_path, keys = self.shards[0]
            final_name = SAFE_WEIGHTS_NAME
            tmp_path.rename(self.save_dir / final_name)
            self.tensor_to_file = {key: final_name for key in keys}
            return

        total = len(self.shards)
        final_weight_map: dict[str, str] = {}
        for idx, (tmp_path, keys) in enumerate(self.shards, start=1):
            final_name = f"model-{idx:05d}-of-{total:05d}.safetensors"
            tmp_path.rename(self.save_dir / final_name)
            for key in keys:
                final_weight_map[key] = final_name

        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": final_weight_map,
        }
        with open(self.save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")
        self.tensor_to_file = final_weight_map


def parse_size(value: str) -> int:
    text = value.strip().upper().replace(" ", "")
    units = [
        ("GIB", 1024**3),
        ("GB", 1000**3),
        ("MIB", 1024**2),
        ("MB", 1000**2),
        ("KIB", 1024),
        ("KB", 1000),
        ("B", 1),
    ]
    for suffix, multiplier in units:
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def strip_wrapper_prefixes(key: str) -> str:
    for prefix in PYTORCH_WRAPPER_PREFIXES:
        key = key.replace(prefix, "")
    return key


def canonical_fqn(model: torch.nn.Module, key: str) -> str:
    fqns = get_fqns(model, key)
    if len(fqns) != 1:
        raise ValueError(f"Expected one FQN for {key}, got {sorted(fqns)}")
    return strip_wrapper_prefixes(next(iter(fqns)))


def layer_idx(name: str) -> int | None:
    match = LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def sort_key(name: str) -> tuple[int, int, str]:
    idx = layer_idx(name)
    if idx is None:
        return (0, -1, name)
    return (1, idx, name)


def is_floating_dtype(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point or dtype in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }


def materialize_tensor(tensor: Tensor | DTensor, dtype: torch.dtype, is_master: bool) -> Tensor | None:
    with torch.no_grad():
        if isinstance(tensor, DTensor):
            value = tensor.detach()
            if is_floating_dtype(value.dtype):
                value = value.to(dtype)
            full = value.full_tensor()
        else:
            full = tensor.detach()
            if is_floating_dtype(full.dtype):
                full = full.to(dtype)

        if is_master:
            out = full.to("cpu", non_blocking=False).contiguous()
        else:
            out = None

        del full
        torch.cuda.empty_cache()
        return out


def should_skip_tensor(name: str) -> bool:
    return name.endswith(".mlp.tokens_per_expert")


def should_quantize_glm5_hf_tensor(name: str, tensor: Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False

    if name in {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}:
        return False

    bf16_markers = [
        ".input_layernorm.",
        ".post_attention_layernorm.",
        ".self_attn.q_a_layernorm.",
        ".self_attn.kv_a_layernorm.",
        ".self_attn.indexer.k_norm.",
        ".self_attn.indexer.weights_proj.",
        ".mlp.gate.",
    ]
    return not any(marker in name for marker in bf16_markers)


def add_tensor_with_glm5_fp8_policy(
    writer: IncrementalSafeTensorWriter,
    name: str,
    tensor: Tensor,
    stats: ExportStats,
) -> None:
    if should_skip_tensor(name):
        stats.skipped += 1
        return

    if should_quantize_glm5_hf_tensor(name, tensor):
        fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
        writer.add(name, fp8_weight)
        writer.add(name.removesuffix(".weight") + ".weight_scale_inv", scale)
        stats.quantized += 1
    else:
        writer.add(name, tensor)
        stats.bf16 += 1
    stats.tensors += 1


def build_glm5_quantization_config(num_layers: int, first_k_dense_replace: int) -> dict[str, Any]:
    modules_to_not_convert = ["lm_head", "model.embed_tokens", "model.norm"]
    for i in range(num_layers):
        modules_to_not_convert.extend(
            [
                f"model.layers.{i}.input_layernorm",
                f"model.layers.{i}.post_attention_layernorm",
                f"model.layers.{i}.self_attn.indexer.k_norm",
                f"model.layers.{i}.self_attn.indexer.weights_proj",
                f"model.layers.{i}.self_attn.indexers_proj",
                f"model.layers.{i}.self_attn.kv_a_layernorm",
                f"model.layers.{i}.self_attn.q_a_layernorm",
            ]
        )
        if i >= first_k_dense_replace:
            modules_to_not_convert.extend(
                [
                    f"model.layers.{i}.mlp.gate",
                    f"model.layers.{i}.mlp.gate.e_score_correction_bias",
                ]
            )

    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "modules_to_not_convert": modules_to_not_convert,
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }


def save_metadata_files(
    output_dir: Path,
    metadata_model: str,
    tokenizer_name: str | None,
    trust_remote_code: bool | None,
    fallback_model_config: Any,
) -> None:
    logger = get_logger()
    config = AutoConfig.from_pretrained(metadata_model, trust_remote_code=bool(trust_remote_code))
    if getattr(config, "model_type", None) != "glm_moe_dsa":
        raise ValueError(f"Metadata model must be glm_moe_dsa, got model_type={getattr(config, 'model_type', None)}")

    if not getattr(config, "quantization_config", None):
        num_layers = getattr(config, "num_hidden_layers", getattr(fallback_model_config, "num_hidden_layers"))
        first_k_dense_replace = getattr(
            config,
            "first_k_dense_replace",
            getattr(fallback_model_config, "first_k_dense_replace", 0),
        )
        config.quantization_config = build_glm5_quantization_config(num_layers, first_k_dense_replace)

    config.use_cache = True
    config.save_pretrained(output_dir)

    try:
        generation_config = GenerationConfig.from_pretrained(metadata_model, trust_remote_code=bool(trust_remote_code))
        generation_config.use_cache = True
        generation_config.save_pretrained(output_dir)
    except Exception as exc:
        logger.warning(f"Could not save generation_config from {metadata_model}: {exc}")

    tokenizer_source = tokenizer_name or metadata_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=bool(trust_remote_code))
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:
        logger.warning(f"Could not save tokenizer from {tokenizer_source}: {exc}")


def resolve_checkpoint_path(rl_config: RLConfig, checkpoint_step: int | None, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        path = checkpoint_path
        if path.name.startswith("step_"):
            path = path / "trainer"
        elif (path / "trainer").exists():
            path = path / "trainer"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        return path

    if rl_config.trainer.ckpt is None:
        raise ValueError("RL config has no trainer.ckpt; pass --checkpoint-path explicitly")

    ckpt_output_dir = rl_config.trainer.ckpt.output_dir or rl_config.trainer.output_dir
    ckpt_dir = get_ckpt_dir(ckpt_output_dir)
    step = checkpoint_step
    if step is None or step == -1:
        step = resolve_latest_ckpt_step(ckpt_dir)
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    path = get_step_path(ckpt_dir, step) / "trainer"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return path


def is_tensor_metadata(value: Any) -> bool:
    return hasattr(value, "size") and hasattr(value, "properties")


def checkpoint_model_items(checkpoint_path: Path) -> list[tuple[str, str, Any]]:
    metadata = FileSystemReader(checkpoint_path).read_metadata()
    items: list[tuple[str, str, Any]] = []
    for dcp_name, tensor_metadata in metadata.state_dict_metadata.items():
        if not dcp_name.startswith(DCP_MODEL_PREFIX) or not is_tensor_metadata(tensor_metadata):
            continue
        model_name = strip_wrapper_prefixes(dcp_name.removeprefix(DCP_MODEL_PREFIX))
        items.append((model_name, dcp_name, tensor_metadata))
    return sorted(items, key=lambda item: sort_key(item[0]))


def tensor_metadata_shape(tensor_metadata: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor_metadata.size)


def tensor_metadata_dtype(tensor_metadata: Any) -> torch.dtype:
    return tensor_metadata.properties.dtype


def load_checkpoint_tensor(
    checkpoint_path: Path,
    dcp_name: str,
    tensor_metadata: Any,
    floating_dtype: torch.dtype,
) -> Tensor:
    source_dtype = tensor_metadata_dtype(tensor_metadata)
    target_dtype = floating_dtype if is_floating_dtype(source_dtype) else source_dtype
    tensor = torch.empty(tensor_metadata_shape(tensor_metadata), dtype=target_dtype)
    dcp_load({dcp_name: tensor}, checkpoint_id=checkpoint_path, no_dist=True)
    return tensor


def add_loaded_tensor(
    writer: IncrementalSafeTensorWriter,
    checkpoint_path: Path,
    name: str,
    dcp_name: str,
    tensor_metadata: Any,
    stats: ExportStats,
    floating_dtype: torch.dtype,
) -> None:
    if should_skip_tensor(name):
        stats.skipped += 1
        return
    tensor = load_checkpoint_tensor(checkpoint_path, dcp_name, tensor_metadata, floating_dtype)
    add_tensor_with_glm5_fp8_policy(writer, name, tensor, stats)
    del tensor
    gc.collect()


def export_streaming_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    max_shard_bytes: int,
    gather_dtype: torch.dtype,
) -> ExportStats:
    logger = get_logger()
    items = checkpoint_model_items(checkpoint_path)
    if not items:
        raise RuntimeError(f"No model tensors found in checkpoint metadata: {checkpoint_path}")

    non_layer_items: list[tuple[str, str, Any]] = []
    layer_items: dict[int, list[tuple[str, str, Any]]] = {}
    for item in items:
        idx = layer_idx(item[0])
        if idx is None:
            non_layer_items.append(item)
        else:
            layer_items.setdefault(idx, []).append(item)

    writer = IncrementalSafeTensorWriter(output_dir, max_shard_bytes)
    stats = ExportStats()
    logger.info(
        f"Streaming {len(items)} model tensors from {checkpoint_path} "
        f"({len(non_layer_items)} non-layer tensors, {len(layer_items)} layers)"
    )

    for name, dcp_name, tensor_metadata in non_layer_items:
        logger.info(f"Loading {name} shape={tensor_metadata_shape(tensor_metadata)}")
        add_loaded_tensor(writer, checkpoint_path, name, dcp_name, tensor_metadata, stats, gather_dtype)

    for idx in sorted(layer_items):
        layer_state: dict[str, Tensor] = {}
        logger.info(f"Loading layer {idx} ({len(layer_items[idx])} tensors)")
        for name, dcp_name, tensor_metadata in layer_items[idx]:
            if should_skip_tensor(name):
                stats.skipped += 1
                continue
            layer_state[name] = load_checkpoint_tensor(checkpoint_path, dcp_name, tensor_metadata, gather_dtype)

        GlmMoeDsaForCausalLM.convert_layer_to_hf(layer_state, idx)
        for name, tensor in sorted(layer_state.items(), key=lambda item: item[0]):
            add_tensor_with_glm5_fp8_policy(writer, name, tensor, stats)
        layer_state.clear()
        gc.collect()
        logger.info(
            f"Finished layer {idx} (written_tensors={stats.tensors}, quantized={stats.quantized}, "
            f"bf16={stats.bf16}, skipped={stats.skipped})"
        )

    writer.finalize()
    return stats


def iter_state_items(model: torch.nn.Module) -> list[tuple[str, str, Tensor | DTensor]]:
    items = []
    for raw_key, tensor in model.state_dict().items():
        items.append((canonical_fqn(model, raw_key), raw_key, tensor))
    return sorted(items, key=lambda item: sort_key(item[0]))


def flush_layer(
    writer: IncrementalSafeTensorWriter,
    model: torch.nn.Module,
    current_layer_idx: int | None,
    layer_state: dict[str, Tensor],
    stats: ExportStats,
) -> None:
    if current_layer_idx is None:
        return
    if isinstance(model, PreTrainedModelPrimeRL):
        model.convert_layer_to_hf(layer_state, current_layer_idx)
    for name, tensor in sorted(layer_state.items(), key=lambda item: item[0]):
        add_tensor_with_glm5_fp8_policy(writer, name, tensor, stats)
    layer_state.clear()


def export_loaded_model(
    model: torch.nn.Module,
    output_dir: Path,
    max_shard_bytes: int,
    gather_dtype: torch.dtype,
) -> ExportStats:
    world = get_world()
    writer = IncrementalSafeTensorWriter(output_dir, max_shard_bytes) if world.is_master else None
    stats = ExportStats()
    current_layer_idx: int | None = None
    layer_state: dict[str, Tensor] = {}

    state_items = iter_state_items(model)
    logger = get_logger()
    if world.is_master:
        logger.info(f"Exporting {len(state_items)} state tensors")

    for idx, (name, _raw_key, tensor) in enumerate(state_items, start=1):
        idx_for_name = layer_idx(name)
        if world.is_master and idx % 100 == 0:
            logger.info(f"Materialized {idx}/{len(state_items)} tensors")

        materialized = materialize_tensor(tensor, gather_dtype, world.is_master)
        if not world.is_master:
            continue

        assert writer is not None
        assert materialized is not None
        if idx_for_name is None:
            if current_layer_idx is not None:
                flush_layer(writer, model, current_layer_idx, layer_state, stats)
                current_layer_idx = None
            add_tensor_with_glm5_fp8_policy(writer, name, materialized, stats)
        else:
            if current_layer_idx is None:
                current_layer_idx = idx_for_name
            elif current_layer_idx != idx_for_name:
                flush_layer(writer, model, current_layer_idx, layer_state, stats)
                current_layer_idx = idx_for_name
            layer_state[name] = materialized

    if world.is_master:
        assert writer is not None
        flush_layer(writer, model, current_layer_idx, layer_state, stats)
        writer.finalize()

    dist.barrier()
    return stats


def prepare_model_config_for_export(config: ModelConfig, fsdp_cpu_offload: bool | None) -> ModelConfig:
    export_config = deepcopy(config)
    export_config.compile = None
    export_config.ac = None
    export_config.ac_offloading = None
    export_config.fsdp_cpu_offload = True if fsdp_cpu_offload is None else fsdp_cpu_offload
    export_config.optim_cpu_offload = False
    export_config.fp8 = False
    return export_config


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PrimeRL GLM5 trainer checkpoint to HF-style FP8 safetensors")
    parser.add_argument("--rl-config", required=True, type=Path, help="Path to the RL config used for the run")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for the exported FP8 model")
    parser.add_argument("--checkpoint-step", type=int, default=-1, help="Checkpoint step to export; -1 uses latest")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None, help="Explicit trainer checkpoint path or step dir"
    )
    parser.add_argument("--metadata-model", default=None, help="Model to copy config/generation metadata from")
    parser.add_argument(
        "--tokenizer-name", default=None, help="Tokenizer source; defaults to config tokenizer or metadata model"
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum safetensors shard size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output-dir if it already exists")
    parser.add_argument("--log-level", default=None, help="Override log level")
    parser.add_argument(
        "--fsdp-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Materialize FSDP parameters on CPU. Defaults to enabled for single-node exports.",
    )
    parser.add_argument(
        "--distributed-model-load",
        action="store_true",
        help="Use the legacy distributed model materialization path instead of streaming tensors from DCP.",
    )
    return parser.parse_args(args)


def export(args: argparse.Namespace) -> None:
    rl_config = cli(RLConfig, args=["@", str(args.rl_config)])
    world = get_world()
    log_level = args.log_level or rl_config.trainer.log.level or "info"
    setup_logger(log_level, tag=f"export:{world.rank}", json_logging=rl_config.trainer.log.json_logging)
    logger = get_logger()
    model_config = prepare_model_config_for_export(rl_config.trainer.model, args.fsdp_cpu_offload)

    if not args.distributed_model_load:
        output_dir = args.output_dir
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        checkpoint_path = resolve_checkpoint_path(rl_config, args.checkpoint_step, args.checkpoint_path)
        logger.info(f"Streaming trainer checkpoint from {checkpoint_path}")

        metadata_model = args.metadata_model
        if metadata_model is None and rl_config.inference is not None:
            metadata_model = rl_config.inference.model.name
        if metadata_model is None:
            metadata_model = rl_config.trainer.model.name

        tokenizer_name = args.tokenizer_name
        if tokenizer_name is None and rl_config.trainer.tokenizer.name is not None:
            tokenizer_name = rl_config.trainer.tokenizer.name

        fallback_model_config = AutoConfig.from_pretrained(
            rl_config.trainer.model.name,
            trust_remote_code=bool(rl_config.trainer.model.trust_remote_code),
        )
        save_metadata_files(
            output_dir,
            metadata_model,
            tokenizer_name,
            rl_config.trainer.model.trust_remote_code,
            fallback_model_config,
        )

        t0 = time.perf_counter()
        stats = export_streaming_checkpoint(
            checkpoint_path,
            output_dir,
            max_shard_bytes=parse_size(args.max_shard_size),
            gather_dtype=torch.bfloat16,
        )
        (output_dir / "STABLE").touch()
        logger.success(
            "Exported GLM5 FP8 model to "
            f"{output_dir} in {time.perf_counter() - t0:.2f}s "
            f"(tensors={stats.tensors}, quantized={stats.quantized}, "
            f"bf16={stats.bf16}, skipped={stats.skipped})"
        )
        return

    setup_torch_distributed(
        timeout=timedelta(seconds=rl_config.trainer.dist_timeout_seconds),
        enable_gloo=model_config.fsdp_cpu_offload,
    )
    logger.info(
        "Export setup: "
        f"world_size={dist.get_world_size()}, cp={model_config.cp}, ep={model_config.ep}, "
        f"fsdp_cpu_offload={model_config.fsdp_cpu_offload}, fp8_wrappers={model_config.fp8}"
    )

    output_dir = args.output_dir
    if world.is_master:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    parallel_dims = get_parallel_dims(model_config)
    model = setup_model(model_config, parallel_dims, loading_from_checkpoint_later=True)

    if getattr(model.config, "model_type", None) != "glm_moe_dsa":
        raise ValueError(f"export-glm5-fp8 only supports glm_moe_dsa, got {getattr(model.config, 'model_type', None)}")

    checkpoint_path = resolve_checkpoint_path(rl_config, args.checkpoint_step, args.checkpoint_path)
    logger.info(f"Loading trainer checkpoint from {checkpoint_path}")
    t0 = time.perf_counter()
    dcp_load({"app": ModelOnlyAppState(model)}, checkpoint_id=checkpoint_path)
    logger.info(f"Loaded checkpoint in {time.perf_counter() - t0:.2f}s")

    metadata_model = args.metadata_model
    if metadata_model is None and rl_config.inference is not None:
        metadata_model = rl_config.inference.model.name
    if metadata_model is None:
        metadata_model = rl_config.trainer.model.name

    tokenizer_name = args.tokenizer_name
    if tokenizer_name is None and rl_config.trainer.tokenizer.name is not None:
        tokenizer_name = rl_config.trainer.tokenizer.name

    if world.is_master:
        save_metadata_files(
            output_dir,
            metadata_model,
            tokenizer_name,
            rl_config.trainer.model.trust_remote_code,
            model.config,
        )

    stats = export_loaded_model(
        model,
        output_dir,
        max_shard_bytes=parse_size(args.max_shard_size),
        gather_dtype=torch.bfloat16,
    )

    if world.is_master:
        (output_dir / "STABLE").touch()
        logger.success(
            "Exported GLM5 FP8 model to "
            f"{output_dir} (tensors={stats.tensors}, quantized={stats.quantized}, "
            f"bf16={stats.bf16}, skipped={stats.skipped})"
        )


def main() -> None:
    try:
        export(parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
