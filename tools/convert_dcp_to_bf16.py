"""Export DCP checkpoints as layer-parallel BF16 HF weights on CPU."""

import argparse
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader, Metadata
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model, setup_tokenizer
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversion_ops import ChunkedConvOp
from prime_rl.trainer.vlm import setup_processor
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    get_chunked_conversion_ops,
    group_by_layer,
    resolve_wire_dtype,
)

RUN_CONFIG_NAMES = ("trainer.json", "sft.json", "pretrain.json")


def resolve_dcp_dir(ckpt_dir: Path) -> Path:
    for directory in (ckpt_dir / "trainer", ckpt_dir):
        if (directory / ".metadata").exists():
            return directory
    raise FileNotFoundError(f"No DCP checkpoint found at {ckpt_dir}")


def resolve_run_configs(step_dir: Path) -> tuple[ModelConfig, TokenizerConfig]:
    logger = get_logger()
    config_dir = get_config_dir(step_dir.parent.parent)
    for name in RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            logger.info(f"Reading model config from {path}")
            config = json.loads(path.read_text())
            return ModelConfig(**config["model"]), TokenizerConfig(**config["tokenizer"])
    raise FileNotFoundError(f"No resolved run config ({' or '.join(RUN_CONFIG_NAMES)}) found under {config_dir}")


def check_not_lora(model_config: ModelConfig, metadata: Metadata) -> None:
    if model_config.lora is not None:
        raise ValueError("LoRA checkpoints are not supported - CPU export supports full fine-tunes only")
    lora_keys = [
        key for key in metadata.state_dict_metadata if "lora_A" in key or "lora_B" in key or ".base_layer." in key
    ]
    if lora_keys:
        raise ValueError(f"LoRA weights found in the checkpoint: {lora_keys[:3]}")


@dataclass
class Checkpoint:
    dcp_dir: Path
    metadata: Metadata
    model: nn.Module
    model_config: ModelConfig
    tokenizer_config: TokenizerConfig


class MetadataReader(FileSystemReader):
    """Reuse the checkpoint metadata for every layer."""

    def __init__(self, path: Path, metadata: Metadata):
        super().__init__(path)
        self.metadata = metadata

    def read_metadata(self, *args, **kwargs) -> Metadata:
        return self.metadata


class RowSlicePlanner(DefaultLoadPlanner):
    """Read a row interval directly into a shard-sized CPU tensor."""

    def __init__(self, start: int):
        super().__init__(flatten_state_dict=False)
        self.start = start

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None
        items = []
        for key, tensor in self.state_dict.items():
            chunk = ChunkStorageMetadata(torch.Size([self.start, *([0] * (tensor.ndim - 1))]), tensor.size())
            items.extend(create_read_items_for_chunk_list(key, self.metadata.state_dict_metadata[key], [chunk]))
        return LoadPlan(items)

    def lookup_tensor(self, index: MetadataIndex) -> Tensor:
        return self.state_dict[index.fqn]


def open_checkpoint(ckpt_dir: Path) -> Checkpoint:
    logger = get_logger()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("CPU export requires one process; use --workers instead of torchrun")
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    metadata = FileSystemReader(dcp_dir).read_metadata()
    model_config, tokenizer_config = resolve_run_configs(dcp_dir.parent)
    check_not_lora(model_config, metadata)
    if model_config.attn == "auto":
        # attn=auto resolves on the training GPU; any flash backend builds the same meta model.
        model_config.attn = "flash_attention_2"
    logger.info(f"Building {model_config.name} on the meta device for its conversion rules")
    model = get_model(model_config, device=torch.device("meta"), dtype=torch.bfloat16)
    return Checkpoint(dcp_dir, metadata, model, model_config, tokenizer_config)


def get_layers(model: nn.Module) -> list[list[str]]:
    """Group ordinary weights by decoder layer, excluding tied heads and chunked tables."""
    state = model.state_dict()
    chunked = get_chunked_conversion_ops(model, state)
    keys = [key for key in state if key not in chunked]
    # The trainer ties the head on the top-level config flag, so the export must drop it on the same one.
    if getattr(model.config, "tie_word_embeddings", False):
        tied_keys = set(getattr(model, "_tied_weights_keys", None) or ()) | {"lm_head.weight"}
        keys = [key for key in keys if key not in tied_keys]
    return list(group_by_layer(keys).values())


def export_dtype(checkpoint: Checkpoint, key: str) -> torch.dtype:
    stored_dtype = checkpoint.metadata.state_dict_metadata[f"app.model.{key}"].properties.dtype
    if not stored_dtype.is_floating_point or key.endswith("mlp.router.selection_bias"):
        return stored_dtype
    model = checkpoint.model
    keep_in_fp32 = model.keep_in_fp32_for_weight_transfer if isinstance(model, PreTrainedModelPrimeRL) else None
    return resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)


def load_layer(checkpoint: Checkpoint, keys: list[str]) -> dict[str, Tensor]:
    """Load one decoder layer from the checkpoint on CPU and convert it to HF format."""
    model = checkpoint.model
    skeleton = model.state_dict()
    state_dict = {key: torch.empty(skeleton[key].shape, dtype=export_dtype(checkpoint, key)) for key in keys}
    dcp_load(
        {"app": {"model": state_dict}},
        storage_reader=MetadataReader(checkpoint.dcp_dir, checkpoint.metadata),
        no_dist=True,
    )
    return convert_state_dict_to_hf(model, state_dict)


def stored_row_ranges(checkpoint: Checkpoint, key: str) -> list[tuple[int, int]]:
    """Split a tensor's rows at every stored chunk boundary, so each range reads whole chunks."""
    chunks = checkpoint.metadata.state_dict_metadata[f"app.model.{key}"].chunks
    bounds = sorted({bound for chunk in chunks for bound in (chunk.offsets[0], chunk.offsets[0] + chunk.sizes[0])})
    return list(zip(bounds, bounds[1:]))


def load_rows(checkpoint: Checkpoint, key: str, start: int, stop: int) -> Tensor:
    metadata = checkpoint.metadata.state_dict_metadata[f"app.model.{key}"]
    tensor = torch.empty((stop - start, *metadata.size[1:]), dtype=export_dtype(checkpoint, key))
    dcp_load(
        {f"app.model.{key}": tensor},
        storage_reader=MetadataReader(checkpoint.dcp_dir, checkpoint.metadata),
        planner=RowSlicePlanner(start),
        no_dist=True,
    )
    return tensor


def write_weights(
    checkpoint: Checkpoint,
    output_dir: Path,
    workers: int,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] = lambda shard: shard,
) -> dict[str, str]:
    """Write decoder layers and streamed table shards, then return their completed index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_layers(checkpoint.model)
    skeleton = checkpoint.model.state_dict()
    chunked = get_chunked_conversion_ops(checkpoint.model, skeleton)
    num_shards = len(layers) + sum(op.shard_count for op in chunked.values())

    def write_shard(index: int, shard: dict[str, Tensor]) -> tuple[str, dict[str, int]]:
        name = f"model-{index:05d}-of-{num_shards:05d}.safetensors"
        shard = transform(shard)
        save_file(
            {key: tensor.contiguous() for key, tensor in shard.items()}, output_dir / name, metadata={"format": "pt"}
        )
        return name, {key: tensor.nbytes for key, tensor in shard.items()}

    def write_layer(index: int, keys: list[str]) -> tuple[str, dict[str, int]]:
        return write_shard(index, load_layer(checkpoint, keys))

    def write_table(key: str, op: ChunkedConvOp, index: int) -> list[tuple[str, dict[str, int]]]:
        # DCP reads a stored chunk in full to serve any of its rows, so read each
        # chunk once and slice the HF shards, which arrive in row order, from it.
        row_ranges = stored_row_ranges(checkpoint, key)
        loaded: dict[tuple[int, int], Tensor] = {}

        def resolve_rows(weight: Tensor, start: int, stop: int) -> Tensor:
            parts = []
            for row_range in row_ranges:
                first, last = row_range
                if last <= start:
                    loaded.pop(row_range, None)
                elif first < stop:
                    if row_range not in loaded:
                        loaded[row_range] = load_rows(checkpoint, key, first, last)
                    parts.append(loaded[row_range][max(start, first) - first : min(stop, last) - first])
            if not parts:
                return torch.empty((0, *weight.shape[1:]), dtype=export_dtype(checkpoint, key))
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        shards = []
        for offset, packet in enumerate(op.iter_prime_to_hf(skeleton[key], resolve_rows)):
            shards.append(write_shard(index + offset, packet))
            del packet  # a view of its chunk, which must be free before the next chunk loads
        return shards

    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(write_layer, range(1, len(layers) + 1), layers))
        table_indices = []
        next_index = len(layers) + 1
        for op in chunked.values():
            table_indices.append(next_index)
            next_index += op.shard_count
        for table_shards in pool.map(write_table, chunked, chunked.values(), table_indices):
            shards.extend(table_shards)

    weight_map = {key: name for name, sizes in shards for key in sizes}
    total_size = sum(sum(sizes.values()) for _, sizes in shards)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return weight_map


def _export_config(config: PretrainedConfig) -> None:
    config.dtype = "bfloat16"
    vars(config).pop("quantization_config", None)
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            _export_config(value)


def save_model_assets(
    model: nn.Module, model_config: ModelConfig, tokenizer_config: TokenizerConfig, output_dir: Path
) -> None:
    """Save model config, generation config, processor and tokenizer next to the weights."""
    _export_config(model.config)
    model.config.save_pretrained(output_dir)
    if model.generation_config:
        generation_config = deepcopy(model.generation_config)
        generation_config.use_cache = True
        generation_config.save_pretrained(output_dir)
    # Processor first: it saves its own (unmodified) tokenizer, which the configured
    # tokenizer (pad token, custom chat template) must override.
    processor = setup_processor(model_config)
    if processor is not None:
        processor.save_pretrained(output_dir)
    if tokenizer_config.name is None:
        tokenizer_config = tokenizer_config.model_copy(update={"name": model_config.name})
    setup_tokenizer(tokenizer_config).save_pretrained(output_dir)
    if Path(model_config.name).is_dir():
        for path in Path(model_config.name).glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)


def publish_export(staging_dir: Path, output_dir: Path) -> None:
    """Replace a complete export, restoring the old directory if publication fails or is interrupted."""
    logger = get_logger()
    umask = os.umask(0)
    os.umask(umask)
    # Keep the old export's mode, or the umask default plus any setgid bit inherited from the parent.
    if output_dir.is_dir():
        mode = output_dir.stat().st_mode & 0o7777
    else:
        mode = (0o777 & ~umask) | (staging_dir.stat().st_mode & stat.S_ISGID)
    staging_dir.chmod(mode)
    if not output_dir.exists():
        staging_dir.rename(output_dir)
        return
    backup_parent = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-backup-", dir=output_dir.parent))
    backup_dir = backup_parent / "export"
    try:
        output_dir.rename(backup_dir)
    except BaseException:
        backup_parent.rmdir()
        raise
    try:
        staging_dir.rename(output_dir)
    except BaseException:
        backup_dir.rename(output_dir)
        backup_parent.rmdir()
        raise
    try:
        shutil.rmtree(backup_parent)
    except OSError as error:
        logger.warning(f"Published {output_dir}, but could not remove the previous export at {backup_parent}: {error}")


def resolve_output_dir(checkpoint: Checkpoint, output_dir: Path | None, name: str, overwrite: bool = False) -> Path:
    """Pick the export directory; only the default one may replace existing content without ``overwrite``."""
    default_dir = checkpoint.dcp_dir.parent / name
    output_dir = Path(output_dir) if output_dir is not None else default_dir
    source = checkpoint.dcp_dir.resolve()
    if output_dir.resolve() == source or output_dir.resolve() in source.parents:
        raise ValueError("The export directory must not contain the source checkpoint")
    is_empty_dir = output_dir.is_dir() and not any(output_dir.iterdir())
    if output_dir.exists() and not is_empty_dir and output_dir.resolve() != default_dir.resolve() and not overwrite:
        raise ValueError(f"{output_dir} exists and is not the default export dir; pass --overwrite to replace it")
    if output_dir.is_mount():
        raise ValueError(f"{output_dir} is a mount point and cannot be replaced; export to a directory inside it")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"The export is staged in {output_dir.parent}, which is not writable")
    return output_dir


def convert(ckpt_dir: Path, output_dir: Path | None = None, *, workers: int = 1, overwrite: bool = False) -> Path:
    """Export a full trainer checkpoint with CPU streaming only."""
    logger = get_logger()
    checkpoint = open_checkpoint(ckpt_dir)
    output_dir = resolve_output_dir(checkpoint, output_dir, "weights", overwrite)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, staging_dir)
        logger.info(f"Reading model tensors from {checkpoint.dcp_dir} on CPU")
        write_weights(checkpoint, staging_dir, workers)
        publish_export(staging_dir, output_dir)
    logger.info(f"Done: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <step_dir>/weights")
    parser.add_argument(
        "--workers", type=int, default=1, help="decoder layers or n-gram tables loaded and written at once (default: 1)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing non-default output_dir and everything in it"
    )
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, workers=args.workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
