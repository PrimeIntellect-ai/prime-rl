"""Export DCP checkpoints as layer-parallel BF16 HF weights on CPU."""

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint import FileSystemReader, Metadata
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model, setup_tokenizer
from prime_rl.trainer.vlm import setup_processor
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.weights import convert_state_dict_to_hf, group_by_layer, resolve_wire_dtype

logger = get_logger()
RUN_CONFIG_NAMES = ("trainer.json", "sft.json")


def resolve_dcp_dir(ckpt_dir: Path) -> Path:
    for directory in (ckpt_dir / "trainer", ckpt_dir):
        if (directory / ".metadata").exists():
            return directory
    raise FileNotFoundError(f"No DCP checkpoint found at {ckpt_dir}")


def resolve_run_configs(step_dir: Path) -> tuple[ModelConfig, TokenizerConfig]:
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


def open_checkpoint(ckpt_dir: Path) -> Checkpoint:
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
    """The model's state-dict keys grouped by decoder layer, without the tied LM head."""
    text_config = getattr(model.config, "text_config", model.config)
    keys = list(model.state_dict())
    if getattr(text_config, "tie_word_embeddings", False):
        tied_keys = set(getattr(model, "_tied_weights_keys", None) or ()) | {"lm_head.weight"}
        keys = [key for key in keys if key not in tied_keys]
    return list(group_by_layer(keys).values())


def load_layer(checkpoint: Checkpoint, keys: list[str]) -> dict[str, Tensor]:
    """Load one decoder layer from the checkpoint on CPU and convert it to HF format."""
    model = checkpoint.model
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    skeleton = model.state_dict()
    state_dict = {}
    for key in keys:
        stored_dtype = checkpoint.metadata.state_dict_metadata[f"app.model.{key}"].properties.dtype
        dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)
        if not stored_dtype.is_floating_point or key.endswith("mlp.router.selection_bias"):
            dtype = stored_dtype
        state_dict[key] = torch.empty(skeleton[key].shape, dtype=dtype)
    dcp_load(
        {"app": {"model": state_dict}},
        storage_reader=MetadataReader(checkpoint.dcp_dir, checkpoint.metadata),
        no_dist=True,
    )
    return convert_state_dict_to_hf(model, state_dict)


def write_weights(
    checkpoint: Checkpoint,
    output_dir: Path,
    workers: int,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] = lambda shard: shard,
) -> dict[str, str]:
    """Write layer shards in parallel, then return their completed index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_layers(checkpoint.model)
    names = [f"model-{index:05d}-of-{len(layers):05d}.safetensors" for index in range(1, len(layers) + 1)]

    def write_layer(name: str, keys: list[str]) -> dict[str, int]:
        shard = transform(load_layer(checkpoint, keys))
        save_file(
            {key: tensor.contiguous() for key, tensor in shard.items()}, output_dir / name, metadata={"format": "pt"}
        )
        return {key: tensor.nbytes for key, tensor in shard.items()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        key_to_size_per_shard = list(pool.map(write_layer, names, layers))

    weight_map = {key: name for name, sizes in zip(names, key_to_size_per_shard) for key in sizes}
    total_size = sum(sum(sizes.values()) for sizes in key_to_size_per_shard)
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
    """Publish atomically to a new directory, or overwrite matching files in an existing one."""
    if not output_dir.exists():
        staging_dir.rename(output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_names = {path.name for path in staging_dir.iterdir()}
    for path in [output_dir / "model.safetensors", *output_dir.glob("model-*.safetensors")]:
        if path.name not in staged_names:
            path.unlink(missing_ok=True)
    if "model.safetensors.index.json" not in staged_names:
        (output_dir / "model.safetensors.index.json").unlink(missing_ok=True)
    for path in staging_dir.iterdir():
        path.replace(output_dir / path.name)
    staging_dir.rmdir()


def convert(ckpt_dir: Path, output_dir: Path | None = None, *, workers: int = 1) -> Path:
    """Export a full trainer checkpoint with CPU streaming only."""
    checkpoint = open_checkpoint(ckpt_dir)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint.dcp_dir.parent / "weights"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--workers", type=int, default=1, help="decoder layers loaded and written at once (default: 1)")
    args = parser.parse_args()
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
