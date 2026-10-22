"""Convert a DCP trainer checkpoint into bf16 HF-format weights offline, on CPU.

The trainer's own ``get_model`` builds the model on the meta device only for its
parameter names and prime->HF conversion rules, so every trainer config policy
(depth trimming, IndexShare, KV precision, pad-token fallback) applies to exports
without a second implementation. Each decoder layer is loaded, converted and
written as its own shard by one of ``--workers`` CPU threads, so host memory
stays near ``workers`` times one layer and no GPU is needed.

Usage:
    uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_{n} [output_dir] [--workers 8]

WORLD_SIZE must be 1. Writes to <step_dir>/weights by default. Existing output files are overwritten.
"""

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
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model, setup_processor, setup_tokenizer
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.weights import convert_state_dict_to_hf, group_by_layer, resolve_wire_dtype

logger = get_logger()
SINGLE_PROCESS_ERROR = (
    "CPU-only DCP export does not support WORLD_SIZE>1. Run a single process and use --workers for parallelism."
)
RUN_CONFIG_NAMES = ("trainer.json", "sft.json")


def _require_single_process() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError(SINGLE_PROCESS_ERROR)


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


def strip_quantization_metadata(config) -> None:
    values = config if isinstance(config, dict) else vars(config)
    values.pop("quantization_config", None)
    for name, value in values.items():
        if name in ("text_config", "vision_config", "audio_config", "encoder", "decoder", "generator"):
            if isinstance(value, dict) or hasattr(value, "to_dict"):
                strip_quantization_metadata(value)
        elif isinstance(value, dict) and "model_type" in value:
            strip_quantization_metadata(value)


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
    """A DCP checkpoint opened for export: its metadata (read once) and the run's model
    on the meta device, which provides the parameter names and prime->HF conversion rules."""

    dcp_dir: Path
    metadata: Metadata
    model: nn.Module
    model_config: ModelConfig
    tokenizer_config: TokenizerConfig

    def stored_dtype(self, key: str) -> torch.dtype:
        return self.metadata.state_dict_metadata[f"app.model.{key}"].properties.dtype


class MetadataReader(FileSystemReader):
    """Serve metadata read once, so loading one layer doesn't re-read the checkpoint's
    ``.metadata``, which grows with the model."""

    def __init__(self, path: Path, metadata: Metadata):
        super().__init__(path)
        self.metadata = metadata

    def read_metadata(self, *args, **kwargs) -> Metadata:
        return self.metadata


def open_checkpoint(ckpt_dir: Path) -> Checkpoint:
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
        keys = [key for key in keys if key != "lm_head.weight"]
    return list(group_by_layer(keys).values())


def load_layer(checkpoint: Checkpoint, keys: list[str]) -> dict[str, Tensor]:
    """Load one decoder layer from the checkpoint on CPU and convert it to HF format."""
    model = checkpoint.model
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    skeleton = model.state_dict()
    state_dict = {}
    for key in keys:
        dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16)
        if key.endswith("mlp.router.selection_bias"):
            dtype = checkpoint.stored_dtype(key)
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
) -> None:
    """Write one safetensors shard per decoder layer, ``workers`` layers at a time, then
    the index last, so a complete index implies complete shards. ``transform`` maps each
    converted layer to the tensors written for it."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_layers(checkpoint.model)
    names = [
        SAFE_WEIGHTS_NAME if len(layers) == 1 else f"model-{index:05d}-of-{len(layers):05d}.safetensors"
        for index in range(1, len(layers) + 1)
    ]

    def write_layer(name: str, keys: list[str]) -> dict[str, int]:
        shard = transform(load_layer(checkpoint, keys))
        save_file(
            {key: tensor.contiguous() for key, tensor in shard.items()}, output_dir / name, metadata={"format": "pt"}
        )
        return {key: tensor.nbytes for key, tensor in shard.items()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        key_to_size_per_shard = list(pool.map(write_layer, names, layers))

    if len(layers) == 1:
        return
    weight_map = {key: name for name, sizes in zip(names, key_to_size_per_shard) for key in sizes}
    total_size = sum(sum(sizes.values()) for sizes in key_to_size_per_shard)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def _export_config(config: PretrainedConfig) -> None:
    """Exported configs describe the bf16 safetensors actually written, not the source."""
    strip_quantization_metadata(config)
    config.dtype = "bfloat16"
    for name in ("text_config", "vision_config", "audio_config", "encoder", "decoder", "generator"):
        if isinstance(getattr(config, name, None), PretrainedConfig):
            _export_config(getattr(config, name))


def save_model_assets(
    model: nn.Module, model_config: ModelConfig, tokenizer_config: TokenizerConfig, output_dir: Path
) -> None:
    """Save model config, generation config, processor and tokenizer next to the weights."""
    _export_config(model.config)
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None):
        # training sets use_cache=False which can conflict with cache_implementation -
        # save with use_cache=True without mutating the model's config
        generation_config = deepcopy(model.generation_config)
        generation_config.use_cache = True
        generation_config.save_pretrained(output_dir)
    processor = setup_processor(model_config)
    if processor is not None:
        processor.save_pretrained(output_dir)
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
    _require_single_process()
    checkpoint = open_checkpoint(ckpt_dir)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint.dcp_dir.parent / "weights"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        logger.info(f"Saving model assets to {staging_dir}")
        save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, staging_dir)
        logger.info(f"Reading model tensors from {checkpoint.dcp_dir} on CPU")
        write_weights(checkpoint, staging_dir, workers)
        publish_export(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
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
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error(SINGLE_PROCESS_ERROR)
    setup_logger("info")
    convert(args.ckpt_dir, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
