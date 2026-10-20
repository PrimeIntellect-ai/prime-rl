"""Convert a DCP trainer checkpoint into bf16 HF-format weights offline, on CPU.

The trainer saves only DCP checkpoints; every HF export goes through this
script. The model is built on the meta device only for its parameter names and
prime->HF conversion rules. Each decoder layer is then loaded, converted and
written as its own shard by one of ``--workers`` threads, so host memory stays
near ``workers`` times the largest layer and no GPU or process group is needed.
The checkpoint metadata is read once and shared by every layer.

The model and tokenizer configs are read from the run's resolved config
(``<run>/configs/latest/resolved/trainer.json`` or ``sft.json``). LoRA checkpoints are not
supported — the script exports full fine-tunes only.

Usage (from the prime-rl repo):
    uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_{n} [output_dir] [--workers 8]

Writes to ``<ckpt_dir>/weights`` by default.
"""

import argparse
import json
import shutil
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
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.trainer.model import get_model, setup_processor, setup_tokenizer
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_config_dir
from prime_rl.utils.weights import convert_state_dict_to_hf, group_by_layer, resolve_wire_dtype

# The meta skeleton only provides names and conversion rules; any attention kernel builds it.
CONVERSION_OVERRIDES = {"attn": "flash_attention_2"}

RUN_CONFIG_NAMES = ("trainer.json", "sft.json")


def resolve_dcp_dir(ckpt_dir: Path) -> Path:
    """The DCP checkpoint id for a step dir (``step_{n}``) or its ``trainer`` subdir."""
    if (ckpt_dir / "trainer" / ".metadata").exists():
        return ckpt_dir / "trainer"
    if (ckpt_dir / ".metadata").exists():
        return ckpt_dir
    raise FileNotFoundError(f"No DCP checkpoint found at {ckpt_dir} (expected {ckpt_dir}/trainer/.metadata)")


def resolve_run_configs(step_dir: Path) -> tuple[ModelConfig, TokenizerConfig]:
    """Model/tokenizer configs from the run's resolved config."""
    logger = get_logger()
    config_dir = get_config_dir(step_dir.parent.parent)
    for name in RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            logger.info(f"Reading model config from {path}")
            run_config = json.loads(path.read_text())
            break
    else:
        raise FileNotFoundError(
            f"No resolved run config ({' or '.join(RUN_CONFIG_NAMES)}) found under {config_dir} "
            "- the checkpoint must live in its run directory"
        )

    model = ModelConfig(**run_config["model"])
    tokenizer = TokenizerConfig(**run_config["tokenizer"])
    return model.model_copy(update=CONVERSION_OVERRIDES), tokenizer


def check_not_lora(model_config: ModelConfig, metadata: Metadata) -> None:
    if model_config.lora is not None:
        raise ValueError("LoRA checkpoints are not supported - dcp_to_bf16 exports full fine-tunes only")
    lora_keys = [k for k in metadata.state_dict_metadata if "lora_A" in k or "lora_B" in k or ".base_layer." in k]
    if lora_keys:
        raise ValueError(
            f"Checkpoint contains LoRA keys (e.g. {lora_keys[0]}) - dcp_to_bf16 exports full fine-tunes only"
        )


def save_model_assets(model, model_config: ModelConfig, tokenizer_config: TokenizerConfig, output_dir: Path) -> None:
    """Save model config, generation config, processor and tokenizer next to the weights."""
    model.config.save_pretrained(output_dir)
    if getattr(model, "generation_config", None):
        # training sets use_cache=False which can conflict with cache_implementation —
        # save with use_cache=True without mutating the model's config
        gen_config = deepcopy(model.generation_config)
        gen_config.use_cache = True
        gen_config.save_pretrained(output_dir)
    # Processor first: it saves its own (unmodified) tokenizer, which the configured
    # tokenizer (pad token, custom chat template) must override.
    processor = setup_processor(model_config)
    if processor is not None:
        processor.save_pretrained(output_dir)
    if tokenizer_config.name is None:
        tokenizer_config = tokenizer_config.model_copy(update={"name": model_config.name})
    setup_tokenizer(tokenizer_config).save_pretrained(output_dir)
    # Local trust_remote_code models localize auto_map to plain module names, so the
    # custom code files must travel with the weights.
    source_dir = Path(model_config.name)
    if source_dir.is_dir():
        for path in source_dir.glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)


@dataclass
class Checkpoint:
    """A DCP checkpoint opened for export: its metadata (read once) and the run's model
    on the meta device, which provides the parameter names and prime->HF conversion rules."""

    dcp_dir: Path
    metadata: Metadata
    model: nn.Module
    model_config: ModelConfig
    tokenizer_config: TokenizerConfig


class MetadataReader(FileSystemReader):
    """Serves metadata read once, so loading one layer doesn't re-read the checkpoint's
    ``.metadata``, which grows with the model."""

    def __init__(self, path: Path, metadata: Metadata):
        super().__init__(path)
        self.metadata = metadata

    def read_metadata(self, *args, **kwargs) -> Metadata:
        return self.metadata


def convert(ckpt_dir: Path, output_dir: Path | None = None, workers: int = 1) -> Path:
    logger = setup_logger("info")
    checkpoint = open_checkpoint(ckpt_dir)
    output_dir = output_dir if output_dir is not None else checkpoint.dcp_dir.parent / "weights"

    logger.info(f"Converting {checkpoint.dcp_dir} to HF weights in {output_dir} with {workers} workers")
    write_weights(checkpoint, output_dir, workers)
    save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, output_dir)
    logger.info(f"Done: {output_dir}")
    return output_dir


def get_layers(model: nn.Module) -> list[list[str]]:
    """The model's state-dict keys grouped by decoder layer, without tied duplicates."""
    keys = list(model.state_dict())
    if getattr(model.config, "tie_word_embeddings", False):
        tied_keys = set(getattr(model, "_tied_weights_keys", []))
        keys = [key for key in keys if key not in tied_keys]
    return list(group_by_layer(keys).values())


def load_hf_layer(checkpoint: Checkpoint, keys: list[str]) -> dict[str, Tensor]:
    """Load one layer's weights from the checkpoint on CPU and convert them to HF format."""
    model = checkpoint.model
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    skeleton = model.state_dict()
    state_dict = {
        key: torch.empty(skeleton[key].shape, dtype=resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16))
        for key in keys
    }
    dcp_load(
        {"app": {"model": state_dict}},
        storage_reader=MetadataReader(checkpoint.dcp_dir, checkpoint.metadata),
        no_dist=True,
    )
    return convert_state_dict_to_hf(model, state_dict)


def open_checkpoint(ckpt_dir: Path) -> Checkpoint:
    dcp_dir = resolve_dcp_dir(ckpt_dir)
    metadata = FileSystemReader(dcp_dir).read_metadata()
    model_config, tokenizer_config = resolve_run_configs(dcp_dir.parent)
    check_not_lora(model_config, metadata)
    model = get_model(model_config, device=torch.device("meta"), dtype=torch.bfloat16)
    return Checkpoint(dcp_dir, metadata, model, model_config, tokenizer_config)


def write_weights(
    checkpoint: Checkpoint,
    output_dir: Path,
    workers: int,
    transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] = lambda shard: shard,
) -> None:
    """Write one safetensors shard per layer, ``workers`` layers at a time, then the
    index last, so a complete index implies complete shards. ``transform`` maps each
    converted layer to the tensors written for it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = get_layers(checkpoint.model)
    names = [
        SAFE_WEIGHTS_NAME.replace(".safetensors", f"-{index:05d}-of-{len(layers):05d}.safetensors")
        for index in range(1, len(layers) + 1)
    ]

    def write_layer(name: str, keys: list[str]) -> dict[str, int]:
        shard = transform(load_hf_layer(checkpoint, keys))
        save_file(
            {key: tensor.contiguous() for key, tensor in shard.items()}, output_dir / name, metadata={"format": "pt"}
        )
        return {key: tensor.nbytes for key, tensor in shard.items()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        key_to_size_per_shard = list(pool.map(write_layer, names, layers))

    weight_map = {key: name for name, key_to_size in zip(names, key_to_size_per_shard) for key in key_to_size}
    total_size = sum(sum(key_to_size.values()) for key_to_size in key_to_size_per_shard)
    index_json = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index_json, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <ckpt_dir>/weights")
    parser.add_argument("--workers", type=int, default=1, help="layers converted at once")
    args = parser.parse_args()
    convert(args.ckpt_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
