"""Checkpoint and resolved-config discovery shared by export commands."""

import json
from pathlib import Path

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_config_dir

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
            get_logger().info(f"Reading model config from {path}")
            config = json.loads(path.read_text())
            return ModelConfig(**config["model"]), TokenizerConfig(**config["tokenizer"])
    raise FileNotFoundError(f"No resolved run config ({' or '.join(RUN_CONFIG_NAMES)}) found under {config_dir}")


def uses_tied_embeddings(config) -> bool:
    """Whether the text model ties its LM head to its input embedding (nested text_config wins)."""

    def field(source, key):
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    text = field(config, "text_config")
    return bool(field(text if text is not None else config, "tie_word_embeddings"))


def strip_quantization_metadata(config) -> None:
    values = config if isinstance(config, dict) else vars(config)
    values.pop("quantization_config", None)
    for name, value in values.items():
        if name in ("text_config", "vision_config", "audio_config", "encoder", "decoder", "generator"):
            if isinstance(value, dict) or hasattr(value, "to_dict"):
                strip_quantization_metadata(value)
        elif isinstance(value, dict) and "model_type" in value:
            strip_quantization_metadata(value)
