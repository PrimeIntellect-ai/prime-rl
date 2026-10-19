"""Configuration and tokenizer assets for model-free HF checkpoint export."""

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, PretrainedConfig

logger = logging.getLogger(__name__)


def read_model_config(model_config: dict) -> dict:
    """Read only config.json, never model weights or remote Python code."""
    name = model_config["name"]
    source = Path(name)
    path = source / "config.json" if source.is_dir() else Path(hf_hub_download(name, "config.json"))
    return json.loads(path.read_text())


def prepare_export_config(config: dict, model_config: dict) -> dict:
    """Apply the trainer's relevant config changes without constructing a model."""
    if model_config.get("lora") is not None:
        raise ValueError("LoRA checkpoints are not supported - CPU export supports full fine-tunes only")
    if model_config.get("vlm") is not None or "vision_config" in config or "text_config" in config:
        raise ValueError("CPU export supports text-only models; VLM checkpoints are not supported")
    quantization = config.get("quantization_config")
    if quantization is not None and quantization != {}:
        raise ValueError(
            "CPU export does not support this quantization_config encoding "
            f"({type(quantization).__name__}, including FP8 releases); "
            "use an unquantized base config that matches the checkpoint"
        )
    auto_map = config.get("auto_map") or {}
    if model_config.get("trust_remote_code", False) and any(
        key == "AutoConfig" or key.startswith("AutoModel") for key in auto_map
    ):
        raise ValueError(
            "CPU export cannot reproduce trusted custom model/config code; use a native Transformers config"
        )
    result = deepcopy(config)
    result["dtype"] = "bfloat16"
    result.pop("torch_dtype", None)
    result["use_cache"] = False
    if "dsa_backend" in model_config:
        result["dsa_backend"] = model_config["dsa_backend"]
    indexer_types = result.get("indexer_types")
    if indexer_types and "shared" in indexer_types:
        result["use_index_cache"] = True
    layers = (model_config.get("debug") or {}).get("num_layers")
    if layers is not None:
        result["num_hidden_layers"] = min(layers, result["num_hidden_layers"])
    pad = result.get("pad_token_id")
    if pad is None:
        pad = result.get("eos_token_id")
    if isinstance(pad, list):
        pad = pad[0]
    result["pad_token_id"] = pad
    return result


def save_model_assets(config: dict, model_config: dict, tokenizer_config: dict, output_dir: Path) -> None:
    """Save HF assets without loading or instantiating model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Enable generation caching before GenerationConfig validates cache settings.
    generation_source = deepcopy(config)
    generation_source["use_cache"] = True
    generation_config = GenerationConfig.from_model_config(PretrainedConfig.from_dict(generation_source))
    generation_config.use_cache = True
    generation_config.save_pretrained(output_dir)

    name = model_config["name"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    # AutoProcessor is optional for text-only models and may resolve to a tokenizer.
    try:
        processor = AutoProcessor.from_pretrained(name, trust_remote_code=trust_remote_code)
    except (ValueError, OSError, KeyError) as exc:
        logger.debug("No optional AutoProcessor for %s: %s", name, exc)
        processor = None
    if processor is not None and (
        getattr(processor, "image_processor", None) or getattr(processor, "video_processor", None)
    ):
        processor.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.get("name") or name,
        trust_remote_code=tokenizer_config.get("trust_remote_code", False),
    )
    chat_template = tokenizer_config.get("chat_template")
    if chat_template is not None:
        path = Path(chat_template)
        tokenizer.chat_template = path.read_text() if path.is_file() else chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(output_dir)
    source_dir = Path(name)
    if source_dir.is_dir():
        for path in source_dir.glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)
    # Save the original JSON attributes, rather than a generic config's defaults.
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
