"""Configuration and tokenizer assets for model-free HF checkpoint export."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import GenerationConfig, PretrainedConfig

from prime_rl.utils.export_config import strip_quantization_metadata
from prime_rl.utils.export_conversion import resolve_config
from prime_rl.utils.logger import get_logger
from prime_rl.utils.model_config import (
    is_multimodal_processor,
    load_tokenizer,
    truncate_model_depth,
    try_load_processor,
)

logger = get_logger()


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
    result = deepcopy(config)
    strip_quantization_metadata(result)

    def normalize(subconfig: dict) -> None:
        subconfig["dtype"] = "bfloat16"
        subconfig.pop("torch_dtype", None)
        subconfig["use_cache"] = False
        for name, value in subconfig.items():
            if isinstance(value, dict) and (
                "model_type" in value
                or name in ("text_config", "vision_config", "audio_config", "encoder", "decoder", "generator")
            ):
                normalize(value)

    normalize(result)
    resolved = resolve_config(result)
    resolved_text = getattr(resolved, "text_config", resolved)
    layers = (model_config.get("debug") or {}).get("num_layers")
    # The trainer auto-enables IndexShare from the pre-trim indexer schedule.
    pre_trim_indexer_types = getattr(resolved_text, "indexer_types", None)
    if layers is not None and hasattr(resolved_text, "num_hidden_layers"):
        # Trim the per-layer schedules with the depth on the resolved config, before any
        # strict HF rebuild, so the shortened config keeps the trainer's first-N-layer layout.
        truncate_model_depth(resolved_text, layers)
    if hasattr(resolved, "to_dict"):
        result = resolved.to_dict()
        normalize(result)
    elif layers is not None and hasattr(resolved_text, "num_hidden_layers"):
        # Custom and unknown model types stay opaque: trim their dict schedules too.
        truncate_model_depth(result.get("text_config", result), layers)
    text_config = result.get("text_config", result)
    if hasattr(resolved_text, "num_hidden_layers"):
        text_config["num_hidden_layers"] = resolved_text.num_hidden_layers
    if "dsa_backend" in model_config:
        result["dsa_backend"] = model_config["dsa_backend"]
    if result.get("model_type") == "deepseek_v4" and "dsv4_kv_precision" in model_config:
        result["kv_precision"] = model_config["dsv4_kv_precision"]
    indexer_types = pre_trim_indexer_types if layers is not None else text_config.get("indexer_types")
    if indexer_types and "shared" in indexer_types:
        text_config["use_index_cache"] = True
    pad = result.get("pad_token_id")
    if pad is None:
        pad = result.get("eos_token_id")
    if pad is None:
        pad = text_config.get("pad_token_id")
    if pad is None:
        pad = text_config.get("eos_token_id")
    if isinstance(pad, list):
        pad = pad[0]
    result["pad_token_id"] = pad
    return result


def _save_trusted_code(config: dict, model_config: dict, output_dir: Path) -> None:
    """Ship explicitly trusted auto-map modules and their relative imports."""
    from transformers.dynamic_module_utils import get_cached_module_file, get_relative_import_files
    from transformers.utils.hub import HF_MODULES_CACHE

    def localize(reference):
        if isinstance(reference, (tuple, list)):
            return [localize(item) for item in reference]
        if reference is None:
            return None
        if "--" in reference:
            source, reference = reference.split("--", 1)
        else:
            source = model_config["name"]
        module, _ = reference.rsplit(".", 1)
        revision = model_config.get("code_revision")
        if revision is None and source == model_config["name"]:
            revision = model_config.get("revision")
        cached = get_cached_module_file(source, module + ".py", revision=revision)
        module_path = Path(HF_MODULES_CACHE) / cached
        for path in [module_path, *map(Path, get_relative_import_files(module_path))]:
            destination = output_dir / path.name
            if destination.exists() and destination.read_bytes() != path.read_bytes():
                raise ValueError(f"Conflicting trusted custom-code modules named {path.name}")
            shutil.copyfile(path, destination)
        return reference

    auto_map = config.get("auto_map")
    if auto_map:
        config["auto_map"] = {name: localize(reference) for name, reference in auto_map.items()}
    for name in ("text_config", "vision_config", "audio_config", "encoder", "decoder", "generator"):
        if isinstance(config.get(name), dict):
            _save_trusted_code(config[name], model_config, output_dir)


def publish_export(staging_dir: Path, output_dir: Path) -> None:
    """Publish atomically to a new directory, or overwrite matching files in an existing one."""
    if not output_dir.exists():
        staging_dir.rename(output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if (staging_dir / "model.safetensors").exists():
        (output_dir / "model.safetensors.index.json").unlink(missing_ok=True)
    else:
        (output_dir / "model.safetensors").unlink(missing_ok=True)
    for path in staging_dir.iterdir():
        path.replace(output_dir / path.name)
    staging_dir.rmdir()


def save_model_assets(config: dict, model_config: dict, tokenizer_config: dict, output_dir: Path) -> None:
    """Save HF assets without loading or instantiating model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Enable generation caching before GenerationConfig validates cache settings.
    generation_source = deepcopy(config)
    generation_source["use_cache"] = True
    generation_model_config = PretrainedConfig.from_dict(generation_source)
    for name in ("text_config", "decoder", "generator"):
        if isinstance(generation_source.get(name), dict):
            nested = deepcopy(generation_source[name])
            nested["use_cache"] = True
            setattr(generation_model_config, name, PretrainedConfig.from_dict(nested))
    generation_config = GenerationConfig.from_model_config(generation_model_config)
    generation_config.use_cache = True
    generation_config.save_pretrained(output_dir)

    name = model_config["name"]
    trust_remote_code = model_config.get("trust_remote_code", False)
    # AutoProcessor is optional for text-only models and may resolve to a tokenizer.
    processor = try_load_processor(name, trust_remote_code=trust_remote_code)
    if is_multimodal_processor(processor):
        processor.save_pretrained(output_dir)

    load_tokenizer(
        tokenizer_config.get("name") or name,
        trust_remote_code=tokenizer_config.get("trust_remote_code", False),
        chat_template=tokenizer_config.get("chat_template"),
    ).save_pretrained(output_dir)
    source_dir = Path(name)
    if source_dir.is_dir():
        for path in source_dir.glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)
    # Save the prepared config without constructing a model.
    exported_config = deepcopy(config)
    if trust_remote_code:
        _save_trusted_code(exported_config, model_config, output_dir)
    native_config = resolve_config(exported_config)
    if isinstance(native_config, PretrainedConfig):
        native_config.use_cache = False
        native_config.to_json_file(output_dir / "config.json", use_diff=True)
    else:
        # Unknown trusted model types have no config class: generic PretrainedConfig diff
        # serialization drops their explicit model_type, so write the prepared dict as-is.
        exported_config["use_cache"] = False
        (output_dir / "config.json").write_text(json.dumps(exported_config, indent=2, sort_keys=True) + "\n")
