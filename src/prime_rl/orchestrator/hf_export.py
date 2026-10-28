"""Validate and publish Hugging Face exports of trainer DCP checkpoints.

The by-step exports are immutable.  ``best`` and ``latest`` are relative symlinks,
and ``COMPLETE.json`` is the commit marker for a requested pair of exports.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from prime_rl.configs.trainer import ModelConfig, TokenizerConfig
from prime_rl.utils.pathing import get_config_dir, preflight_checkpoint_storage

_EXPORT_MANIFEST = "EXPORT.json"
_COMPLETE = "COMPLETE.json"
_RUN_CONFIG_NAMES = ("trainer.json", "sft.json")
# These PrimeRL implementations explicitly use the standard HF state-dict layout.
_DIRECT_HF_MODEL_TYPES = {"llama", "qwen3"}


def export_best_and_latest(*, output_dir: Path, best_step: int | None, latest_step: int) -> bool:
    """Export BEST then LATEST, publishing them only after both validate.

    DCP input is read from ``output_dir/checkpoints/step_N/trainer``.  Valid HF
    artifacts are published under ``output_dir/exports``.  ``best_step=None``
    requests only LATEST.  Any prior COMPLETE marker is preserved through
    conversion and validation, then invalidated immediately before pointer updates.
    """
    output_dir = Path(output_dir)
    steps = [latest_step] if best_step is None else [best_step, latest_step]
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("Checkpoint steps must be non-negative integers")

    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    required_bytes = 0
    for step in dict.fromkeys(steps):
        if (export_dir / f"step_{step}").exists():
            continue
        source = checkpoints_dir / f"step_{step}"
        required_bytes += sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    preflight_checkpoint_storage(export_dir, required_bytes=required_bytes)

    model_config, tokenizer_config, config_path, config_sha256 = _resolve_run_config(output_dir)
    if model_config.lora is not None:
        raise ValueError("LoRA checkpoints cannot be exported as full HF models")

    # Do not publish either pointer until every requested conversion has passed.
    manifests: dict[int, dict[str, Any]] = {}
    for step in dict.fromkeys(steps):
        dcp_dir = _resolve_dcp_dir(checkpoints_dir, step)
        provenance = _provenance(dcp_dir, config_path, config_sha256, model_config, tokenizer_config)
        manifests[step] = _export_step(
            step=step,
            dcp_dir=dcp_dir,
            export_dir=export_dir,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            provenance=provenance,
        )

    # Preserve the prior committed pair until all replacements are valid. From
    # this point onward, a failure must leave no marker for a partially swapped pair.
    (export_dir / _COMPLETE).unlink(missing_ok=True)
    if best_step is not None:
        _atomic_symlink(f"step_{best_step}", export_dir / "best", export_dir)
    _atomic_symlink(f"step_{latest_step}", export_dir / "latest", export_dir)

    def summary(step: int) -> dict[str, Any]:
        manifest = manifests[step]
        return {
            "step": step,
            "path": f"step_{step}",
            "manifest": f"step_{step}/{_EXPORT_MANIFEST}",
            "provenance_sha256": manifest["provenance_sha256"],
            "shards": manifest["shards"],
        }

    complete = {
        "format_version": 1,
        "status": "complete",
        "best": summary(best_step) if best_step is not None else None,
        "latest": summary(latest_step),
        "exports": [summary(step) for step in dict.fromkeys(steps)],
    }
    _write_json_atomic(export_dir / _COMPLETE, complete)
    return True


def _export_step(
    *,
    step: int,
    dcp_dir: Path,
    export_dir: Path,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    final_dir = export_dir / f"step_{step}"
    if final_dir.exists() or final_dir.is_symlink():
        if final_dir.is_symlink() or not final_dir.is_dir():
            raise RuntimeError(f"Immutable export path is not a directory: {final_dir}")
        manifest = _read_json(final_dir / _EXPORT_MANIFEST)
        if manifest.get("provenance") != provenance:
            raise RuntimeError(f"Existing export for step {step} has different provenance; refusing to replace it")
        _validate_export(final_dir, manifest)
        _assert_source_unchanged(step, provenance)
        return manifest

    staging = Path(tempfile.mkdtemp(prefix=f".step_{step}.", suffix=".tmp", dir=export_dir))
    try:
        manifest = _convert_checkpoint(
            step=step,
            dcp_dir=dcp_dir,
            staging=staging,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            provenance=provenance,
        )
        _assert_source_unchanged(step, provenance)
        # Another exporter may have won the publish race while this one worked.
        if final_dir.exists():
            existing = _read_json(final_dir / _EXPORT_MANIFEST)
            if existing.get("provenance") != provenance:
                raise RuntimeError(f"Existing export for step {step} has different provenance")
            _validate_export(final_dir, existing)
            _assert_source_unchanged(step, provenance)
            return existing
        staging.rename(final_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _convert_checkpoint(
    *,
    step: int,
    dcp_dir: Path,
    staging: Path,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    metadata = FileSystemReader(dcp_dir).read_metadata()
    lora_keys = [
        key for key in metadata.state_dict_metadata if "lora_A" in key or "lora_B" in key or ".base_layer." in key
    ]
    if lora_keys:
        raise ValueError(f"Checkpoint contains LoRA weights (for example {lora_keys[0]})")

    model, prime_layout = _load_conversion_model(model_config)
    dcp_load(state_dict={"app": {"model": model.state_dict()}}, checkpoint_id=dcp_dir)

    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        value = value.detach().to(device="cpu")
        if value.is_floating_point():
            dtype = torch.float32 if keep_in_fp32 is not None and keep_in_fp32(key) else torch.bfloat16
            value = value.to(dtype=dtype)
        state_dict[key] = value.contiguous()

    if getattr(model.config, "tie_word_embeddings", False):
        for key in getattr(model, "_tied_weights_keys", []):
            state_dict.pop(key, None)

    if prime_layout:
        from prime_rl.utils.weights import convert_state_dict_to_hf

        state_dict = convert_state_dict_to_hf(model, state_dict)
    else:
        from transformers.core_model_loading import revert_weight_conversion

        state_dict = revert_weight_conversion(model, state_dict)

    tensors = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state_dict.items()}
    if not tensors:
        raise RuntimeError(f"Checkpoint at {dcp_dir} produced an empty HF state dict")

    from prime_rl.utils.weights import save_state_dict

    # save_state_dict consumes its input, so tensor metadata was captured above.
    save_state_dict(state_dict, staging, save_sharded=True)
    _save_model_assets(model, model_config, tokenizer_config, staging)
    # Validation loads a second model; release the conversion model first.
    del model

    shards = [
        {"path": path.name, "size": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(staging.glob("*.safetensors"))
    ]
    if not shards:
        raise RuntimeError(f"Conversion produced no safetensors shards in {staging}")
    manifest = {
        "format_version": 1,
        "step": step,
        "model": model_config.name,
        "trust_remote_code": bool(model_config.trust_remote_code),
        "tokenizer_trust_remote_code": bool(
            tokenizer_config.trust_remote_code
            if tokenizer_config.trust_remote_code is not None
            else model_config.trust_remote_code
        ),
        "provenance": provenance,
        "provenance_sha256": _hash_json(provenance),
        "tensors": tensors,
        "shards": shards,
    }
    _write_json(staging / _EXPORT_MANIFEST, manifest)
    _validate_export(staging, manifest)
    return manifest


def _load_conversion_model(model_config: ModelConfig):
    config = AutoConfig.from_pretrained(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
    )
    if config.model_type in _DIRECT_HF_MODEL_TYPES:
        target_config = getattr(config, "text_config", config)
        if model_config.debug.num_layers is not None:
            target_config.num_hidden_layers = min(model_config.debug.num_layers, target_config.num_hidden_layers)
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        return model, False

    # Custom PrimeRL layouts need the training model's declarative conversion.
    from prime_rl.trainer.model import get_model

    conversion_config = model_config.model_copy(update={"attn": "flash_attention_2", "cp": 1, "dp_replicate": 1})
    return get_model(conversion_config, device=torch.device("cpu"), dtype=torch.float32), True


def _save_model_assets(
    model,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    output_dir: Path,
) -> None:
    model.config.save_pretrained(output_dir)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        generation_config = GenerationConfig.from_model_config(model.config)
    generation_config = deepcopy(generation_config)
    generation_config.use_cache = True
    generation_config.save_pretrained(output_dir)

    tokenizer_name = tokenizer_config.name or model_config.name
    trust_remote_code = tokenizer_config.trust_remote_code
    if trust_remote_code is None:
        trust_remote_code = model_config.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    if tokenizer_config.chat_template is not None:
        template = Path(tokenizer_config.chat_template)
        tokenizer.chat_template = template.read_text() if template.is_file() else tokenizer_config.chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(output_dir)

    # Local remote-code models need their code beside the exported config.
    source_dir = Path(model_config.name)
    if source_dir.is_dir():
        for path in source_dir.glob("*.py"):
            shutil.copyfile(path, output_dir / path.name)


def _validate_export(export_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("tensors"), dict):
        raise RuntimeError(f"Invalid export manifest in {export_dir}")
    if manifest.get("provenance_sha256") != _hash_json(manifest.get("provenance")):
        raise RuntimeError(f"Provenance hash mismatch in {export_dir}")

    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    tokenizer_trust_remote_code = bool(manifest.get("tokenizer_trust_remote_code", trust_remote_code))
    AutoConfig.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=tokenizer_trust_remote_code,
    )
    encoded = tokenizer("HF export smoke", return_tensors="pt")
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise RuntimeError(f"Tokenizer in {export_dir} produced invalid input IDs")
    if not isinstance(tokenizer.decode(input_ids[0]), str):
        raise RuntimeError(f"Tokenizer in {export_dir} failed decode smoke test")

    _validate_safetensors(export_dir, manifest)
    _validate_shard_hashes(export_dir, manifest)

    loaded = AutoModelForCausalLM.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
        dtype="auto",
        attn_implementation="eager",
    )
    model, loading_info = loaded
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if loading_info.get(key):
            raise RuntimeError(f"HF load reported {key}: {loading_info[key][:5]}")
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            raise RuntimeError(f"HF load left {name} on the meta device")
        if (parameter.is_floating_point() or parameter.is_complex()) and not torch.isfinite(parameter).all():
            raise RuntimeError(f"HF load produced non-finite parameter {name}")

    model.eval()
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    logits = getattr(output, "logits", None)
    if logits is None or logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise RuntimeError(f"HF forward in {export_dir} did not produce token logits")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"HF forward in {export_dir} produced non-finite logits")


def _validate_safetensors(export_dir: Path, manifest: dict[str, Any]) -> None:
    shard_paths = sorted(export_dir.glob("*.safetensors"))
    if not shard_paths:
        raise RuntimeError(f"No safetensors shards in {export_dir}")
    index_path = export_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid safetensors index in {export_dir}")
    else:
        if [path.name for path in shard_paths] != [SAFE_WEIGHTS_NAME]:
            raise RuntimeError(f"Unindexed export must contain only {SAFE_WEIGHTS_NAME}")
        weight_map = None

    actual_map: dict[str, str] = {}
    total_size = 0
    expected = manifest["tensors"]
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in actual_map:
                    raise RuntimeError(f"Tensor {key} occurs in multiple shards")
                actual_map[key] = shard_path.name
                if key not in expected:
                    raise RuntimeError(f"Unexpected tensor {key} in {shard_path.name}")
                tensor = shard.get_tensor(key)
                wanted = expected[key]
                if list(tensor.shape) != wanted["shape"]:
                    raise RuntimeError(f"Shape mismatch for {key}: {list(tensor.shape)} != {wanted['shape']}")
                if str(tensor.dtype) != wanted["dtype"]:
                    raise RuntimeError(f"Dtype mismatch for {key}: {tensor.dtype} != {wanted['dtype']}")
                if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Non-finite values in tensor {key}")
                total_size += tensor.numel() * tensor.element_size()

    if set(actual_map) != set(expected):
        missing = sorted(set(expected) - set(actual_map))
        raise RuntimeError(f"Missing tensors: {missing[:5]}")
    if weight_map is not None:
        if weight_map != actual_map:
            raise RuntimeError(f"Safetensors index does not match shards in {export_dir}")
        if set(weight_map.values()) != {path.name for path in shard_paths}:
            raise RuntimeError(f"Safetensors index and shard files disagree in {export_dir}")
        indexed_size = index.get("metadata", {}).get("total_size")
        if indexed_size is not None and indexed_size != total_size:
            raise RuntimeError(f"Safetensors index total_size mismatch in {export_dir}")


def _validate_shard_hashes(export_dir: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("shards")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Missing shard hashes in {export_dir}")
    actual_names = {path.name for path in export_dir.glob("*.safetensors")}
    if {record.get("path") for record in records} != actual_names:
        raise RuntimeError(f"Shard manifest does not match files in {export_dir}")
    for record in records:
        path = export_dir / record["path"]
        if path.stat().st_size != record["size"] or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Shard hash mismatch: {path}")


def _resolve_run_config(output_dir: Path) -> tuple[ModelConfig, TokenizerConfig, Path, str]:
    config_dir = get_config_dir(output_dir)
    for name in _RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            raw = json.loads(path.read_text())
            model_config = ModelConfig(**raw["model"])
            tokenizer_config = TokenizerConfig(**raw.get("tokenizer", {}))
            if tokenizer_config.name is None:
                tokenizer_config.name = model_config.name
            return model_config, tokenizer_config, path, _sha256_file(path)
    raise FileNotFoundError(f"No resolved trainer.json or sft.json found under {config_dir}")


def _resolve_dcp_dir(checkpoints_dir: Path, step: int) -> Path:
    step_dir = checkpoints_dir / f"step_{step}"
    dcp_dir = step_dir / "trainer"
    if not (dcp_dir / ".metadata").is_file():
        raise FileNotFoundError(f"No DCP checkpoint metadata at {dcp_dir / '.metadata'}")
    return dcp_dir


def _provenance(
    dcp_dir: Path,
    config_path: Path,
    config_sha256: str | None,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
) -> dict[str, Any]:
    return {
        "checkpoint": str(dcp_dir.resolve()),
        "checkpoint_metadata_sha256": _sha256_file(dcp_dir / ".metadata"),
        "checkpoint_files_sha256": _tree_sha256(dcp_dir),
        "run_config": str(config_path.resolve()),
        "run_config_sha256": config_sha256 or _sha256_file(config_path),
        "model": model_config.name,
        "tokenizer": tokenizer_config.name or model_config.name,
    }


def _assert_source_unchanged(step: int, provenance: dict[str, Any]) -> None:
    checkpoint = Path(provenance["checkpoint"])
    config_path = Path(provenance["run_config"])
    if (
        _tree_sha256(checkpoint) != provenance["checkpoint_files_sha256"]
        or _sha256_file(config_path) != provenance["run_config_sha256"]
    ):
        raise RuntimeError(f"Checkpoint or run config changed while exporting step {step}")


def _tree_sha256(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    return _hash_json({str(file.relative_to(path)): _sha256_file(file) for file in files})


def _atomic_symlink(target: str, pointer: Path, parent: Path) -> None:
    temporary = parent / f".{pointer.name}.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(target, temporary)
        os.replace(temporary, pointer)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def _write_json_atomic(path: Path, value: Any) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}") from error


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_best_and_latest"]
