"""Validate and publish Hugging Face exports of trainer DCP checkpoints.

The by-step exports are immutable.  ``best`` and ``latest`` are relative symlinks,
and ``COMPLETE.json`` is the commit marker for a requested pair of exports.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
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
_CONVERTER_VERSION = 2
_RUN_CONFIG_NAMES = ("trainer.json", "sft.json")
_SOURCE_ASSET_SUFFIXES = {".json", ".jinja", ".txt", ".py", ".model", ".tiktoken"}
_UTF8_ASSET_SUFFIXES = {".json", ".jinja", ".txt", ".py", ".tiktoken"}
# These PrimeRL implementations explicitly use the standard HF state-dict layout.
_DIRECT_HF_MODEL_TYPES = {"llama", "qwen3"}


def export_best_and_latest(*, output_dir: Path, best_step: int | None, latest_step: int) -> bool:
    """Export BEST then LATEST, publishing them only after both validate.

    All operations for one export root are serialized. A prior COMPLETE marker
    survives unrelated early failures, but is removed before pointer mutation or
    when an already-published export fails revalidation.
    """
    output_dir = Path(output_dir)
    steps = [latest_step] if best_step is None else [best_step, latest_step]
    if any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("Checkpoint steps must be non-negative integers")

    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    with _export_lock(export_dir):
        return _export_locked(
            output_dir=output_dir,
            export_dir=export_dir,
            steps=list(dict.fromkeys(steps)),
            best_step=best_step,
            latest_step=latest_step,
        )


def _export_locked(
    *,
    output_dir: Path,
    export_dir: Path,
    steps: list[int],
    best_step: int | None,
    latest_step: int,
) -> bool:
    checkpoints_dir = output_dir / "checkpoints"
    required_bytes = 0
    for step in steps:
        if (export_dir / f"step_{step}").exists():
            continue
        required_bytes += _model_checkpoint_bytes(checkpoints_dir, step)
    preflight_checkpoint_storage(export_dir, required_bytes=required_bytes)

    model_config, tokenizer_config, config_path, config_sha256 = _resolve_run_config(output_dir)
    if model_config.lora is not None:
        raise ValueError("LoRA checkpoints cannot be exported as full HF models")

    # Do not publish either pointer until every requested conversion has passed.
    manifests: dict[int, dict[str, Any]] = {}
    for step in steps:
        dcp_dir = _resolve_dcp_dir(checkpoints_dir, step)
        provenance = _provenance(dcp_dir, config_path, config_sha256, model_config, tokenizer_config)
        manifests[step] = _export_step(
            step=step,
            dcp_dir=dcp_dir,
            export_dir=export_dir,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            provenance=provenance,
            config_path=config_path,
        )

    # From this point onward, a failure must leave no marker for a partial pair.
    _invalidate_complete(export_dir)
    if best_step is not None:
        _atomic_symlink(f"step_{best_step}", export_dir / "best", export_dir)
    else:
        _remove_pointer(export_dir / "best", export_dir)
    _atomic_symlink(f"step_{latest_step}", export_dir / "latest", export_dir)

    def summary(step: int) -> dict[str, Any]:
        manifest = manifests[step]
        return {
            "step": step,
            "path": f"step_{step}",
            "manifest": f"step_{step}/{_EXPORT_MANIFEST}",
            "manifest_sha256": _sha256_file(export_dir / f"step_{step}" / _EXPORT_MANIFEST),
            "provenance_sha256": manifest["provenance_sha256"],
            "shards": manifest["shards"],
        }

    complete = {
        "format_version": 1,
        "status": "complete",
        "best": summary(best_step) if best_step is not None else None,
        "latest": summary(latest_step),
        "exports": [summary(step) for step in steps],
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
    config_path: Path,
) -> dict[str, Any]:
    final_dir = export_dir / f"step_{step}"
    if final_dir.exists() or final_dir.is_symlink():
        if final_dir.is_symlink() or not final_dir.is_dir():
            _invalidate_complete(export_dir)
            raise RuntimeError(f"Immutable export path is not a directory: {final_dir}")
        manifest = _revalidate_published_export(
            final_dir=final_dir,
            export_dir=export_dir,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            provenance=provenance,
        )
        _assert_source_unchanged(step, dcp_dir, config_path, model_config, tokenizer_config, provenance)
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
        _assert_source_unchanged(step, dcp_dir, config_path, model_config, tokenizer_config, provenance)
        if final_dir.exists():
            existing = _revalidate_published_export(
                final_dir=final_dir,
                export_dir=export_dir,
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                provenance=provenance,
            )
            _assert_source_unchanged(step, dcp_dir, config_path, model_config, tokenizer_config, provenance)
            return existing
        _fsync_tree(staging)
        try:
            staging.rename(final_dir)
            _fsync_directory(export_dir)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not final_dir.is_dir():
                raise
            existing = _revalidate_published_export(
                final_dir=final_dir,
                export_dir=export_dir,
                model_config=model_config,
                tokenizer_config=tokenizer_config,
                provenance=provenance,
            )
            _assert_source_unchanged(step, dcp_dir, config_path, model_config, tokenizer_config, provenance)
            return existing
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _revalidate_published_export(
    *,
    final_dir: Path,
    export_dir: Path,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    try:
        _reject_symlinked_assets(final_dir)
        manifest_path = final_dir / _EXPORT_MANIFEST
        _validate_committed_manifest_hash(export_dir, final_dir, manifest_path)
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("provenance") != provenance:
            raise RuntimeError("Published export provenance does not match the requested checkpoint")
        _validate_export(
            final_dir,
            manifest,
            trust_flags=_resolved_trust_flags(model_config, tokenizer_config),
        )
        return manifest
    except Exception:
        _invalidate_complete(export_dir)
        raise


def _validate_committed_manifest_hash(export_dir: Path, final_dir: Path, manifest_path: Path) -> None:
    complete_path = export_dir / _COMPLETE
    if complete_path.is_symlink():
        raise RuntimeError(f"Completion marker must not be a symlink: {complete_path}")
    if not complete_path.exists():
        return
    complete = _read_json(complete_path)
    if not isinstance(complete, dict) or not isinstance(complete.get("exports"), list):
        raise RuntimeError(f"Invalid completion marker schema in {complete_path}")
    matching = [
        entry for entry in complete["exports"] if isinstance(entry, dict) and entry.get("path") == final_dir.name
    ]
    if not matching:
        return
    if len(matching) != 1:
        raise RuntimeError(f"Duplicate step in completion marker: {final_dir.name}")
    entry = matching[0]
    if (
        entry.get("manifest") != f"{final_dir.name}/{_EXPORT_MANIFEST}"
        or not _is_sha256(entry.get("manifest_sha256"))
        or entry["manifest_sha256"] != _sha256_file(manifest_path)
    ):
        raise RuntimeError(f"Published manifest hash mismatch: {manifest_path}")


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
            nemotron_fp32 = model.config.model_type == "nemotron_h" and key.endswith((".mamba.A_log", ".mamba.D"))
            keep_fp32 = keep_in_fp32 is not None and keep_in_fp32(key)
            value = value.to(dtype=torch.float32 if keep_fp32 or nemotron_fp32 else torch.bfloat16)
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

    assets = _asset_records(staging)
    shards = [record for record in assets if record["path"].endswith(".safetensors")]
    if not shards:
        raise RuntimeError(f"Conversion produced no safetensors shards in {staging}")
    model_trust, tokenizer_trust = _resolved_trust_flags(model_config, tokenizer_config)
    manifest = {
        "format_version": 2,
        "step": step,
        "model": model_config.name,
        "trust_remote_code": model_trust,
        "tokenizer_trust_remote_code": tokenizer_trust,
        "provenance": provenance,
        "provenance_sha256": _hash_json(provenance),
        "tensors": tensors,
        "shards": shards,
        "assets": assets,
    }
    _write_json(staging / _EXPORT_MANIFEST, manifest)
    _validate_export(staging, manifest, trust_flags=(model_trust, tokenizer_trust))
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

    if config.model_type == "nemotron_h":
        # Construct only the required custom architecture. Converting the HF
        # config to the PrimeRL config adds training-only defaults without
        # loading any original HF weights.
        from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM

        prime_config = NemotronHConfig(**config.to_dict())
        prime_config._attn_implementation = "flash_attention_2"
        if model_config.debug.num_layers is not None:
            prime_config.num_hidden_layers = min(model_config.debug.num_layers, prime_config.num_hidden_layers)
        return NemotronHForCausalLM.from_config(prime_config), True

    # Custom PrimeRL layouts need the training model's declarative conversion.
    from prime_rl.trainer.model import get_model

    conversion_config = model_config.model_copy(update={"attn": "flash_attention_2", "cp": 1, "dp_replicate": 1})
    # get_model uses model_cls.from_config on CPU. Original HF weights must never
    # be loaded: every parameter comes from the DCP checkpoint below.
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
        tokenizer.chat_template = _resolve_chat_template(tokenizer_config)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(output_dir)

    # Local/cached remote code must travel with the weights. Never silently
    # overwrite differing model and tokenizer modules with the same filename.
    for source in dict.fromkeys((model_config.name, tokenizer_name)):
        for path in _source_directory(source).glob("*.py"):
            target = output_dir / path.name
            if target.exists() and _sha256_file(target) != _sha256_file(path):
                raise RuntimeError(f"Conflicting remote-code assets named {path.name}")
            shutil.copyfile(path, target)


def _validate_export(
    export_dir: Path,
    manifest: dict[str, Any],
    *,
    trust_flags: tuple[bool, bool],
) -> None:
    if not isinstance(manifest, dict) or manifest.get("format_version") != 2:
        raise RuntimeError(f"Invalid export manifest in {export_dir}")
    if type(manifest.get("step")) is not int or manifest["step"] < 0:
        raise RuntimeError(f"Invalid checkpoint step in {export_dir}")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(f"Invalid tensor manifest in {export_dir}")
    for key, tensor in tensors.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, dict)
            or not isinstance(tensor.get("shape"), list)
            or any(type(size) is not int or size < 0 for size in tensor["shape"])
            or not isinstance(tensor.get("dtype"), str)
            or not tensor["dtype"].startswith("torch.")
        ):
            raise RuntimeError(f"Invalid tensor record in {export_dir}: {key}")
    if not isinstance(manifest.get("provenance"), dict):
        raise RuntimeError(f"Invalid provenance in {export_dir}")
    if not _is_sha256(manifest.get("provenance_sha256")) or manifest["provenance_sha256"] != _hash_json(
        manifest["provenance"]
    ):
        raise RuntimeError(f"Provenance hash mismatch in {export_dir}")
    model_trust, tokenizer_trust = trust_flags
    if type(manifest.get("trust_remote_code")) is not bool or manifest["trust_remote_code"] is not model_trust:
        raise RuntimeError(f"Model trust_remote_code does not match resolved config in {export_dir}")
    if (
        type(manifest.get("tokenizer_trust_remote_code")) is not bool
        or manifest["tokenizer_trust_remote_code"] is not tokenizer_trust
    ):
        raise RuntimeError(f"Tokenizer trust_remote_code does not match resolved config in {export_dir}")
    _validate_assets(export_dir, manifest)

    AutoConfig.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=model_trust,
    )
    GenerationConfig.from_pretrained(export_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=tokenizer_trust,
    )
    encoded = tokenizer("HF export smoke", return_tensors="pt")
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise RuntimeError(f"Tokenizer in {export_dir} produced invalid input IDs")
    if not isinstance(tokenizer.decode(input_ids[0]), str):
        raise RuntimeError(f"Tokenizer in {export_dir} failed decode smoke test")

    _validate_safetensors(export_dir, manifest)

    loaded = AutoModelForCausalLM.from_pretrained(
        export_dir,
        local_files_only=True,
        trust_remote_code=model_trust,
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
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(key, str) or not isinstance(shard, str) for key, shard in weight_map.items())
        ):
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
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid safetensors index metadata in {export_dir}")
        indexed_size = metadata.get("total_size")
        if indexed_size is not None and (type(indexed_size) is not int or indexed_size < 0):
            raise RuntimeError(f"Invalid safetensors index total_size in {export_dir}")
        if indexed_size is not None and indexed_size != total_size:
            raise RuntimeError(f"Safetensors index total_size mismatch in {export_dir}")


def _asset_records(export_dir: Path) -> list[dict[str, Any]]:
    _reject_symlinked_assets(export_dir)
    records = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(export_dir).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return records


def _validate_assets(export_dir: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("assets")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Missing asset manifest in {export_dir}")
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid asset record in {export_dir}")
        name = record.get("path")
        size = record.get("size")
        sha256 = record.get("sha256")
        if not isinstance(name, str) or type(size) is not int or size < 0 or not _is_sha256(sha256):
            raise RuntimeError(f"Invalid asset record in {export_dir}: {record}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise RuntimeError(f"Unsafe asset path in {export_dir}: {name}")
        if name == _EXPORT_MANIFEST or "\\" in name or name in expected:
            raise RuntimeError(f"Duplicate or reserved asset path in {export_dir}: {name}")
        expected[name] = record

    _reject_symlinked_assets(export_dir)
    actual = {
        path.relative_to(export_dir).as_posix(): path
        for path in export_dir.rglob("*")
        if path.is_file() and path != export_dir / _EXPORT_MANIFEST
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"Published asset set mismatch; missing={missing[:3]}, extra={extra[:3]}")
    for name, path in actual.items():
        record = expected[name]
        if path.stat().st_size != record["size"] or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Asset hash mismatch: {path}")
        if path.suffix in _UTF8_ASSET_SUFFIXES:
            path.read_bytes().decode("utf-8")
        if path.suffix == ".json":
            _read_json(path)

    shards = manifest.get("shards")
    expected_shards = [record for record in records if record["path"].endswith(".safetensors")]
    if shards != expected_shards:
        raise RuntimeError(f"Shard manifest does not match asset manifest in {export_dir}")


def _reject_symlinked_assets(export_dir: Path) -> None:
    if export_dir.is_symlink():
        raise RuntimeError(f"Export root must not be a symlink: {export_dir}")
    for path in export_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Symlinked assets are not allowed: {path}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _resolve_run_config(output_dir: Path) -> tuple[ModelConfig, TokenizerConfig, Path, str]:
    config_dir = get_config_dir(output_dir)
    for name in _RUN_CONFIG_NAMES:
        path = config_dir / name
        if path.exists():
            raw = _read_json(path)
            if not isinstance(raw, dict):
                raise RuntimeError(f"Resolved run config must be a JSON object: {path}")
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


def _model_checkpoint_bytes(checkpoints_dir: Path, step: int) -> int:
    dcp_dir = _resolve_dcp_dir(checkpoints_dir, step)
    metadata = FileSystemReader(dcp_dir).read_metadata()
    total = sum(
        storage.length
        for index, storage in metadata.storage_data.items()
        if index.fqn == "app.model" or index.fqn.startswith("app.model.")
    )
    if total <= 0:
        raise RuntimeError(f"No model payload found in DCP checkpoint {dcp_dir}")
    return total


def _provenance(
    dcp_dir: Path,
    config_path: Path,
    config_sha256: str | None,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
) -> dict[str, Any]:
    model_trust, tokenizer_trust = _resolved_trust_flags(model_config, tokenizer_config)
    template = _resolve_chat_template(tokenizer_config)
    return {
        "checkpoint_step": dcp_dir.parent.name,
        "checkpoint_metadata_sha256": _sha256_file(dcp_dir / ".metadata"),
        "checkpoint_files_sha256": _tree_sha256(dcp_dir),
        "run_config_sha256": config_sha256 or _sha256_file(config_path),
        "model": model_config.name,
        "tokenizer": tokenizer_config.name or model_config.name,
        "trust_remote_code": {"model": model_trust, "tokenizer": tokenizer_trust},
        "model_assets_sha256": _source_assets_sha256(model_config.name),
        "tokenizer_assets_sha256": _source_assets_sha256(tokenizer_config.name or model_config.name),
        "code_sha256": _source_code_sha256(model_config.name),
        "tokenizer_code_sha256": _source_code_sha256(tokenizer_config.name or model_config.name),
        "chat_template": None
        if template is None
        else {"content": template, "content_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest()},
        "converter": _converter_provenance(),
    }


def _assert_source_unchanged(
    step: int,
    dcp_dir: Path,
    config_path: Path,
    model_config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    provenance: dict[str, Any],
) -> None:
    current = _provenance(dcp_dir, config_path, None, model_config, tokenizer_config)
    if current != provenance:
        raise RuntimeError(f"Checkpoint or export source assets changed while exporting step {step}")


def _tree_sha256(path: Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    return _hash_json({str(file.relative_to(path)): _sha256_file(file) for file in files})


def _source_assets_sha256(source: str) -> str:
    source_dir = _source_directory(source)
    records = {
        path.name: _sha256_file(path)
        for path in sorted(source_dir.iterdir())
        if path.is_file()
        and path.suffix in _SOURCE_ASSET_SUFFIXES
        and not path.name.endswith(".safetensors.index.json")
    }
    if not records:
        raise FileNotFoundError(f"No config/tokenizer assets found in {source_dir}")
    return _hash_json(records)


def _source_code_sha256(source: str) -> str:
    source_dir = _source_directory(source)
    return _hash_json({path.name: _sha256_file(path) for path in sorted(source_dir.glob("*.py")) if path.is_file()})


def _source_directory(source: str) -> Path:
    source_dir = Path(source)
    if source_dir.is_dir():
        return source_dir
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=source,
            repo_type="model",
            local_files_only=True,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.py", "*.model", "*.tiktoken"],
        )
    )


def _resolve_chat_template(tokenizer_config: TokenizerConfig) -> str | None:
    if tokenizer_config.chat_template is None:
        return None
    template = Path(tokenizer_config.chat_template)
    return template.read_text(encoding="utf-8") if template.is_file() else tokenizer_config.chat_template


def _resolved_trust_flags(model_config: ModelConfig, tokenizer_config: TokenizerConfig) -> tuple[bool, bool]:
    model_trust = model_config.trust_remote_code
    tokenizer_trust = tokenizer_config.trust_remote_code
    if tokenizer_trust is None:
        tokenizer_trust = model_trust
    if type(model_trust) is not bool or type(tokenizer_trust) is not bool:
        raise TypeError("Resolved trust_remote_code values must be booleans")
    return model_trust, tokenizer_trust


def _converter_provenance() -> dict[str, Any]:
    from prime_rl.utils import weights

    return {
        "name": "prime_rl.orchestrator.hf_export",
        "version": _CONVERTER_VERSION,
        "content_sha256": {
            "hf_export.py": _sha256_file(Path(__file__)),
            "weights.py": _sha256_file(Path(weights.__file__)),
        },
    }


@contextmanager
def _export_lock(export_dir: Path):
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(export_dir / ".export.lock", flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _invalidate_complete(export_dir: Path) -> None:
    (export_dir / _COMPLETE).unlink(missing_ok=True)
    _fsync_directory(export_dir)


def _remove_pointer(pointer: Path, parent: Path) -> None:
    if pointer.is_symlink():
        pointer.unlink()
        _fsync_directory(parent)
    elif pointer.exists():
        raise RuntimeError(f"Refusing to replace non-symlink pointer {pointer}")


def _atomic_symlink(target: str, pointer: Path, parent: Path) -> None:
    if pointer.exists() and not pointer.is_symlink():
        raise RuntimeError(f"Refusing to replace non-symlink pointer {pointer}")
    temporary = parent / f".{pointer.name}.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(target, temporary)
        os.replace(temporary, pointer)
        _fsync_directory(parent)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def _write_json_atomic(path: Path, value: Any) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _fsync_tree(path: Path) -> None:
    _reject_symlinked_assets(path)
    for asset in path.rglob("*"):
        if asset.is_file():
            fd = os.open(asset, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_directory(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Directory fsync is not available on every supported filesystem.
        pass


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_best_and_latest"]
