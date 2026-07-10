"""Standalone HF -> PrimeRL weight-format conversion.

Produces `<snapshot>/prime/`, the grouped-MoE format `load_dcp_from_hf` loads
directly (skipping the inline, distributed conversion). Single process,
CPU-only; the write is atomic so a crash never leaves a partial `prime/`.
Reused by the model-cache converter sidecar to prepare the cache without a
live trainer.
"""

from pathlib import Path
from typing import Literal

from prime_rl.configs.convert import ConvertConfig
from prime_rl.utils.logger import get_logger

# Terminal outcomes for a snapshot; returned so callers/tests need not scrape logs.
ConvertStatus = Literal["converted", "exists", "already-prime", "unsupported", "no-safetensors", "not-hf"]


def resolve_snapshot(model: str) -> Path:
    """Local snapshot dir for a repo id or path. Downloads if not cached
    (honours HF_HUB_OFFLINE / HF_HUB_CACHE); repo-id resolution follows the
    revision consumers use, so multiple on-disk snapshots aren't a hazard."""
    if Path(model).exists():
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model, repo_type="model"))


def convert_snapshot_to_prime(
    snapshot: Path,
    model_cls: type | None = None,
    conversion_dir: Path | None = None,
) -> ConvertStatus:
    """Write `<snapshot>/prime/` if the snapshot is a convertible HF model."""
    from transformers import AutoConfig

    from prime_rl.trainer.models import get_custom_causal_lm_cls, get_custom_vlm_cls
    from prime_rl.trainer.weights import (
        is_state_dict_complete,
        load_state_dict_keys,
        stream_convert_state_dict,
    )

    logger = get_logger()
    prime = (conversion_dir or snapshot) / "prime"
    if model_cls is None:
        try:
            model_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=False)
        except ValueError as exc:
            logger.warning(f"unsupported model configuration: {exc}")
            return "unsupported"
        model_cls = get_custom_vlm_cls(model_config) or get_custom_causal_lm_cls(model_config)

    if model_cls is None:
        logger.warning("no PrimeRL custom model for this architecture; nothing to convert")
        return "unsupported"

    if prime.exists():
        prime_keys = dict.fromkeys(load_state_dict_keys(prime))
        if is_state_dict_complete(prime) and model_cls.is_prime_state_dict(prime_keys):
            logger.info(f"complete prime/ already present at {prime}, nothing to do")
            return "exists"
        logger.warning(f"existing prime/ at {prime} is incomplete or invalid; rebuilding it")

    keys = dict.fromkeys(load_state_dict_keys(snapshot))
    if not keys:
        logger.warning("no safetensors weights in snapshot; conversion needs safetensors (bin format?)")
        return "no-safetensors"
    if model_cls.is_prime_state_dict(keys):
        logger.info("snapshot already in PrimeRL format; trainers load it directly")
        return "already-prime"
    if not model_cls.is_hf_state_dict(keys):
        logger.info("snapshot not in HF format; nothing to convert")
        return "not-hf"

    logger.info(f"converting HF -> PrimeRL with {model_cls.__name__}")
    stream_convert_state_dict(
        snapshot,
        prime,
        model_cls.convert_layer_to_prime,
        model_cls.is_prime_state_dict,
        overwrite=prime.exists(),
    )
    logger.info(f"wrote {prime}")
    return "converted"


def run_convert(config: ConvertConfig) -> ConvertStatus:
    snapshot = resolve_snapshot(config.model)
    get_logger().info(f"resolved {config.model} -> {snapshot}")
    return convert_snapshot_to_prime(snapshot)
