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


def convert_snapshot_to_prime(snapshot: Path) -> ConvertStatus:
    """Write `<snapshot>/prime/` if the snapshot is a convertible HF model."""
    from transformers import AutoConfig

    from prime_rl.trainer.models import get_custom_causal_lm_cls
    from prime_rl.trainer.weights import atomic_save_state_dict, load_state_dict, load_state_dict_keys

    logger = get_logger()
    prime = snapshot / "prime"
    if prime.exists():
        logger.info(f"prime/ already present at {prime}, nothing to do")
        return "exists"

    cls = get_custom_causal_lm_cls(AutoConfig.from_pretrained(snapshot, trust_remote_code=False))
    if cls is None:
        logger.warning("no PrimeRL custom model for this architecture; nothing to convert")
        return "unsupported"

    keys = dict.fromkeys(load_state_dict_keys(snapshot))
    if not keys:
        logger.warning("no safetensors weights in snapshot; conversion needs safetensors (bin format?)")
        return "no-safetensors"
    if cls.is_prime_state_dict(keys):
        logger.info("snapshot already in PrimeRL format; trainers load it directly")
        return "already-prime"
    if not cls.is_hf_state_dict(keys):
        logger.info("snapshot not in HF format; nothing to convert")
        return "not-hf"

    logger.info(f"converting HF -> PrimeRL with {cls.__name__}")
    state_dict = load_state_dict(snapshot)
    cls.convert_to_prime(state_dict)
    atomic_save_state_dict(state_dict, prime)
    logger.info(f"wrote {prime}")
    return "converted"


def run_convert(config: ConvertConfig) -> ConvertStatus:
    snapshot = resolve_snapshot(config.model)
    get_logger().info(f"resolved {config.model} -> {snapshot}")
    return convert_snapshot_to_prime(snapshot)
