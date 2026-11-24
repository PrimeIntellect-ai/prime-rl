"""Hugging Face Hub downloads for the launchers, which run before any trainer process and import no torch."""

import time
from pathlib import Path

from huggingface_hub import snapshot_download

from prime_rl.configs.supervised import DataConfig
from prime_rl.configs.val import HFDatasetConfig
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import has_step_dir_name


def pre_download_model(model_name: str, *, skip_weights: bool = False) -> None:
    """Pre-download model from HuggingFace Hub so all nodes have cached weights before training.

    With ``skip_weights`` (weights that do not come from the HF model), only config and tokenizer files are fetched.
    """
    # A checkpoint step directory (gone by a resume, which reads the run's own copies of its files) is never on the Hub.
    if Path(model_name).exists() or has_step_dir_name(model_name):
        get_logger().info(f"Model {model_name} is a local path, skipping download")
        return
    t0 = time.perf_counter()
    if skip_weights:
        get_logger().info(f"Pre-downloading config and tokenizer for {model_name} (skipping weights)")
        path = snapshot_download(
            repo_id=model_name, repo_type="model", allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"]
        )
    else:
        get_logger().info(f"Pre-downloading model {model_name}")
        path = snapshot_download(repo_id=model_name, repo_type="model")
    get_logger().debug(
        f"Finished pre-downloading model {model_name} to {path} in {format_time(time.perf_counter() - t0)}"
    )


def pre_download_data(data: DataConfig, env_vars: dict[str, str]) -> None:
    if data.type == "chat" and data.fake is None:
        pre_download_sources(data.source, env_vars)


def pre_download_sources(sources: list[HFDatasetConfig], env_vars: dict[str, str]) -> None:
    """Download every source's HF dataset into the HF cache and point the source at the snapshot."""
    snapshots: dict[tuple[str, str | None], str] = {}
    for source in sources:
        if Path(source.dataset).exists():
            continue
        key = (source.dataset, source.revision)
        if key not in snapshots:
            t0 = time.perf_counter()
            get_logger().info(f"Pre-downloading data {source.dataset} at revision {source.revision or 'main'}")
            snapshots[key] = snapshot_download(
                repo_id=source.dataset,
                repo_type="dataset",
                revision=source.revision,
                cache_dir=env_vars.get("HF_HUB_CACHE"),
            )
            get_logger().debug(
                f"Finished pre-downloading data {source.dataset} to {snapshots[key]} in "
                f"{format_time(time.perf_counter() - t0)}"
            )
        source.dataset = snapshots[key]
