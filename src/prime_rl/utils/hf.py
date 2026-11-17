"""Hugging Face Hub downloads for the launchers, which run before any trainer process and import no torch."""

import time
from pathlib import Path

from huggingface_hub import snapshot_download

from prime_rl.configs.sft import DataConfig, SFTDataConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import has_step_dir_name
from prime_rl.utils.utils import format_time


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
    if not isinstance(data, SFTDataConfig):
        return
    if Path(data.name).exists():
        get_logger().info(f"Data {data.name} found at local path, skipping download")
        return

    dataset_name = data.name
    t0 = time.perf_counter()
    get_logger().info(f"Pre-downloading data {dataset_name} at revision {data.revision or 'main'}")
    snapshot = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        revision=data.revision,
        cache_dir=env_vars.get("HF_HUB_CACHE"),
    )
    data.name = snapshot
    get_logger().debug(
        f"Finished pre-downloading data {dataset_name} to {snapshot} in {format_time(time.perf_counter() - t0)}"
    )
