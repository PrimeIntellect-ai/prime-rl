"""Prepare data on one launcher process before trainer ranks start."""

import argparse
import json
from pathlib import Path

from prime_rl.configs.sft import SFTDataConfig
from prime_rl.utils.logger import get_logger


def pre_download_data(data: SFTDataConfig, env_vars: dict[str, str]) -> str:
    if Path(data.name).exists():
        return data.name

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    get_logger().info(f"Pre-downloading data {data.name} at revision {data.revision or 'main'}")
    snapshot = snapshot_download(
        repo_id=data.name,
        repo_type="dataset",
        revision=data.revision,
        cache_dir=env_vars.get("HF_HUB_CACHE"),
    )
    subsets = data.subsets if data.subsets is not None else [None] * (len(data.splits) if data.splits else 1)
    splits = data.splits if data.splits is not None else ["train"] * len(subsets)
    for subset, split in zip(subsets, splits, strict=True):
        load_dataset(snapshot, subset, split=split, cache_dir=env_vars.get("HF_DATASETS_CACHE"))
    get_logger().info(f"Using local data snapshot {snapshot}")
    return snapshot


def prepare_data_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    env_vars = config.get("env_vars", {})
    changed = False
    for data in (config.get("data"), (config.get("val") or {}).get("data")):
        if data is None or data.get("type") != "sft":
            continue
        source = pre_download_data(SFTDataConfig.model_validate(data), env_vars)
        if source != data["name"]:
            data["name"] = source
            changed = True
    if changed:
        temporary = config_path.with_suffix(".prepared.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    prepare_data_config(args.config)


if __name__ == "__main__":
    main()
