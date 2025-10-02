#! /usr/bin/env python3
#
# Example Usage:
# python scripts/convert_moe_to_hf.py --repo-id Jackmin108/Qwen3-30B-A3B-Fast --output-dir hf_model
#
# options:
#   --repo-id REPO_ID     The Hugging Face repo ID containing the torchtitan model
#   --output-dir OUTPUT_DIR
#                         The directory to save the Hugging Face repo
#   --dtype DTYPE         The dtype to save the model in

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.trainer.weights import _convert_tt_moe_to_hf_

TARGET_SHARD_SIZE = 3 * 2**30
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main(output_dir: Path, repo_id: str, dtype: torch.dtype):
    logger.info(f"Downloading utils from {repo_id}")
    path_snapshot = snapshot_download(repo_id=repo_id, repo_type="model")

    logger.info(f"Loading state dict from {path_snapshot}")
    state_dict = {}
    for path in Path(path_snapshot).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    logger.info("Converting TT-MoE layers to HF format")
    _convert_tt_moe_to_hf_(state_dict)

    shard_keys = [[]]
    cur_size = 0
    for key in state_dict.keys():
        shard_keys[-1].append(key)
        cur_size += state_dict[key].numel() * dtype.itemsize
        if cur_size >= TARGET_SHARD_SIZE:
            shard_keys.append([])
            cur_size = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(shard_keys)} shards to {output_dir}")
    for i, _shard_keys in enumerate(shard_keys, start=1):
        shard_state_dict = {k: state_dict[k] for k in _shard_keys}
        save_file(shard_state_dict, output_dir / f"model-{i:05d}-of-{len(shard_keys):05d}.safetensors")

    utils_paths = [p for p in Path(path_snapshot).glob("*") if "safetensors" not in str(p)]
    logger.info(f"Saving utils to {output_dir}")
    for path in utils_paths:
        shutil.copy(path, output_dir / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Jackmin108/Qwen3-30B-A3B-Fast",
        help="The Hugging Face repo ID containing the torchtitan model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="hf_model", help="The directory to save the Hugging Face repo"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The dtype to save the model in")
    args = parser.parse_args()
    main(Path(args.output_dir), args.repo_id, DTYPE_MAP[args.dtype])
