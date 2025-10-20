#! /usr/bin/env python3
#
# Example Usage:
# python scripts/extract_hf_from_ckpt.py --ckpt-dir outputs/checkpoints/step_2/trainer --output-dir hf_model --utils-repo-id Jackmin108/Qwen3-30B-A3B-Fast
#
# options:
#   --ckpt-dir CKPT_DIR   The directory containing the *.distcp files
#   --output-dir OUTPUT_DIR
#                         The directory to save the Hugging Face repo
#   --utils-repo-id UTILS_REPO_ID
#                         The Hugging Face repo ID containing the config, tokenizer
#                         and chat template
#   --dtype DTYPE         The dtype to save the model in
#

import argparse
import pickle
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import save_file

from prime_rl.trainer.weights import _convert_tt_moe_to_hf_, _has_tt_moe_layers

TARGET_SHARD_SIZE = 3 * 2**30
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main(ckpt_dir: Path, output_dir: Path, utils_repo_id: str, dtype: torch.dtype):
    logger.info(f"Downloading utils from {utils_repo_id}")
    path_snapshot = snapshot_download(
        repo_id=utils_repo_id, repo_type="model", ignore_patterns=["*.safetensors", "model.safetensors.index.json"]
    )

    logger.info(f"Loading metadata from {ckpt_dir}")
    with open(ckpt_dir / ".metadata", "rb") as f:
        metadata = pickle.load(f)

    logger.info("Creating empty state dict")
    state_dict = {"app": {"model": {}}}
    for k in metadata.state_dict_metadata.keys():
        if not k.startswith("app.model"):
            continue
        state_dict["app"]["model"][k.replace("app.model.", "")] = torch.empty(
            metadata.state_dict_metadata[k].size, dtype=dtype
        )

    logger.info(f"Loading checkpoint from {ckpt_dir}")
    dcp.load(state_dict=state_dict, checkpoint_id=str(ckpt_dir))

    if _has_tt_moe_layers(state_dict["app"]["model"]):
        logger.info("Converting TT-MoE layers to HF format")
        _convert_tt_moe_to_hf_(state_dict["app"]["model"])

    shard_keys = [[]]
    cur_size = 0
    for key in state_dict["app"]["model"].keys():
        shard_keys[-1].append(key)
        cur_size += state_dict["app"]["model"][key].numel() * dtype.itemsize
        if cur_size >= TARGET_SHARD_SIZE:
            shard_keys.append([])
            cur_size = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(shard_keys)} shards to {output_dir}")
    for i, _shard_keys in enumerate(shard_keys, start=1):
        shard_state_dict = {k: state_dict["app"]["model"][k] for k in _shard_keys}
        save_file(shard_state_dict, output_dir / f"model-{i:05d}-of-{len(shard_keys):05d}.safetensors")

    utils_paths = [p for p in Path(path_snapshot).glob("*") if "safetensors" not in str(p)]
    logger.info(f"Saving utils to {output_dir}")
    for path in utils_paths:
        shutil.copy(path, output_dir / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="outputs/checkpoints/step_2/trainer",
        help="The directory containing the *.distcp files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="hf_model", help="The directory to save the Hugging Face repo"
    )
    parser.add_argument(
        "--utils-repo-id",
        type=str,
        default="Jackmin108/Qwen3-30B-A3B-Fast",
        help="The Hugging Face repo ID containing the config, tokenizer and chat template",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The dtype to save the model in")
    args = parser.parse_args()
    main(Path(args.ckpt_dir), Path(args.output_dir), args.utils_repo_id, DTYPE_MAP[args.dtype])
