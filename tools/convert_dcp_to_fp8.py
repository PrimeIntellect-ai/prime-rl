"""Convert a DCP trainer checkpoint into blockwise-FP8 HF weights.

Same layer-parallel CPU conversion as ``dcp_to_bf16``, but each layer is
quantized (on GPU when one is available) before it is written, so no
intermediate bf16 export lands on disk.

Usage (from the prime-rl repo):
    uv run python tools/convert_dcp_to_fp8.py <run>/checkpoints/step_{n} [output_dir]

Writes to ``<ckpt_dir>/weights-FP8`` by default.
"""

import argparse
import json
from pathlib import Path

import torch
from convert_bf16_to_fp8 import quantization_config, quantize_state_dict
from convert_dcp_to_bf16 import open_checkpoint, save_model_assets, write_weights
from torch import Tensor

from prime_rl.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "ckpt_dir", type=Path, help="the DCP checkpoint (<run>/checkpoints/step_{n} or .../step_{n}/trainer)"
    )
    parser.add_argument("output_dir", type=Path, nargs="?", default=None, help="default: <ckpt_dir>/weights-FP8")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1, help="layers converted at once")
    args = parser.parse_args()

    logger = setup_logger("info")
    checkpoint = open_checkpoint(args.ckpt_dir)
    output_dir = args.output_dir if args.output_dir is not None else checkpoint.dcp_dir.parent / "weights-FP8"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modules_to_not_convert: list[str] = []

    def quantize(shard: dict[str, Tensor]) -> dict[str, Tensor]:
        quantized, modules = quantize_state_dict(shard, args.block_size, device)
        modules_to_not_convert.extend(modules)
        return quantized

    logger.info(f"Converting {checkpoint.dcp_dir} to blockwise fp8 weights in {output_dir}")
    write_weights(checkpoint, output_dir, args.workers, transform=quantize)
    save_model_assets(checkpoint.model, checkpoint.model_config, checkpoint.tokenizer_config, output_dir)
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = quantization_config(modules_to_not_convert, args.block_size)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
