#! /usr/bin/env python3
#
# Example Usage:
# uv run python scripts/convert_to_fp8.py \
#     /tmp/PrimeIntellect_Qwen3-0.6B-Reverse-Text-RL-safetensors \
#     /root/prime-rl/models/Qwen3-0.6B-Reverse-Text-RL-fp8 \
#     --max-workers 2
#
# options:
#   input_path          Path to the input model directory (HF format with safetensors)
#   output_path         Path to save the FP8 quantized model
#   --max-workers MAX_WORKERS
#                       Number of worker threads for parallel processing (default: 4)

import argparse
from pathlib import Path

from loguru import logger

from prime_rl.trainer.weights import convert_model_to_fp8


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to FP8 format using blockwise (128x128) quantization"
    )
    parser.add_argument("input_path", type=Path, help="Path to the input model directory (HF format with safetensors)")
    parser.add_argument("output_path", type=Path, help="Path to save the FP8 quantized model")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing (default: 4)",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        logger.error(f"Input path does not exist: {args.input_path}")
        return 1

    if not args.input_path.is_dir():
        logger.error(f"Input path is not a directory: {args.input_path}")
        return 1

    logger.info(f"Converting model from {args.input_path} to {args.output_path}")
    logger.info(f"Using {args.max_workers} worker threads")

    try:
        convert_model_to_fp8(args.input_path, args.output_path, max_workers=args.max_workers)
        logger.info(f"Successfully converted model to FP8 format: {args.output_path}")
        return 0
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        raise


if __name__ == "__main__":
    exit(main())

