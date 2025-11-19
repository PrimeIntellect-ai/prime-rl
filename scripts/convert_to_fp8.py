#!/usr/bin/env python3
"""
Convert a HuggingFace model to FP8 format using blockwise (128x128) quantization.

Usage:
    python scripts/convert_to_fp8.py <model_path_or_id> <output_path> [--max-workers N]

Examples:
    # Convert a local model
    python scripts/convert_to_fp8.py /path/to/model /path/to/output

    # Convert a HuggingFace model (will download first)
    python scripts/convert_to_fp8.py Qwen/Qwen3-0.6B /path/to/output

    # Use custom number of workers
    python scripts/convert_to_fp8.py /path/to/model /path/to/output --max-workers 4
"""

import argparse
import sys
from pathlib import Path

from prime_rl.trainer.weights import convert_model_to_fp8
from prime_rl.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace model to FP8 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the input model directory (HF format) or HuggingFace model ID",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the FP8 quantized model",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(args.log_level)

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    # Check if model_path is a HuggingFace model ID (contains '/')
    # If it doesn't exist as a local path, try downloading from HF
    if "/" in args.model_path and not model_path.exists():
        try:
            from huggingface_hub import snapshot_download

            print(f"Model path '{args.model_path}' not found locally. Downloading from HuggingFace...")
            download_path = Path("/tmp") / args.model_path.replace("/", "_")
            snapshot_download(
                repo_id=args.model_path,
                local_dir=str(download_path),
                local_dir_use_symlinks=False,
            )
            model_path = download_path
            print(f"Downloaded model to {model_path}")
        except ImportError:
            print("Error: huggingface_hub not available. Please install it or provide a local model path.")
            sys.exit(1)
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model path '{model_path}' does not exist.")
        sys.exit(1)

    print(f"Converting model from {model_path} to FP8 format...")
    print(f"Output will be saved to {output_path}")
    print(f"Using {args.max_workers} worker threads")

    try:
        convert_model_to_fp8(
            model_path=str(model_path),
            output_path=str(output_path),
            max_workers=args.max_workers,
        )
        print("\n✓ Successfully converted model to FP8 format!")
        print(f"  Output directory: {output_path}")
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

