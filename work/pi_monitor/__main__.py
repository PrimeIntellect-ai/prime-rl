# prime_monitor/__main__.py
"""Entry point for Prime Monitor."""

import argparse
import sys
from pathlib import Path

from .app import PrimeMonitor
from .config import MonitorConfig


def main():
    parser = argparse.ArgumentParser(
        description="Prime-RL Monitor - TUI for monitoring prime-rl training"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the outputs directory (default: outputs)"
    )
    parser.add_argument(
        "-r", "--refresh",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--trainer-gpus",
        type=str,
        default="0",
        help="Comma-separated list of trainer GPU IDs (default: 0)"
    )
    parser.add_argument(
        "--inference-gpus",
        type=str,
        default="1",
        help="Comma-separated list of inference GPU IDs (default: 1)"
    )
    
    args = parser.parse_args()
    
    config = MonitorConfig(
        output_dir=args.output_dir,
        refresh_interval=args.refresh,
        trainer_gpu_ids=[int(x) for x in args.trainer_gpus.split(",")],
        inference_gpu_ids=[int(x) for x in args.inference_gpus.split(",")],
    )
    
    # Debug output
    print(f"[PrimeMonitor] Config:", file=sys.stderr)
    print(f"  output_dir: {config.output_dir}", file=sys.stderr)
    print(f"  trainer_gpus: {config.trainer_gpu_ids}", file=sys.stderr)
    print(f"  inference_gpus: {config.inference_gpu_ids}", file=sys.stderr)
    
    app = PrimeMonitor(config)
    app.run()


if __name__ == "__main__":
    main()