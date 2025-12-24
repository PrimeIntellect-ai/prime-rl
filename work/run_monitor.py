# run_monitor.py
#!/usr/bin/env python3
"""
Standalone runner for Prime Monitor.
Copy this entire 'prime_monitor' directory to your project and run:
    python run_monitor.py -o outputs --trainer-gpus 2,3,4,5,6,7 --inference-gpus 0,1
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import prime_monitor
sys.path.insert(0, str(Path(__file__).parent))

from prime_monitor import PrimeMonitor, MonitorConfig
import argparse


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
    
    print(f"[PrimeMonitor] Config:", file=sys.stderr)
    print(f"  output_dir: {config.output_dir}", file=sys.stderr)
    print(f"  trainer_gpus: {config.trainer_gpu_ids}", file=sys.stderr)
    print(f"  inference_gpus: {config.inference_gpu_ids}", file=sys.stderr)
    
    app = PrimeMonitor(config)
    app.run()


if __name__ == "__main__":
    main()