"""Publish an HF safetensors checkpoint to ModelExpress, for running engines to install with `/swap-checkpoint`.

    uv run python tools/publish_checkpoint.py MODEL_DIR STEP --model-name NAME [--mx-host H] [--mx-port P] [--ranks N]
        [--control-port 8090]

MODEL_DIR is published as the version of STEP under the ModelExpress model NAME, which must be the engines'
`weight_broadcast.model_name` (by default their `vllm.model`). The publisher holds the checkpoint in host memory and
serves it until released. Control it over HTTP on rank 0:

    curl http://<host>:8090/status
    curl -X POST http://<host>:8090/publish -d '{"model_dir": "/path/to/step-300", "step": 300}'
    curl -X POST http://<host>:8090/release      # or SIGTERM

With --serve-modelexpress, the publisher also runs the ModelExpress server the engines connect to
(`weight_broadcast.host` = this host, `weight_broadcast.port` = --mx-port).

The publisher refuses to start, and a publish fails, rather than leave less host memory available than the larger of
--min-available-gib and --min-available-fraction of the total. Reads bypass the page cache (O_DIRECT).

`/publish` rolls to another checkpoint of the same model: it retires the current version (waiting for reads in
flight), reads the new checkpoint into the same buffers and returns once the new version is READY.
"""

import argparse
from pathlib import Path

from prime_rl.transports.weights.checkpoint_publisher import MemoryFloor, PublisherConfig, run_publisher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--model-name", required=True, help="ModelExpress model name the engines follow")
    parser.add_argument("--mx-host", default="localhost")
    parser.add_argument("--mx-port", type=int, default=8001)
    parser.add_argument("--ranks", type=int, default=1, help="publisher processes sharing the checkpoint")
    parser.add_argument("--read-threads", type=int, default=16, help="reader threads per rank")
    parser.add_argument("--no-pin", action="store_true", help="leave pinning the host buffers to NIXL registration")
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=8090)
    parser.add_argument("--metadata-port", type=int, default=15555, help="first NIXL metadata port (one per rank)")
    parser.add_argument("--worker-port", type=int, default=16555, help="first ModelExpress worker port (one per rank)")
    parser.add_argument("--release-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--min-available-gib",
        type=float,
        default=200.0,
        help="host memory to leave available (with --min-available-fraction, the larger applies)",
    )
    parser.add_argument("--min-available-fraction", type=float, default=0.15)
    parser.add_argument(
        "--serve-modelexpress",
        type=Path,
        metavar="STATE_DIR",
        help="start a ModelExpress server on --mx-port beside the publisher (state and logs in STATE_DIR)",
    )
    args = parser.parse_args()
    run_publisher(
        PublisherConfig(
            model_dir=args.model_dir.resolve(),
            step=args.step,
            model_name=args.model_name,
            mx_host=args.mx_host,
            mx_port=args.mx_port,
            ranks=args.ranks,
            read_threads=args.read_threads,
            pin=not args.no_pin,
            control_host=args.control_host,
            control_port=args.control_port,
            metadata_port=args.metadata_port,
            worker_port=args.worker_port,
            release_timeout=args.release_timeout,
            memory_floor=MemoryFloor(gib=args.min_available_gib, fraction=args.min_available_fraction),
            serve_modelexpress=args.serve_modelexpress,
        )
    )


if __name__ == "__main__":
    main()
