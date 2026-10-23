#!/usr/bin/env bash
# Run the nightly distributed suite on one node with at least 4 GPUs. Extra args go to pytest, e.g. -k qwen3_moe.
set -euo pipefail

cd "$(dirname "$0")/../.."
uv run pytest tests/regression/distributed --durations=0 "$@"
