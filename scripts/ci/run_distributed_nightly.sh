#!/usr/bin/env bash
# Run the nightly distributed suite on one 8-GPU node. Extra args go to pytest, e.g. -k qwen3_moe.
set -euo pipefail

cd "$(dirname "$0")/../.."
uv run pytest tests/nightly/distributed --durations=0 "$@"
