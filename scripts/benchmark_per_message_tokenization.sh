#!/bin/bash
# Benchmark: per-message tokenization vs baseline
#
# Runs wordle RL with TITO disabled (use_token_client=false) to simulate
# standard OpenAI API usage, and compares samples_per_rollout between:
#   1. Baseline (standard tokenization)
#   2. Per-message tokenization (AR token caching)
#
# Key metric: samples_per_rollout
#   - =1 means extension property held (all turns merged into one sample)
#   - >1 means tokenization mismatch broke the extension property
#
# Usage:
#   bash scripts/benchmark_per_message_tokenization.sh

set -euo pipefail

CONFIG="configs/benchmark/per_message_tokenization.toml"

echo "============================================"
echo "Run 1: BASELINE (no per-message tokenization)"
echo "============================================"
uv run rl @ "$CONFIG" \
    --wandb.name baseline \
    --no-inference.per-message-tokenization

echo ""
echo "============================================"
echo "Run 2: PER-MESSAGE TOKENIZATION"
echo "============================================"
uv run rl @ "$CONFIG" \
    --wandb.name per-message-tokenization \
    --inference.per-message-tokenization
