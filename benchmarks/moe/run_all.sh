#!/usr/bin/env bash
set -euo pipefail

echo "[benchmarks] prime_moe_scatter"
uv run python benchmarks/moe/benchmark_moe_scatter.py

echo
echo "[benchmarks] prime_moe_gather"
uv run python benchmarks/moe/benchmark_moe_gather.py

echo
echo "[benchmarks] prime_moe_routed_experts"
uv run python benchmarks/moe/benchmark_moe_routed_experts.py
