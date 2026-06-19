#!/usr/bin/env bash
# Pin every torch.compile / Triton / Inductor / vLLM compile cache to each node's
# local tmpfs. These caches hold mmap'd, executable kernel artifacts; N concurrent
# writers on a shared Lustre/NFS path corrupt them (observed "Stale file handle"
# autotune races, and a suspected source of cross-rank engine faults). /tmp is a
# per-node tmpfs on Isambard GH200; scope the dirs by job + host so concurrent
# jobs (and the shared, never-wiped /tmp) never collide.
#
# Single source of truth: every launch path (sbatch RL/SFT/inference, in-alloc
# gpu_layout, offline-eval) sources this so no path can leak compile artifacts
# onto a shared filesystem. HF model caches (HF_HOME/HF_HUB_CACHE) are NOT touched
# here — they are read-mostly shared weights and stay wherever the caller set them.
#
# Idempotent and side-effect-safe to source from `set -euo pipefail` scripts.

_cache_scope="${SLURM_JOB_ID:-${USER:-prime}}_$(hostname -s)"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_inductor_${_cache_scope}}"
export INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${_cache_scope}}"
# vLLM keeps its torch.compile cache under $VLLM_CACHE_ROOT/torch_compile_cache;
# this is the only knob vLLM 0.22 reads for it (VLLM_TORCH_COMPILE_CACHE_DIR is
# not a vLLM env var). Default is ~/.cache/vllm (shared home) — redirect to /tmp.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache_${_cache_scope}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${_cache_scope}}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

unset _cache_scope
