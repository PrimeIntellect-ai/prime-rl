# Shared environment for every arm. Usage: source env.sh ARM NAME (ARM: bf16, before, after)
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup
C=$WT/benchmarks/scripts/fp8_cleanup_h200_table
MAIN_SRC=$HOME/tmp/profiling/fp8-cleanup-h200/arms/main-88faa6dc2/src
export PRL_OUTPUT_DIR=/home/garrett/prl_output_dir HF_HOME=/home/garrett/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH=""; [ "$1" = before ] && export PYTHONPATH=$MAIN_SRC
export TRITON_CACHE_DIR=/tmp/$USER/$2/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/$2/inductor
export DG_JIT_CACHE_DIR=/tmp/$USER/$2/dg TILELANG_CACHE_DIR=/tmp/$USER/$2/tilelang
OVERLAY=(@ $C/sft_fp8.toml); [ "$1" = bf16 ] && OVERLAY=()
