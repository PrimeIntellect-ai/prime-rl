#!/bin/bash
# Usage: sweep.sh <slurm_jobid> <label> [extra bench.py args...]
# Runs bench.py over the layer/seq_len grid on one allocated node, one config per GPU, each with
# an empty tilelang cache so every width compiles cold.
set -u
JOBID=$1
LABEL=$2
shift 2
EXTRA="$*"
WT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$HOME/tmp/dsv4-attn-dyn/logs
mkdir -p "$OUT"
CONFIGS=("hca 65536" "hca 131072" "hca 262144" "csa 65536" "csa 131072" "sliding 131072")
for i in "${!CONFIGS[@]}"; do
  read -r LAYER SEQ <<< "${CONFIGS[$i]}"
  CACHE=$HOME/tmp/dsv4-attn-dyn/tlcache-$LABEL-$i
  rm -rf "$CACHE"
  srun --jobid="$JOBID" --overlap -N1 -n1 bash -c \
    "cd $WT && CUDA_VISIBLE_DEVICES=$i TILELANG_CACHE_DIR=$CACHE uv run --no-sync python benchmarks/dsv4_attn_dyn/bench.py $LABEL $LAYER $SEQ $EXTRA" \
    > "$OUT/$LABEL-$LAYER-$SEQ.log" 2>&1 &
done
wait
grep -h SUMMARY "$OUT/$LABEL"-*.log
