#!/bin/bash
# Usage: run.sh MODEL ARM SHA [extra trainer args]
# MODEL is a configs/*.toml name, ARM a label (base or fix), SHA the commit to run.
# Env: PYTHON (interpreter with prime-rl's dependencies), MODELS_DIR, DATA_DIR, OUT_DIR (default ./ab-runs),
# CACHE_ROOT (compile caches, default OUT_DIR/cache; use node-local disk when OUT_DIR is on a network filesystem),
# HOOK_DIR (directory holding the sitecustomize.py hook, default peaklog).
set -euo pipefail
MODEL=$1
ARM=$2
SHA=$(git -C "$(dirname "$0")" rev-parse "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
OUT_DIR=${OUT_DIR:-$PWD/ab-runs}
SRC=$OUT_DIR/src/$SHA
RUN=$OUT_DIR/runs/$MODEL-$ARM
CACHE=${CACHE_ROOT:-$OUT_DIR/cache}/$MODEL-$ARM
[ ! -e "$RUN/rc.txt" ] || { echo "$RUN already holds a run" >&2; exit 1; }
if [ ! -d "$SRC/src/prime_rl" ]; then
  mkdir -p "$SRC"
  git -C "$REPO" archive "$SHA" | tar -x -C "$SRC"
fi
rm -rf -- "${CACHE:?}"
mkdir -p "$RUN/logs" "$CACHE"
for f in common "$MODEL"; do
  sed -e "s#MODELS_DIR#$MODELS_DIR#" -e "s#DATA_DIR#$DATA_DIR#" "$HERE/configs/$f.toml" > "$RUN/$f.toml"
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE=$CACHE/datasets
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export TORCH_LOGS=recompiles,graph_breaks TORCH_TRACE=$RUN/torch_trace PEAK_LOG_DIR=$RUN/peaks
export PYTHONPATH=${HOOK_DIR:-$HERE/peaklog}:$SRC/src:$SRC/packages/prime-rl-configs/src
echo "$SHA" > "$RUN/sha.txt"
date +%s.%N > "$RUN/start_epoch.txt"
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 --log-dir="$RUN/torchrun" \
  --local-ranks-filter=0 --redirect=3 --tee=3 \
  -m prime_rl.trainer.sft.train @ "$RUN/common.toml" @ "$RUN/$MODEL.toml" --output-dir "$RUN/out" "${@:4}" \
  > "$RUN/logs/trainer.log" 2>&1
RC=$?
set -e
date +%s.%N > "$RUN/end_epoch.txt"
echo "$RC" > "$RUN/rc.txt"
echo "rc=$RC run=$RUN"
