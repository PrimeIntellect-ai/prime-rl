#!/bin/bash
# Runs the mock matrix serially on the current node; skips runs whose metrics already have MAX_STEPS steps.
source ~/.localrc
G=/home/garrett/github/PrimeIntellect-ai
OUT=/home/garrett/prl_output_dir/mock
LOG=/home/garrett/tmp/rl-data-workers/matrix.log
MAX_STEPS=12
runs=(
  "base 1024 - a" "branch 1024 0 a" "branch 1024 2 a" "branch 1024 4 a" "3907 1024 - a" "base 1024 - b"
  "base 1440 - a" "branch 1440 0 a" "branch 1440 2 a" "branch 1440 4 a" "3907 1440 - a"
  "base 512 - a" "branch 512 0 a" "branch 512 2 a" "branch 512 4 a" "3907 512 - a"
)
for r in "${runs[@]}"; do
  read arm px nw rep <<< "$r"
  name="mock-$arm-px$px"; [ "$nw" != "-" ] && name="$name-nw$nw"; name="$name-$rep"
  done_steps=$(grep -c '"time/step"' $OUT/$name/monitors/file/metrics.jsonl 2>/dev/null || echo 0)
  if [ "$done_steps" -ge "$MAX_STEPS" ]; then echo "$(date +%T) skip $name (complete)" >> $LOG; continue; fi
  rm -rf $OUT/$name
  extra=""; [ "$nw" != "-" ] && extra="--data.num-workers $nw"
  echo "$(date +%T) start $name commit=$(git -C $G/prime-rl-arm-$arm rev-parse --short HEAD)" >> $LOG
  cd $G/prime-rl-arm-$arm
  CUDA_VISIBLE_DEVICES=0,1,2,3 PRL_MOCK_MM_IMAGE_PX=$px timeout 1500 uv run --no-sync torchrun --standalone --nproc-per-node 4 \
    -m prime_rl.trainer.rl.train @ /home/garrett/tmp/rl-data-workers/mock-trainer.toml --no-model.ac-offloading \
    --max-steps $MAX_STEPS --output-dir $OUT/$name $extra > $OUT/$name.out 2>&1
  echo "$(date +%T) end $name exit=$?" >> $LOG
done
echo "$(date +%T) matrix finished" >> $LOG
