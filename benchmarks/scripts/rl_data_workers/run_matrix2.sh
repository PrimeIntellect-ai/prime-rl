#!/bin/bash
# Second mock matrix (1024 px): serial on the current node, skips complete runs.
# Each entry: arm num_workers(- for none) steps offload(0/1) sample_rss(0/1) rep
source ~/.localrc
G=/home/garrett/github/PrimeIntellect-ai
OUT=/home/garrett/prl_output_dir/mock2
LOG=/home/garrett/tmp/rl-data-workers/matrix2.log
mkdir -p $OUT
runs=(
  "base - 12 0 0 a" "3907 - 12 0 0 a" "thread 1 12 0 0 a" "thread 2 12 0 0 a" "3907-nopin - 12 0 0 a"
  "base - 12 1 0 a" "thread 1 12 1 0 a" "thread 1 30 0 1 a" "base - 12 0 0 b"
)
for r in "${runs[@]}"; do
  read arm nw steps offload rss rep <<< "$r"
  name="mock2-$arm"; [ "$nw" != "-" ] && name="$name-nw$nw"; [ "$offload" = 1 ] && name="$name-offload"; [ "$steps" != 12 ] && name="$name-s$steps"; name="$name-$rep"
  done_steps=$(grep -c '"time/step"' $OUT/$name/monitors/file/metrics.jsonl 2>/dev/null || echo 0)
  if [ "$done_steps" -ge "$steps" ]; then echo "$(date +%T) skip $name" >> $LOG; continue; fi
  case $arm in thread|3907-nopin) until [ -e $G/prime-rl-arm-$arm/.sync-done ]; do sleep 20; done
    [ "$(cat $G/prime-rl-arm-$arm/.sync-done)" = 0 ] || { echo "$(date +%T) sync failed for $arm, skip $name" >> $LOG; continue; } ;; esac
  rm -rf $OUT/$name
  extra=""; [ "$nw" != "-" ] && extra="--data.num-workers $nw"; [ "$offload" = 1 ] && extra="$extra --model.optim-cpu-offload"
  echo "$(date +%T) start $name commit=$(git -C $G/prime-rl-arm-$arm rev-parse --short HEAD)" >> $LOG
  if [ "$rss" = 1 ]; then
    ( sleep 30; while pgrep -u $USER -f "rl.train @" >/dev/null; do
        for p in $(pgrep -u $USER -f "prime_rl.trainer.rl.train @"); do
          echo "$(date +%s) $p $(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null) $(grep -c '"time/step"' $OUT/$name/monitors/file/metrics.jsonl 2>/dev/null)"
        done; sleep 10; done ) > $OUT/$name.rss 2>&1 &
  fi
  cd $G/prime-rl-arm-$arm
  CUDA_VISIBLE_DEVICES=0,1,2,3 PRL_MOCK_MM_IMAGE_PX=1024 timeout 2400 uv run --no-sync torchrun --standalone --nproc-per-node 4 \
    -m prime_rl.trainer.rl.train @ /home/garrett/tmp/rl-data-workers/mock-trainer.toml --no-model.ac-offloading \
    --max-steps $steps --output-dir $OUT/$name $extra > $OUT/$name.out 2>&1
  echo "$(date +%T) end $name exit=$?" >> $LOG
  wait
done
echo "$(date +%T) matrix finished" >> $LOG
