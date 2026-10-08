#!/bin/bash
# 30-step RSS runs: thread num_workers 1 and base, sampling rank RSS via torchrun's children.
source ~/.localrc
G=/home/garrett/github/PrimeIntellect-ai
OUT=/home/garrett/prl_output_dir/mock2
LOG=/home/garrett/tmp/rl-data-workers/matrix2.log
for r in "thread 1" "base -"; do
  read arm nw <<< "$r"
  name="mock2-$arm"; [ "$nw" != "-" ] && name="$name-nw$nw"; name="$name-s30-rss"
  rm -rf $OUT/$name
  extra=""; [ "$nw" != "-" ] && extra="--data.num-workers $nw"
  echo "$(date +%T) start $name commit=$(git -C $G/prime-rl-arm-$arm rev-parse --short HEAD)" >> $LOG
  cd $G/prime-rl-arm-$arm
  CUDA_VISIBLE_DEVICES=0,1,2,3 PRL_MOCK_MM_IMAGE_PX=1024 timeout 2400 uv run --no-sync torchrun --standalone --nproc-per-node 4 \
    -m prime_rl.trainer.rl.train @ /home/garrett/tmp/rl-data-workers/mock-trainer.toml --no-model.ac-offloading \
    --max-steps 30 --output-dir $OUT/$name $extra > $OUT/$name.out 2>&1 &
  launcher=$!
  sleep 20
  agent=$(pgrep -u $USER -f "bin/torchrun --standalone" | head -1)
  while kill -0 $launcher 2>/dev/null; do
    for p in $(pgrep -P $agent 2>/dev/null); do
      echo "$(date +%s) $p $(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null) $(grep -c 'SUCCESS.*Step' $OUT/$name.out)"
    done
    sleep 10
  done > $OUT/$name.rss
  wait $launcher
  echo "$(date +%T) end $name exit=$?" >> $LOG
done
echo "$(date +%T) rss runs finished" >> $LOG
