#!/bin/bash
# Serial fullgraph A/B series on job 2802; stops launching after 13:25 UTC.
export JOB=2802
D=benchmarks/scripts/dist_moe
L=~/tmp/dist-moe-sft/launch.sh
T=$HOME/tmp/profiling/dist-moe-sft-ab/traces
base() { case $1 in qwen) echo qwen3-30b-a3b-sft.toml;; glm) echo glm-4.5-air-sft.toml;; esac; }
disp() { case $1 in torch) echo dispatch-torch.toml;; distmoe) echo dispatch-dist-moe.toml;; esac; }
go() {
  if [ "$(date -u +%H%M)" -ge 1325 ]; then echo "skip $1 (deadline)"; return; fi
  echo "start $1 $(date -u +%T)"
  $L "$@"
  echo "done $1 $(date -u +%T) $(tail -1 ~/tmp/dist-moe-sft/$1.out)"
}
for r in "" -r2; do for m in qwen glm; do for a in torch distmoe; do
  go fg-$m-$a$r TORCH_LOGS=recompiles -- @ $D/$(base $m) @ $D/$(disp $a) @ $D/compile-fullgraph.toml
done; done; done
for m in qwen glm; do for a in torch distmoe; do
  go tr-fg-$m-$a -- @ $D/$(base $m) @ $D/$(disp $a) @ $D/compile-fullgraph.toml --max-steps 4 --trace-path $T/tr-fg-$m-$a
done; done
