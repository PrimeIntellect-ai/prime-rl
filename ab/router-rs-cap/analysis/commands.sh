#!/usr/bin/env bash
# Every skill-script invocation that produced a result for the router-rs-cap investigation, in order.
set -euo pipefail
S=~/.claude/skills/profiling/scripts
T=~/tmp/router-rs-cap/traces
P=~/tmp/profiling/router-rs-cap
LABELS=(--comm-label 'L0:layers\.0\)$' --comm-label 'E:\) \[pg=2\]$' --comm-label 'D:\) \[pg=16\]$'
  --comm-label 'R:mlp\.router\)$' --comm-label 'D+E:layers\.\d+\)$')

stage() { [[ " ${STAGES:-waits figures} " == *" $1 "* ]]; }

# PREFIX selects an ablation scenario, e.g. PREFIX=s1k-noac- (empty: the 16k, full-AC baseline).
PREFIX=${PREFIX:-}
stage waits && for layout in ${LAYOUTS:-ep8 fsdp16}; do
  for rank in 0 8; do
    traces=()
    for v in fp32-cap1 fp32-cap2 fp32-cap3 bf16-cap1 bf16-cap2; do
      traces+=(--trace "$v=$T/$PREFIX$layout-$v-trace/trace_$rank.json.gz")
    done
    uv run --script $S/comm_waits.py "${traces[@]}" "${LABELS[@]}" --json $P/derived/waits-$PREFIX$layout-rank$rank.json \
      > $P/derived/waits-$PREFIX$layout-rank$rank.txt
  done
done

# Per-layer stream timelines, layer 12 of the last traced step, rank 0, aligned at that layer's expert (ep8) or
# whole-block (fsdp16) reduce-scatter launch.
FIG=${FIG:-$HOME/github/PrimeIntellect-ai/prime-rl-perfrouter-rs-buffer-cap/figures}
COMMON=(--no-delta --comm-kernel ReduceScatter --wait-annotation '^FSDP::post_backward_rs_wait'
  --comm-label 'L0:layers\.0\)$:#999999' --comm-label 'E:\) \[pg=2\]$:#d55e00' --comm-label 'D:\) \[pg=16\]$:#56b4e9'
  --comm-label 'R:mlp\.router\)$:#f0e442' --comm-label 'D+E:layers\.\d+\)$:#d55e00'
  --category 'EP all-to-all:SendRecv:#e69f00' --category 'other NCCL:^nccl:#555555'
  --segment 'recompute:^FSDP::pre_backward \(model\.layers\.\d+\)( \[pg=2\])?$:#cc79a7'
  --segment 'MLP backward:^autograd::engine::evaluate_function: MmBackward0:#0072b2:end'
  --segment 'attention backward:evaluate_function: RMSNormFunctionBackward:#009e73')
declare -A OWNER=([ep8]="FSDP::post_backward_reduce (model.layers.12) [pg=2]" [fsdp16]="FSDP::post_backward_reduce (model.layers.12)")
declare -A ALIGN=([ep8]="layer 12's expert reduce-scatter (E)" [fsdp16]="layer 12's block reduce-scatter (D+E)")
declare -A SETUP=([ep8]="ep=8, dp_shard=16" [fsdp16]="ep=1, dp_shard=16")
stage figures && for layout in ${LAYOUTS:-ep8 fsdp16}; do
  traces=()
  for v in fp32-cap1 fp32-cap2 fp32-cap3 bf16-cap1; do
    traces+=(--trace "$v=$T/$layout-$v-trace/trace_0.json.gz")
  done
  uv run --script $S/timeline_figure.py --out $FIG/timeline-$layout.png --window -10 160 "${COMMON[@]}" \
    --anchor-comm-owner "${OWNER[$layout]}" --anchor-label "the GPU start of ${ALIGN[$layout]}, rank 0" "${traces[@]}" \
    --title "GLM-4.5-Air 24 layers, 2x8 H200, ${SETUP[$layout]}, 16k tokens/GPU, full AC: backward of layer 11 after layer 12's reduce-scatter" \
    > $P/derived/timeline-$layout.txt
done

# Diagnose the fsdp16 fp32 compute-stream idle gap inside layer 11's recompute: what the host was inside.
stage gaps && uv run --script $S/timeline_figure.py --out $P/figures/gaps-fsdp16-fp32-cap1.png --window -10 160 \
  "${COMMON[@]}" --anchor-comm-owner "${OWNER[fsdp16]}" --annotate-gaps 2 --gap-scope main \
  --trace "fp32-cap1=$T/fsdp16-fp32-cap1-trace/trace_0.json.gz" --trace "bf16-cap1=$T/fsdp16-bf16-cap1-trace/trace_0.json.gz" \
  > $P/derived/gaps-fsdp16.txt

# Host-blocking CUDA calls in the last full traced step, fsdp16 rank 0.
stage syncs && for v in fp32-cap1 fp32-cap2 bf16-cap1; do
  python3 $S/find_syncs.py $T/fsdp16-$v-trace/trace_0.json.gz > $P/derived/syncs-fsdp16-$v.txt
done

# fsdp16: reduce-scatters and all-gathers together, to see what the fp32 compute stream waits on mid-recompute.
stage allgather && uv run --script $S/timeline_figure.py --out $FIG/timeline-fsdp16-allgather.png --window -10 160 \
  --no-delta --comm-kernel 'ReduceScatter|AllGather' --launch-annotation '^FSDP::(post_backward_reduce|all_gather) ' \
  --comm-label 'AG router:^FSDP::all_gather .*mlp\.router\)$:#f0e442' --comm-label 'AG block:^FSDP::all_gather :#56b4e9' \
  --comm-label 'RS router:mlp\.router\)$:#e69f00' --comm-label 'RS D+E:^FSDP::post_backward_reduce:#d55e00' \
  --segment 'recompute:^FSDP::pre_backward \(model\.layers\.\d+\)( \[pg=2\])?$:#cc79a7' \
  --segment 'MLP backward:^autograd::engine::evaluate_function: MmBackward0:#0072b2:end' \
  --segment 'attention backward:evaluate_function: RMSNormFunctionBackward:#009e73' \
  --annotate-gaps 2 --gap-scope main --anchor-comm-owner "${OWNER[fsdp16]}" \
  --anchor-label "the GPU start of layer 12's block reduce-scatter (D+E), rank 0" \
  --trace "fp32 router, cap 1=$T/fsdp16-fp32-cap1-trace/trace_0.json.gz" \
  --trace "fp32 router, cap 2=$T/fsdp16-fp32-cap2-trace/trace_0.json.gz" \
  --trace "bf16 router, cap 1=$T/fsdp16-bf16-cap1-trace/trace_0.json.gz" \
  --title "GLM-4.5-Air 24 layers, 2x8 H200, ep=1, dp_shard=16: reduce-scatters and all-gathers around layer 11's backward" \
  > $P/derived/timeline-fsdp16-allgather.txt

# fsdp16: exact timing of all-gathers and reduce-scatters for layers 10 to 12, relative to layer 12's D+E RS start.
stage percomm && for v in fp32-cap1 bf16-cap1; do
  uv run --script $S/comm_waits.py --trace "$v=$T/fsdp16-$v-trace/trace_0.json.gz" \
    --comm-kernel 'ReduceScatter|AllGather' --launch-annotation '^FSDP::(post_backward_reduce|all_gather) ' \
    --per-comm 'layers\.1[0-2][)._]' --list-from "${OWNER[fsdp16]}" > $P/derived/percomm-fsdp16-$v.txt
done

# Whole last step, main-stream idle gaps >= 2 ms, per fsdp16/ep8 arm.
stage stepgaps && for layout in fsdp16 ep8; do
  for v in fp32-cap1 fp32-cap2 bf16-cap1; do
    uv run --script $S/timeline_figure.py --out $P/figures/stepgaps-$layout-$v.png --window -10 5000 --no-delta \
      --annotate-gaps 2 --gap-scope main --trace "$v=$T/$layout-$v-trace/trace_0.json.gz" > $P/derived/stepgaps-$layout-$v.txt
  done
done

# Summary CSV (untraced timing + traced stall/overlap), bar charts, and per-step time lines.
stage bars && {
  (cd ~/github/PrimeIntellect-ai/prime-rl-perfrouter-rs-buffer-cap && uv run --no-sync python ab/router-rs-cap/compare.py --csv $P/derived/summary.csv > $P/derived/compare.txt)
  uv run --script $S/run_metrics.py --csv $P/derived/summary.csv --out $FIG/throughput-memory.png \
    --noise fp32-cap2,fp32-cap2-r2,fp32-main \
    --bar "s_step:median s/step, steps 5-20, untraced (lower is better)" \
    --bar "tok_s_gpu:median tokens/s/GPU, untraced (higher is better)" \
    --bar "peak_gib:peak reserved memory GiB, rank 0 (lower is better)" \
    --title "GLM-4.5-Air 24 layers, 2x8 H200, 16k tokens/GPU: throughput and memory per reduce-scatter buffer cap"
  uv run --script $S/run_metrics.py --csv $P/derived/summary.csv --out $FIG/stall-overlap.png \
    --bar "stall_ms:GPU stall on a popped reduce-scatter, ms/step (lower is better)" \
    --bar "stall_plus_tail_ms:stall + comm tail after last compute kernel, ms/step (lower is better)" \
    --bar "overlap_pct:expert (ep8) or block (fsdp16) RS overlap with compute, % (higher is better)" \
    --title "Traced runs, last backward, mean of rank 0 and rank 8"
  uv run --script $S/run_metrics.py --csv $P/derived/summary.csv --out $FIG/step-time.png --lines time/step \
    --lines loss/mean --steady 5 20 --title "Untraced runs: per-step time and loss (gray: steady-state steps 5-20)"
}

# Ablations: attribute stalls to reduce-scatters and all-gathers together (they share the 16-rank communicator).
stage waits-allcomm && for layout in ${LAYOUTS:-ep8 fsdp16}; do
  for rank in 0 8; do
    traces=()
    for v in fp32-cap1 fp32-cap2 fp32-cap3 bf16-cap1 bf16-cap2; do
      traces+=(--trace "$v=$T/$PREFIX$layout-$v-trace/trace_$rank.json.gz")
    done
    uv run --script $S/comm_waits.py "${traces[@]}" --comm-kernel 'ReduceScatter|AllGather' \
      --launch-annotation '^FSDP::(post_backward_reduce|all_gather) ' \
      --comm-label 'AG router:^FSDP::all_gather .*mlp\.router\)$' --comm-label 'AG:^FSDP::all_gather ' "${LABELS[@]}" \
      --json $P/derived/waits-allcomm-$PREFIX$layout-rank$rank.json > $P/derived/waits-allcomm-$PREFIX$layout-rank$rank.txt
  done
done

# Ablation timelines (no activation checkpointing, so no recompute phase), layer 12, rank 0. On fsdp16 the comm row
# also shows all-gathers, which share the 16-rank communicator with the reduce-scatters; on ep8 they run on another
# stream and communicator, so only reduce-scatters are drawn.
declare -A COMM_KERNEL=([ep8]='ReduceScatter' [fsdp16]='ReduceScatter|AllGather')
declare -A COMM_ROW=([ep8]='reduce-scatter' [fsdp16]='RS and AG')
stage ablation-figures && for layout in ${LAYOUTS:-ep8 fsdp16}; do
  traces=()
  for v in fp32-cap1 fp32-cap2 fp32-cap3 bf16-cap1 bf16-cap2; do
    traces+=(--trace "$v=$T/$PREFIX$layout-$v-trace/trace_0.json.gz")
  done
  uv run --script $S/timeline_figure.py --out $FIG/timeline-$PREFIX$layout.png --window ${WINDOW:--2 60} --no-delta \
    --main-exclude '^nccl' --comm-kernel "${COMM_KERNEL[$layout]}" --launch-annotation '^FSDP::(post_backward_reduce|all_gather) ' \
    --wait-annotation '^FSDP::post_backward_rs_wait' --comm-row-label "${COMM_ROW[$layout]}" --comm-legend-suffix '' \
    --comm-label 'AG router:^FSDP::all_gather .*mlp\.router\)$:#cc79a7' --comm-label 'AG:^FSDP::all_gather :#999999' \
    --comm-label 'RS E:\) \[pg=2\]$:#d55e00' --comm-label 'RS D:\) \[pg=16\]$:#56b4e9' \
    --comm-label 'RS R:mlp\.router\)$:#f0e442' --comm-label 'RS D+E:layers\.\d+\)$:#d55e00' \
    --category 'EP all-to-all:SendRecv:#e69f00' --category 'other NCCL:^nccl:#555555' --category 'compute:.:#0072b2' \
    --anchor-comm-owner "${OWNER[$layout]}" --anchor-label "the GPU start of ${ALIGN[$layout]}, rank 0" "${traces[@]}" \
    --title "${TITLE_PREFIX:-} ${SETUP[$layout]}: backward after layer 12's reduce-scatter" \
    > $P/derived/timeline-$PREFIX$layout.txt
done
