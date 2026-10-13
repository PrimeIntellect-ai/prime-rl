#!/bin/bash
# Commands that produced the ep-a2a-static-size results, in order.
S=~/.claude/skills/profiling/scripts
I=~/tmp/profiling/ep-a2a-static-size
STAGE=${1:-all}

if [ "$STAGE" = all ] || [ "$STAGE" = mini ]; then
CAT=(--category 'inductor:^triton_' --category 'grouped_gemm:grouped|cutlass|sm90|gemm|Gemm' --category 'nccl:nccl'
  --category 'eager elementwise:elementwise|vectorized|reduce|index|scatter|gather|copy|fill|Memcpy|Memset')
for a in main-off br-off; do
  uv run --script $S/top_kernels.py $I/traces/$a/trace_0.json.gz --annotation-index -4 --annotation-span 4 --top 15 \
    --decimals 3 "${CAT[@]}" --csv $I/derived/top_kernels-$a.csv > $I/derived/top_kernels-$a.md
done
uv run --script $S/timeline_figure.py --out $I/figures/moe-layer2-forward.png --window -0.5 25 \
  --anchor-annotation "FSDP::all_gather_copy_out (model.layers.2) [pg=2]" --anchor-index -2 \
  --end-annotation "FSDP::all_gather_copy_out (model.layers.3) [pg=2]" \
  --region-label "layer 2 forward" --anchor-label "the start of layer 2's forward, last micro-batch, rank 0" \
  --trace "Before: main (7c9d474), MoE blocks eager=$I/traces/main-off/trace_0.json.gz" \
  --trace "After: this PR (4464456), MoE blocks compiled=$I/traces/br-off/trace_0.json.gz" \
  --category "Inductor (fused Triton):^triton_:#009e73" --category "grouped GEMM:GemmUniversal|grouped:#0072b2" \
  --category "flash attention:flash:#cc79a7" --category "NCCL:nccl:#56b4e9" \
  --category "eager elementwise / copy / index:elementwise|copy|Functor|index|scatter|gather|reduce|sort|Cat:#e69f00" \
  --title "mini-glm-moe SFT, ep=2 on 2 x H200, bf16 router, compile on with fullgraph off, main GPU stream" \
  > $I/derived/timeline-moe-layer2-forward.txt
fi

QCAT=(--category 'inductor:^triton_' --category 'grouped GEMM:GemmUniversal|grouped' --category 'flash attention:flash'
  --category 'dense GEMM:nvjet|sm90_xmma|gemm' --category 'eager elementwise / copy / index:elementwise|vectorized|reduce|index|scatter|gather|copy|fill|Cat|sort')
if [ "$STAGE" = all ] || [ "$STAGE" = qwen ]; then
  for a in main-off br-off; do
    uv run --script $S/top_kernels.py $I/traces/qwen3-30b-$a/trace_0.json.gz --annotation-index -2 --annotation-span 2 \
      --top 15 --decimals 3 "${QCAT[@]}" --csv $I/derived/qwen-top_kernels-$a.csv > $I/derived/qwen-top_kernels-$a.md
  done
  uv run --script $S/kernel_diff.py $I/derived/qwen-top_kernels-main-off.csv $I/derived/qwen-top_kernels-br-off.csv \
    --before-label main --after-label branch --top 25 --decimals 3 "${QCAT[@]}" > $I/derived/qwen-kernel_diff.md
  uv run --script $S/step_accounting.py --trace "main=$I/traces/qwen3-30b-main-off/trace_*.json.gz" \
    --trace "branch=$I/traces/qwen3-30b-br-off/trace_*.json.gz" --step-boundary FusedAdamMathFunctor "${QCAT[@]}" \
    --untraced main=5.965 --untraced branch=4.484 --per-rank > $I/derived/qwen-step_accounting.md
fi
if [ "$STAGE" = all ] || [ "$STAGE" = qwen-figure ]; then
  uv run --script $S/timeline_figure.py --out $I/figures/qwen-layer2-forward.png --window -1 60 \
    --anchor-annotation "FSDP::all_gather_copy_out (model.layers.2) [pg=8]" --anchor-index -2 \
    --end-annotation "FSDP::all_gather_copy_out (model.layers.3) [pg=8]" \
    --region-label "layer 2 forward" --anchor-label "the start of layer 2's forward, last micro-batch, rank 0" \
    --trace "Before: main (7c9d474), 5.97 s/step untraced=$I/traces/qwen3-30b-main-off/trace_0.json.gz" \
    --trace "After: this PR (4464456), 4.48 s/step untraced=$I/traces/qwen3-30b-br-off/trace_0.json.gz" \
    --category "Inductor (fused Triton):^triton_:#009e73" --category "grouped GEMM:GemmUniversal|grouped:#0072b2" \
    --category "flash attention:flash:#cc79a7" --category "NCCL:nccl:#56b4e9" \
    --category "eager elementwise / copy / index:elementwise|copy|Functor|index|scatter|gather|reduce|sort|Cat:#e69f00" \
    --title "Qwen3-30B-A3B SFT, 8 x H200, ep=8, 32k tokens/GPU, bf16 router, compile on, fullgraph off" \
    > $I/derived/qwen-timeline-layer2-forward.txt
fi
if [ "$STAGE" = all ] || [ "$STAGE" = glm ]; then
  uv run --script $S/step_accounting.py --trace "main=$I/traces/glm-air-8l-main-off/trace_*.json.gz" \
    --trace "branch=$I/traces/glm-air-8l-br-off/trace_*.json.gz" --step-boundary FusedAdamMathFunctor "${QCAT[@]}" \
    --untraced main=2.957 --untraced branch=2.544 --steps=-1: --per-rank > $I/derived/glm-step_accounting.md
fi
