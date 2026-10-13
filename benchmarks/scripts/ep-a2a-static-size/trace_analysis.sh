#!/bin/bash
# Commands that produced the ep-a2a-static-size results, in order.
S=~/.claude/skills/profiling/scripts
I=~/tmp/profiling/ep-a2a-static-size
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
