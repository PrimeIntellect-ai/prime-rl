#!/bin/bash
# GPU unit tests for both PR worktrees and main, run inside the node allocation after the A/B series.
set -u
W=~/github/PrimeIntellect-ai
while pgrep -f series2.sh > /dev/null; do sleep 10; done
for tree in prime-rl-fix-ep-a2a-static-size prime-rl-fix-cp-attn-params-as-args prime-rl-main; do
  echo "=== $tree"
  srun --jobid=${JOB:-2816} --overlap bash -lc "cd $W/$tree && git rev-parse --short HEAD && \
    uv run --no-sync pytest -q tests/unit/train/models/test_moe.py tests/unit/train/models/test_checkpointing.py 'tests/unit/train/models/qwen3_5/test_model.py::test_context_parallel_setup_chain_text_and_vlm' 2>&1 | tail -15 && \
    uv run --no-sync torchrun --nproc-per-node=2 -m pytest -q tests/unit/train/models/test_gpt_oss.py -k context_parallel 2>&1 | tail -8" < /dev/null
done
