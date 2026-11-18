#!/bin/bash
# Commit (signed) and push exp/dsv4-pd-stress: the merge of feat/dsv4-nvfp4-inference, then the experiment files.
set -euo pipefail
for s in /tmp/ssh-*/agent.*; do SSH_AUTH_SOCK=$s timeout 10 ssh-add -l >/dev/null 2>&1 && export SSH_AUTH_SOCK=$s && break; done
timeout 10 ssh-add -l >/dev/null || { echo "no live SSH agent: reconnect, then rerun"; exit 1; }
cd /home/phi/lib/prime-rl-pd-exp
cp /home/phi/exp/dsv4-pd/NOTES.md configs/experiments/dsv4-pd/NOTES.md
if [ -f "$(git rev-parse --git-dir)/MERGE_HEAD" ]; then
    git reset -q -- configs/experiments/dsv4-pd
    git commit -q -m "Merge feat/dsv4-nvfp4-inference (PR #4564) for P/D + NVFP4 runs

Claude-Session: https://claude.ai/code/session_01PEUKiL3DnJrVbPAXJgvY67"
fi
git add configs/experiments/dsv4-pd
git diff --cached --quiet || git commit -q -m "exp(dsv4-pd): P/D stress test, tuning and P/D + NVFP4 for DeepSeek-V4-Flash

Configs, staircase driver, level loops, analyzers, report builder, router template and notes of the DeepSWE + RLM
stress test behind PR #4565 (equal-compute agg vs P/D vs P/D + Mooncake), the tuning rounds (scheduler step size,
3P:1D vs 4x aggregated) and the P/D 3P:1D + NVFP4 routed-experts run.

Claude-Session: https://claude.ai/code/session_01PEUKiL3DnJrVbPAXJgvY67"
git log --oneline --format="%h %G? %s" -3
timeout 120 git push -u origin exp/dsv4-pd-stress 2>&1 | tail -2
