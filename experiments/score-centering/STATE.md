# Experiment state

User authorized autonomous implementation, launches, monitoring, fixes, and analysis.
Latest main merged: a1822f7a0a8f0e4abcb4b1d07a7f6bba6c101d54.
Do not merge or open a PR unless asked. No subagents are authorized.

## Current work

- Full 400-update pair started at 2026-09-19 06:29 UTC: IPO861 (`ipo-seed42`) and SC860 (`sc-seed42`), four nodes each. Both use FP8 KV, INT8 experts, FP8 dense weights, and uniform task draws with seed 42.
- Launch source: ef2cfad80; latest main a1822f7a0 is merged. Latest parser tests: 155 passed.

- User reported VM service recovery and resumed the experiment at 2026-09-19 05:24 UTC.
- Recovery uv eval completed: Lego 2/2 solved; TB2 1/2 solved; no terminal errors. Both executed tools and scored. One TB2 rollout reached max_turns; hosted model transient429 retried.
- Main was refreshed again before the full launch: a1822f7a0 (expert-parallel load filtering; inactive for this TP-only setup).
- Concurrent smoke4 jobs: IPO849, SC850, four nodes each (eight total). Both completed three finite updates and exited successfully. Logs under outputs/score-centering/smoke4-{ipo,sc}.
- Both arms report occasional NaN serving responses before first updates. These are excluded provider failures, not evidence of training collapse.
- Diagnostic pair smoke5: SC856 and IPO857, four nodes each. Both completed three finite updates with FP8 KV; no observed NaN serving responses. Both jobs exited successfully (0:0) by 06:28 UTC. Main uses FP8 KV and uniform task sampling.
- Previous outage blocker audit resets on this user-requested resumption.

### Implementation and prior attempts

- Primary treatment: standalone score centering, matching the official PyTorch formula.
- Optional IPO-plus-centering ablation remains implemented with its nonconstant tail mask.
- Smoke2 used the IPO ablation; subsequent pairs use centered.toml (standalone).
- Implemented top-k evidence in renderer/verifiers submodules and PRL batch transport.
- Mathematical checks and transport checks pass; scripts are in this directory.
- Existing CUDA loss tests pass (12/12), job 798 completed.
- Smoke2 jobs 802/803 stopped before updates after quantized serving validation.
- Smoke3 jobs 804/805 stopped: startup weight re-quantization passed; all TMax image names were stale.
- No training updates occurred. Smoke3 eval VMs booted but model calls failed: missing auto tool-call parser. Fixed common config with explicit hermes parser.
- User switched training to Terminal Lego. TMax mapping edits were removed.
- common.toml now selects terminal-lego, with the same Terminal Bench 2 evaluation.
- Added 600 creates/min pacing for both arms.
- Terminal Lego checkout complete; revision 92e6b5f577610cec9b040250a94ce66cfce24839, archived in dataset.json.
- User requested runtime/taskset validation through uv run eval; eval skill read.
- Stopped eval-preflight.toml run lego-runtime-preflight: two episodes each on Lego and TB2, concurrency 2, hosted default model.
- Eval log: /tmp/score-centering-eval-preflight.log; unified session 58039.
- Dataset prefetch completed. The dataset.json path is shared across training nodes.
- Single TB2 uv eval completed with SandboxError: RUNNING but gateway unreachable for 355s (being placed on a node). No model turns.
- Mixed Lego/TB2 uv eval stopped after outage confirmation. First TB2 VM failed to reach RUNNING within 355s. First Lego VM g2c5e574flcg4nwhihpry23u failed gateway readiness at 00:56:45, same being-placed error. Second Lego episode also failed provisioning; last TB2 episode was cancelled.
- Default-image control uv eval (gsm8k with bash/prime) failed VM provisioning at 00:59:45; zero turns.
- Worker-host control: SLURM 811 on node022, CPU4/memory16G/no GPUs. Cancelled at01:03 during task loading after user confirmed the outage. No conclusion from this control.
- eval_results.py exports trace-level rewards/errors and confidence intervals. Real failed preflight traces yield zero scored tasks and null success estimates.
- Explicit hermes parser and auto tool choice fixed both training arms; centered config dry-run passed.
- Analysis exporter captures numerical warnings omitted from sanitized metric files.
- Raw SDK smoke jobs cleaned up. One Lego image never became ready; a second reached RUNNING but had no command result before cleanup.
- VM sandboxes reject region overrides; eu-west test returned HTTP 400. Keep default region.
- All outage-era experiment jobs were stopped before the recovery preflight.
- User prefers the existing dashboard on port 7789. The output root is registered; both main runs were verified through /api/runs. An earlier isolated dashboard also remains on port 7790 (PID in dashboard.pid).
- W&B project: https://wandb.ai/primeintellect/score-centering-terminal
- IPO W&B: f2e85d1b20984a64bf6f476ea051bd04; SC W&B: a47c84d8a8bd425f976db52e7939c98b.
- Smoke2 validated all 12 endpoints and INT8 expert kernels / INT4 KV cache.
- Live request returned valid top128 IDs and probabilities for all eight generated tokens.
- No updates occurred in any preflight attempt. Jobs 804/805 are stopped.
- Pure-SC math check and dry-run config pass; dummy CP/diagnostic guards fixed after smoke2 launch.
- Attempts 799/800/801 excluded: checkpoint-conversion race, then missing quantization selector.
- Branch renamed exp/score-centering; main revision above is its base.

## Next

1. Monitor IPO861 and SC860 through 400 updates. Main configs use FP8 KV and uniform task sampling.
2. Fix runtime failures, with equal configuration changes in both arms. Keep total experiment nodes <=8.
3. Analyze the running main pair at matched steps and token budgets. Preserve failed attempts if a restart is needed.
4. Monitor to completion; collect curves and task traces. Apply PROTOCOL.md decision rules. TB2 initial success is sparse (IPO1/51, SC5/43 scored) with many errors. The protocol now fixes a supplemental unseen-Lego audit after training; select by exposure only, then evaluate base and both final checkpoints.
5. Repeat a positive separation with a second seed; use a matched BF16 control pair to check mismatch attribution.
6. Report positive, negative, or inconclusive results honestly.

## Operational details

- Sandbox exec fails with `bwrap: Failed to make / slave: Permission denied`.
  Use require_escalated for shell commands. Auto-review permits authorized work.
- Source ~/.env FIFO on auth failures. Never print or copy secret values.
- The working SSH agent at setup was /tmp/ssh-jqPF9sOsvD/agent.532361.
  HTTPS submodule initialization worked after SSH stalled.
- The fresh pydantic-config clone initially had only .git. Its empty checkout was repaired after read-only verification.
- Runtime installed with uv sync --all-extras --all-packages. Do not edit .venv files.
- Existing /tmp/six.py is unrelated and shadows the real six module.
  Run temporary scripts with `uv run --no-sync python -P ...`, or keep scripts here.
- Model snapshot downloaded and pinned by absolute path in common.toml.
- Sources: /tmp/score-centering-reference at 7c56e9e; /tmp/score-centering-paper.html.
- Dependency changes are archived in patches/; the submodule working trees remain dirty. Refresh patches after any further dependency changes.
- Other cluster jobs are unrelated. Do not touch them.

- Final audit export: checkpoints are DCP-only. After the training allocations finish, use GPU allocations for `uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py <step_dir>`. Serve exported weights with the same online INT8/FP8 quantization and FP8 KV. Never allocate export/eval nodes on top of the current eight training nodes.
- `select_holdout.py` was checked on completed smoke5 logs: 13825 source tasks, 72 excluded, 128 selected. That artifact is only a selector check, not the final audit set.
