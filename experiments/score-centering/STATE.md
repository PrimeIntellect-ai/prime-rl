# Experiment state

User authorized autonomous implementation, launches, monitoring, fixes, and analysis.
Latest main merged: a1822f7a0a8f0e4abcb4b1d07a7f6bba6c101d54.
Do not merge or open a PR unless asked. No subagents are authorized.

## Current work

- Current active pair: IPO893 (`ipo-1plus1-api4-seed42-r3`, launched09:08:21 UTC, nodes014/050) and SC891 (`sc-1plus1-api4-seed42-r2`, launched09:04:46 UTC, nodes057/058). Source differs only in dependency-sync control and experiment records; scientific settings match. User requested close monitoring at09:08 UTC.

- SC891 runtime confirms four API processes per engine. IPO892 inference failed in uv metadata resolution (GitHub wheel HTTP500) before model loading. Cancelled IPO892; relaunch IPO from scratch with UV_NO_SYNC=1 to use the installed environment. SC891 continues. This changes dependency resolution only, not the installed model/training stack.

- Current pair launched at 09:04:46 UTC: IPO892 (`ipo-1plus1-api4-seed42-r2`, nodes014/050), SC891 (`sc-1plus1-api4-seed42-r2`, nodes057/058). Two nodes each, launch source9a282107e.

- Relaunching as `ipo-1plus1-api4-seed42-r2` and `sc-1plus1-api4-seed42-r2` with the per-rank API override fixed. Generated script validation passed; actual API worker counts still require runtime verification.

- At 09:01 UTC, found the per-rank SLURM helper hard-coded --vllm.api-server-count 1 despite the resolved API4 config. Stopped jobs887/888 during startup. Added an experimental per-engine API count environment override, defaulting to the existing value1. Validate the generated script and actual runtime worker count before rollout collection.

- Fresh pair submitted at 08:57:24 UTC: IPO888 (`ipo-1plus1-api4-seed42`, nodes014/050), SC887 (`sc-1plus1-api4-seed42`, nodes057/058). Both RUNNING on two nodes each; source8f56c6326. Resolved configs confirm API4, initial512/max2048, IPOeps0.3 versus standalone SC, and resume=None.

- USER RESUMED at 2026-09-19 08:55 UTC: launch both from scratch. The prior stop is revoked.
- Fetched origin/main; a1822f7a0 remains latest and is already merged. Launch the validated 1+1/API4 configs with initial inflight512, cap2048, and 400 updates.

- Previous USER STOP at 2026-09-19 07:27 UTC: cancelled both runs because another run had priority; superseded by the 08:55 restart request.
- Jobs867/868 were already cancelled. Terminated both pending API4 launchers before SLURM submission; verified no experiment allocations remain. Preserve configs and all logs.

- At 07:24 UTC, interrupted both 1+1 orchestrators after repeated router unhealthy-worker errors; neither completed an optimizer update. Both reported forced cleanup complete. Cancelled allocations867/868 after episode cleanup.
- Preparing the identical 1+1 pair with api_server_count=4 per engine; fresh names `ipo-1plus1-api4-seed42` and `sc-1plus1-api4-seed42`. Initial inflight512 and max2048 stay fixed.

- Fresh pair launched concurrently at 07:10:42 UTC: IPO868 (`ipo-1plus1-seed42`, nodes014/050) and SC867 (`sc-1plus1-seed42`, nodes057/058). Both RUNNING, two nodes each, source 4909f8e98.

- At 07:09 UTC, stopped IPO861 and SC860 at user request to resize. The user explicitly approved starting from scratch. Both allocations have released.
- Replacement pair: `ipo-1plus1-seed42` and `sc-1plus1-seed42`. Each uses one trainer and one inference node, initial inflight 512, cap 2048. Both dry runs passed and request exactly two nodes. Preserve the old runs as a separate early attempt.

- Superseded 400-update pair started at 2026-09-19 06:29 UTC: IPO861 (`ipo-seed42`) and SC860 (`sc-seed42`), four nodes each. Both use FP8 KV, INT8 experts, FP8 dense weights, and uniform task draws with seed 42.
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
- Current IPO W&B: f7acea587c9a4bd5aeb33be0d270fd65; SC W&B: a5741426ea7d458783ceace65a5b1509.
- Superseded IPO W&B: f2e85d1b20984a64bf6f476ea051bd04; SC W&B: a47c84d8a8bd425f976db52e7939c98b.
- Smoke2 validated all 12 endpoints and INT8 expert kernels / INT4 KV cache.
- Live request returned valid top128 IDs and probabilities for all eight generated tokens.
- No updates occurred in any preflight attempt. Jobs 804/805 are stopped.
- Pure-SC math check and dry-run config pass; dummy CP/diagnostic guards fixed after smoke2 launch.
- Attempts 799/800/801 excluded: checkpoint-conversion race, then missing quantization selector.
- Branch renamed exp/score-centering; main revision above is its base.

## Next

1. Launch and monitor the fresh concurrent 1+1/API4 pair. Verify serving health under load before interpreting training.
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

- All 384 active VM sandboxes from the cancelled 4-node pair were deleted using exact old-run labels. Cleanup completed with zero errors; inventories are in results/initial-4node-attempt.
- Final audit selector now writes a complete eval.json preserving the training agent configuration and scheduled-eval sampling. Four attempts per task, pinned concurrency 128 for all models. The generated config passed uv run eval --dry-run against completed smoke5 logs.

- At09:13 UTC, both actual pools have API4 and roughly500 inflight episodes. No router health warnings or unhealthy-worker failures observed. SC has123 completed training episodes and zero training errors; no optimizer update yet. Read-only snapshot helper: /tmp/score-centering-monitor.py; artifacts: results/monitor/snapshots.jsonl.

- At09:26 UTC, IPO893 completed step1 (gradient0.4347, mismatch KL0.0063), SC891 completed step2 (gradient0.0832, KL0.0046); all finite. Weight refresh works with API4. SC step2 took3m39s, including3m03s waiting for rollouts and36s active work. No robustness separation established.
