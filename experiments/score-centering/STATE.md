# Experiment state

User authorized autonomous implementation, launches, monitoring, fixes, and analysis.
Latest main merged: a1822f7a0a8f0e4abcb4b1d07a7f6bba6c101d54.
Do not merge or open a PR unless asked. No subagents are authorized.

## Current state

- The user selected IPO versus IPO plus score centering on 2026-09-20.
- Both arms use IPO eps0.3; only the score-centering switch differs in the objective.
- Resume runs `ipo-eps03-seed42-compact-v1` and `ipo-sc-eps03-seed42-compact-v1` from their matched step-250 checkpoints.
- Do not resume the previous standalone treatment or mix its 50 updates into this comparison.
- `launch-pair.sbatch` reserves four nodes and retains CPU checks. Automatic watchdog cancellation is removed.
- Prior lineage and restart scripts are preserved under `archive/`.
- Both full-config dry runs passed. Resolved trainer, orchestrator, and inference settings match after normalizing run paths, except `trainer.loss.score_centering`.
- Both resolved trainers and orchestrators have `resume.step=250`, IPO eps0.3, adv_tau1.0, and kl_tau0.001.
- Weighted-score gradients match an independent dense reference, including advantages, loss masks, and loss weights.
- User authorized scheduling both fresh experiments at 2026-09-20T20:34:15.489287+00:00. This supersedes the prior stop.
- Paired SLURM job1039 released at 2026-09-20T20:38:17.616927+00:00. Four nodes total; both arms start from base weights.
- Runtime preflight completed: four scored episodes, zero errors; Lego1/2 and TB2 1/2 solved. One TB2 rollout reached its turn limit.
- Job1039 was cancelled by the allocation watchdog at 2026-09-21 16:59:14 UTC. IPO uses inference004/trainer007; IPO+SC uses inference008/trainer002. All four CPU-affinity checks passed at116 CPUs.
- At 2026-09-20T20:49:16.507229+00:00, both inference routers passed live compact-logprob generation checks: sampled scores and top128 head decode correctly. Both orchestrators are collecting initial train/eval rollouts at about517 inflight each. No completed optimizer update yet.
- At startup, the allocation-scoped stall guard emitted heartbeats with no stop reasons. Initial rollout logs include some malformed tool calls, one context overflow, and one sandbox provisioning failure; these are task/provider errors, not training instability.
- W&B IPO: https://wandb.ai/primeintellect/score-centering-terminal/runs/215ad83bb301492184c0ca1fc616a221
- W&B IPO+SC: https://wandb.ai/primeintellect/score-centering-terminal/runs/4ca8f7fb1e5544aa96b016c2f93ac8be
- Both runs are visible at http://localhost:7789. Live check artifact: `results/monitor/compact-live-1039.json`.
- Preflight: `ipo-sc-runtime-preflight-20260920`.
- Paired resume job1073 started at 2026-09-21T18:14:35Z on four nodes. IPO uses inference004/trainer007; IPO+SC uses inference008/trainer002.
- Both attempt-2 launchers resolve trainer and orchestrator resume step250. Each trainer checkpoint contains metadata and eight DCP shards.
- The paired launcher has no watchdog. It waits for both arms even if one arm fails, so one arm cannot cancel the other.
- Both step-250 restores completed. At18:25UTC, both orchestrators were collecting rollouts with about510 inflight episodes each.
- Resume W&B IPO: https://wandb.ai/primeintellect/score-centering-terminal/runs/d86337b26a3e42759894f58c661e846a
- Resume W&B IPO+SC: https://wandb.ai/primeintellect/score-centering-terminal/runs/0550d0f772144ec8bb32bf2436373419
- Job1073 completed normally with exit code0 on 2026-09-22T16:29:14Z.
- Both arms reached step400 and wrote complete final trainer and orchestrator checkpoints.

- User preference (2026-09-21): do not add watchdogs that automatically cancel jobs on stalled progress. Report problems without automatic cancellation. The launcher no longer starts a watchdog; `watch_progress.py` only records warnings.

### Job1039 cancellation (2026-09-21)

- Sandbox gateway failures appeared in both arms at16:18UTC: `Connect RPC failed (unavailable): Service Unavailable`.
- The 16:00 hour contains1284 such trace failures for IPO and1243 for IPO+SC. New sandboxes reported RUNNING but failed gateway reachability for355seconds.
- Both train batches stopped advancing around16:41UTC: IPO103/128 with441 inflight, IPO+SC124/128 with446 inflight.
- IPO completed285 updates (last16:38:50UTC); IPO+SC completed256 (last16:40:06UTC). Final gradients were finite:0.07065 and0.09872.
- At16:58:51UTC the guard recorded1201seconds without an IPO optimizer update. It signalled both orchestrators, then cancelled the shared allocation.
- SLURM reports CANCELLED by2005 because the watchdog runs as mika. The job had not reached its two-day time limit.
- This stop reflects sandbox gateway failures and rollout starvation; it does not establish numerical instability or score-centering robustness.
- Checkpoint directories exist through285/256, but these latest checkpoints have not been validated for restore.
- No restart was submitted during this diagnosis. Evidence: `results/monitor/guard-1039.jsonl`, `results/monitor/ipo_pair_1039.log`, and both run orchestrator logs.

### Previous standalone-estimator comparison

- USER STOP at 2026-09-19T21:32:51.919040+00:00: cancel jobs and summarize. Do not restart without a new user instruction.
- Job938 cancelled after both orchestrators completed forced cleanup at21:31:25UTC. Both exact run-label VM inventories are empty.
- Both arms recorded zero new optimizer updates in the corrected restart. The comparison remains at50 clean updates per arm.
- Job911 had previously wasted6.6hours with zero optimizer updates because the paired launcher restricted each node to one CPU.
- The passive monitor recorded the failure without intervention. This was a monitoring gap, not policy instability.
- The launcher fix requests116 CPUs per node; job938 verified all four affinity checks and232 CPUs per two-node step.
- Job938 restored step50 but was cancelled before update51. The corrected setup has not yet demonstrated training progress.
- `watch_progress.py` adds an automatic stop after20minutes without an update once serving is ready,45minutes of incomplete startup, or a nonfinite gradient. It runs inside the cancelled allocation; no relaunch loop exists.
- Implementation, configs, original step-50 checkpoints, logs, curves, and incident records are preserved.
- No experiment GPU jobs or pending restarts should remain. Unrelated jobs899 and878 were not touched.
- Evidence: `results/main-api4-step50`, `results/resume50-stall`, `results/user-stop-938`.
- The 400-update comparison, frozen-checkpoint audit, second seed, and BF16 control are unfinished. No robustness claim is established.

### Restart history

- Previous pair: IPO893 (`ipo-1plus1-api4-seed42-r3`, launched09:08:21 UTC, nodes014/050) and SC891 (`sc-1plus1-api4-seed42-r2`, launched09:04:46 UTC, nodes057/058). Source differs only in dependency-sync control and experiment records; scientific settings match. User requested close monitoring at09:08 UTC.

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

- Historical treatment: standalone score centering, matching the official PyTorch formula.
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

1. Validate recovery from the gateway incident. Resume both arms from preserved step 50, then continue toward 400.
2. Fix runtime failures, with equal configuration changes in both arms. Keep total experiment nodes <=8.
3. Analyze the running main pair at matched steps and token budgets. Preserve failed attempts if a restart is needed.
4. Monitor to completion; collect curves and task traces. Apply PROTOCOL.md decision rules. Current TB2 initial success is sparse (IPO2/40, SC1/48 scored) with many errors. The protocol now fixes a supplemental unseen-Lego audit after training; select by exposure only, then evaluate base and both final checkpoints.
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

- Final audit export: checkpoints are DCP-only. After the training allocations finish, use GPU allocations for `uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py <step_dir>`. Serve exported weights with the same online INT8/FP8 quantization and FP8 KV. Use released training nodes for final export/eval; preserve the user's one-trainer-plus-one-inference allocation per arm.
- `select_holdout.py` was checked on completed smoke5 logs: 13825 source tasks, 72 excluded, 128 selected. That artifact is only a selector check, not the final audit set.

- All 384 active VM sandboxes from the cancelled 4-node pair were deleted using exact old-run labels. Cleanup completed with zero errors; inventories are in results/initial-4node-attempt.
- Final audit selector now writes a complete eval.json preserving the training agent configuration and scheduled-eval sampling. Four attempts per task, pinned concurrency 128 for all models. The generated config passed uv run eval --dry-run against completed smoke5 logs.

- At09:13 UTC, both actual pools have API4 and roughly500 inflight episodes. No router health warnings or unhealthy-worker failures observed. SC has123 completed training episodes and zero training errors; no optimizer update yet. Read-only snapshot helper: /tmp/score-centering-monitor.py; artifacts: results/monitor/snapshots.jsonl.

- At09:26 UTC, IPO893 completed step1 (gradient0.4347, mismatch KL0.0063), SC891 completed step2 (gradient0.0832, KL0.0046); all finite. Weight refresh works with API4. SC step2 took3m39s, including3m03s waiting for rollouts and36s active work. No robustness separation established.

- Both step25 checkpoints saved. Each DCP metadata file is readable, and all 11,475 referenced chunks fit the shard files. Each trainer checkpoint is 341.24 GiB. Orchestrator progress exists. No full restore performed. Verification artifact: results/monitor/checkpoints-step25.json.
- analyze.py accepts --max-step to cap both curves at a matched optimizer update. Real export with --max-step 25 passed; results/main-api4-step25 contains no nonfinite metrics or numerical log warnings.

- Step25 evaluation repeated the same64 tasks. IPO scored42 and solved3; centered scored44 and solved2. Policy-refresh spans:51 IPO and46 centered episodes. No robustness separation. Most provider errors concern malformed tool-call JSON. Final fixed-checkpoint Lego audit remains required.

- At 11:57 UTC: IPO completed step41 and centered step40; all finite. IPO gradient rose to0.310 at39, then returned to0.108 at40 and0.074 at41. Centered step40 gradient0.065, mismatch KL0.0040. Continue toward400; next checkpoint/evaluation50.

- At 12:40 UTC, the matched 50-update export passed. Both arms remain finite. Training success windows (1–25, 26–50): IPO 58.5%, 55.6%; centered 59.1%, 59.5%. No predefined collapse.
- Both step-50 checkpoint manifests pass all 11,475 shard-extent checks. Each trainer checkpoint is 341.24 GiB. Orchestrator progress exists. No full restore performed.
- Mismatch tail plots now include per-token maximum and standard deviation. Maximum through step 50: IPO 189.65, centered 202.13. Mean mismatch remains about 0.0043. IPO gradient peaks at 39, 43, and 45 subsided on the following update.

- Prepared `audit-inference.json` for the final frozen-checkpoint audit. Dry run passed: one node, two TP4 engines, API4 per engine, same weight and KV quantization. No audit allocation submitted.

- At 13:28 UTC, SLURM estimates job911 start at16:48UTC (not guaranteed). Node031 is drained for a missing NVIDIA driver; only three nodes are schedulable and idle. Do not modify unrelated node state or jobs.
- Persistent monitor is running: PID 2120920, `/tmp/score-centering-watch-pair.py`, every60seconds. It checks job911, records training metrics/errors once running, appends per-run STATUS.md, and exits when the job terminates. Logs: `results/monitor/pair-watch.log`, `pair-watch.jsonl`; PID file:`pair-watch.pid`. This records status locally; it does not send chat notifications or replace the final scientific audit.

- At job938 startup, all four affinity checks reported116 available CPUs and the guard emitted heartbeats. The later user stop is recorded above.


## Compact logprob transport (2026-09-20)

Implemented opt-in packed uint32/float32 logprob responses and the renderer decoder.
Score-head requests enable the format by default; `PRL_COMPACT_LOGPROBS=0` selects the nested response.
The synthetic 4,096-token/top-128 benchmark shows 6.189 s versus 0.0622 s of server response work.
Response size falls from 34.91 MB to 5.76 MB, with exact sampled and head score parity.
See `COMPACT_LOGPROBS.md` for the scope and reproduction command.
No jobs were started. A live one-worker versus four-worker comparison remains pending.
