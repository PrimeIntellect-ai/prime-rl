# Night log: DeepSeek V4 Flash SWE bring-up (2026-09-19)

Companion to `SWE_RUN_HANDOFF.md`. Newest attempt at the bottom of the attempts section. Times are UTC.

## Pre-launch state (02:25)

- Head node has no `prime` CLI anywhere (`which prime`, `.venv/bin`, `~/.local/bin` all empty), so the
  `pre_run_command` (`prime sandbox delete --label dsv4-swe`) is a no-op on the head node and confirmed on
  compute nodes too (job 829 batch log: `timeout: failed to run command 'prime': No such file or directory`). The `|| true`
  keeps it from failing the job. Sandboxes themselves go through the `prime-sandboxes` SDK (0.3.0 installed)
  with `PRIME_API_KEY`, which does not need the CLI. Consequence: stale sandboxes from a killed attempt are
  not reaped between attempts; they fall back on the runtime `idle_timeout` of 3600 s.
  **Resolved 02:52**: Garrett was briefly awake and gave `uv tool install prime`; it installed Prime CLI 0.7.0 at
  `~/.local/bin/prime` (shared `/home`, on the sbatch-inherited `PATH`), and `prime sandbox list --label dsv4-swe`
  authenticates via `PRIME_API_KEY`. The reaper works from the next attempt on. Docs:
  https://docs.primeintellect.ai/cli-reference/introduction
- `attempt_1` log and launcher directories exist but are empty (job 819 never wrote anything into them
  before it was cancelled).
- Model is fully cached: `HF_HOME/hub/models--PrimeIntellect--DeepSeek-V4-Flash-0731-bf16` is 1.1 TB.
- `scaleswe` and `prime_sandboxes` import under `uv run`.
- Dry run took 14 s, not 4 min; the environment was already resolved. It bumped the run dir to `attempt_2`.
- Handoff's three-way resolved-config check: orchestrator OK, trainer OK, inference OK.
- Resolved `runtime.type = "prime"`, `labels = ["dsv4-swe"]`, `cp_style` unset. No checkpoints exist yet, so
  the bare `[resume]` is a fresh start.
- `sinfo`: 26 idle nodes at submit time.

- 02:58, Garrett's request: future attempts log to wandb `primeintellect/deepseek-v4-flash` with a descriptive
  run name and tags. Config updated (`entity`, `name = "swe-scaleswe-131k-fp8-adamw1e-6-bs64g8-8t8i"`, `tags`).
  Job 829 keeps its original `dsv4-swe-131k` wandb name; the new name applies from the next attempt.

## Attempts

### Attempt 3 (SLURM job 829), submitted 02:30, 16 nodes

Command: `uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml`
Config at `e1e8e4881` (unchanged from handoff). Run-dir attempt number is 3 because the dry run consumed 2.
Nodes: `prime-nebius-puku-h200-gpu-[005,012,014-015,023,026-028,033,035,040-041,051,056-058]`.
Logs: `/home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_3/`, batch log `launcher/logs/job_829.log`.

#### Attempt 3 progress

- 02:48 inference replica 0 finished loading weights (858 s from load start, FP8 quantizing online).
- 02:57 **Gate 1 (KV pool) cleared, far above expectation**: `Available KV cache memory: 77.53 GiB`,
  `GPU KV cache size: 1,513,358 tokens, Maximum concurrency for 131,072 tokens per request: 11.55x`.
  The handoff expected ~1.4x per replica; the earlier 183,793-token measurement evidently did not have
  `kv_cache_dtype = "fp8"` + the `fp8_ds_mla` layout in effect, or profiled differently. Consequence: the
  8-replica inference fleet holds ~92 full-length requests, well above `max_inflight = 512 / 8` per replica
  only if requests are short; either way the FP8-for-KV argument is much weaker than the plan assumed.
  Left the config alone (bring-up run); see open questions.
- 03:02:41 **Attempt 3 died**: orchestrator `TimeoutError: Inference server is not ready after 1800 (>1800)
  seconds` (`src/prime_rl/orchestrator/clients.py:365`, `AdminPlane.wait_for_ready`). Log:
  `logs/attempt_3/orchestrator.log`. wandb run: https://wandb.ai/primeintellect/deepseek-v4-flash/runs/399fe8aa204c4b91af5db89777718429
  - Per-replica `Loading weights took`: 859, 878, 881, 909, 935, 939, 1076, 1111 s (spread from NFS read
    contention across 8 nodes each pulling the 1.1 TB bf16 checkpoint). Then ~4 min of KV profiling +
    CUDA graph capture (~65 s). 6 of 8 API servers reported `Application startup complete` between
    02:59:10 and 03:01:04; nodes 2 and 5 were at KV profiling at 03:02:45, four seconds after the timeout.
  - Trainer never got past `Initializing weight broadcast` (waiting on inference); no trainer error.
  - The failed component put the sbatch into its 3600 s checkpoint-flush grace, so I `scancel 829` at 03:04
    rather than let it idle 16 nodes.
  - Fix: `[orchestrator.model.client] wait_for_ready_timeout = 7200` (default 1800, field in
    `configs/shared.py:174`). Rejected: fewer replicas (does not fix a per-replica timing problem) and
    pre-quantized FP8 weights on disk (would cut load time 2x but changes what the run measures).

### Attempt 5 (SLURM job 833), submitted 03:05, 16 nodes

Command: `uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml`. Config at `79292001b` (adds the 7200 s
inference ready timeout and the wandb entity/name/tags). Run-dir attempt 4 was the dry run.
Nodes: `prime-nebius-puku-h200-gpu-[005-006,015,017,024,027,033,036,040,042,046,048-049,052,057,063]`, mostly
different from attempt 3, so node-local JIT caches are cold again on most of them.
Logs: `/home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_5/`, batch log `launcher/logs/job_833.log`.
Expect the inference fleet to be ready around 03:40.

#### Attempt 5 progress

- 03:05 batch log: the reaper now finds the CLI but refuses: `Error: Cannot scope to your sandboxes - no user_id
  configured. Use --all-users ... or configure your user_id.` Team id comes from `PRIME_TEAM_ID`, user id was
  unset. Fixed without a config change: `prime whoami` fetched and persisted the user id into
  `~/.prime/config.json` (shared `/home`, so compute nodes see it). Verified the scoped delete path with a
  no-match label. Applies from the next attempt; this attempt's pre-run reap was a no-op again (harmless, no
  sandboxes existed). `PRIME_USER_ID=<id>` in the `pre_run_command` would be the fallback if the config file
  ever gets reset; the top-level `[env_vars]` are exported only inside the component blocks, after the
  pre-run command, so they cannot carry it.
- 03:07 orchestrator: `Train environments ready in 21.6s` (scaleswe env server up).
- 03:24 replica 0 `Loading weights took 874 s`; 03:34 KV pool again 1,513,358 tokens / 11.55x.
- 03:43:45 **`Policy inference pool ready after 36m 24s`** (the 1800 s default would have failed again by a
  wide margin). Trainer: `Broadcasting startup policy weights (v0) to inference engines`.
- 03:43:56 during the v0 broadcast every inference rank logged
  `CUDACachingAllocator ... memory allocation failed with OOM ... trying to allocate 13191086080 bytes (free: ~3.3 GB,
  total: 150 GB)`. This is the allocator's warn-then-flush-cache-and-retry path, not a crash: all ranks then
  logged `Receiving state dict 44/44`, `POST /update_weights 200 OK`, `UNPAUSED`. So the broadcast receive
  buffer (~12.3 GiB) only fits after vLLM's cache is dropped, i.e. `gpu_memory_utilization = 0.85` leaves ~3 GB
  of true headroom on a 150 GB card. **Hazard**: if a later `update_weights` OOMs for real, drop
  `gpu_memory_utilization` to 0.80 (KV pool has 11.55x concurrency to spare). Not changing it pre-emptively.
- 03:43:45 orchestrator: `Derived initial max inflight 92 - 12.1M KV cache tokens / 131.1K tokens per episode`,
  so the configured `max_inflight = 512` is capped at 92 by the KV pool. 03:44:14 `Starting orchestrator loop`.
- 03:44:14 inference: 2x per rank `DeepseekV4ScalingRotaryEmbedding: Failed to load weights`. Benign. It comes
  from vLLM's reload path (`model_loader/reload/layerwise.py:268`): the broadcast state dict has no entries for
  the rotary layer's precomputed tables, so the loader restores the layer's own tensors, which are derived from
  config and identical. It did not appear at initial load. Expect it on every weight update.
- 03:44:18 first sandboxes up (`aweaiteam/scaleswe:*` images), 63 running by 03:45. The `scaleswe` + prime
  sandbox path works with this model and renderer, at least through provisioning.
- 04:00:13 **orchestrator Step 1**: `15m 56s | Reward 0.6250 | Trainable 64/64 (100.0%) | Turns 25.6 | Branches 1.1
  | Max Off-Policy 0 | Error 0.0% | Cancelled 0.0% | Truncation 0.0%`. The whole scaleswe + prime sandbox +
  deepseek-v4 renderer path works end to end. Batch shipped to the trainer; waiting on trainer step 1 (Gate 2).
- 04:03:50 first rollout failures: all 8 traces of one group (task 41) died with
  `HarnessError: harness setup: RuntimeError: failed to prepare uv script: error: unexpected argument '--script' found`.
  The `uv` baked into that task's sandbox image predates `uv run --script`, so the bash harness cannot set up.
  Per-image, not systemic: 8 of the first 225 finished rollouts, the other 217 `stop=agent_completed`. The group
  yields no signal and is dropped; left alone. Worth a taskset-level fix (pin/upgrade uv in the harness setup)
  but not tonight.
- 04:04:34 **Attempt 5 died**: the v1 weight update after trainer step 1 OOMed on all 64 inference ranks:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.25 GiB. GPU 3 has a total capacity of
  139.80 GiB of which 7.56 GiB is free. Including non-PyTorch memory, this process has 132.23 GiB memory in use.
  Of the allocated memory 116.95 GiB is allocated by PyTorch` at `src/prime_rl/inference/vllm/worker/nccl.py:44`
  (`receive_state_dict`, the per-dtype staging buffer). `POST /update_weights` returned 500, the engine cores
  raised `RuntimeError: Worker failed`, and the trainer hung inside its NCCL send with no step line. Cancelled
  04:06:20. Log: `logs/attempt_5/inference/node_*.log`.
  - Arithmetic: 139.8 GiB card, `gpu_memory_utilization = 0.85` gives vLLM ~118.8 GiB, non-torch allocations
    (NCCL, graphs) ~15.3 GiB, so ~7.5 GiB was free and the receive buffer needs 12.25 GiB. The v0 broadcast
    at 03:43 only survived because the allocator could still flush ~10 GiB of cached blocks.
  - The sender (`src/prime_rl/transports/weights/nccl.py:broadcast_state_dict`) streams one decoder layer at a
    time, grouped by dtype; 12.25 GiB is one layer's bf16 expert weights, so the buffer size is inherent to
    the transport and not configurable. Fix chosen: `gpu_memory_utilization = 0.75` (~19 GiB free). Rejected:
    0.80 (leaves ~14 GiB, too thin against a 12.25 GiB buffer plus fragmentation) and patching the receiver to
    sub-chunk (code change in a hot path, not for tonight).
  - **Gate 2 mostly cleared**: the trainer finished forward, backward and the optimizer step on a 64x131k batch
    at 8 nodes without OOM (step took ~4 min from batch arrival at 04:00 to the broadcast at 04:04). The
    `Peak Mem.` number itself was never printed because the step log follows the broadcast.

### Attempt 7 (SLURM job 837), submitted 04:08, 16 nodes

Command: `uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml`. Config at `f7ceea189`
(`gpu_memory_utilization = 0.75`). Run-dir attempt 6 was the dry run.
Nodes: `prime-nebius-puku-h200-gpu-[005-006,013,015-016,018,020,024,027,033,036,038,042,046,052,055]`.
Logs: `/home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_7/`. Expect inference ready ~04:45, first
trainer step ~05:05, and the v1 update right after is the moment of truth.

#### Attempt 7 progress

- 04:09 batch log: `Processed 127 sandbox(es)` / `Successfully deleted 127 sandbox(es)`. The reaper works now
  that the CLI is installed and the user id is configured; attempt 5's leftovers are gone.
- 04:13:29 replica 0 `Loading weights took 26.72 seconds` (vs 859-1111 s in attempts 3 and 5). Node reuse
  across attempts: the 1.1 TB checkpoint is still in that node's page cache. Boot time is dominated by the
  cold NFS read, so re-landing on warm nodes matters a lot for relaunch cost.
- 04:24:32 KV pool at 0.75: `GPU KV cache size: 1,240,483 tokens, Maximum concurrency for 131,072 tokens per
  request: 9.46x` (was 11.55x at 0.85). Orchestrator inflight cap will be ~75 instead of 92.
- 04:43:57 `Policy inference pool ready after 32m 47s` (3 warm nodes, 5 cold at 1058-1119 s load).
- 04:44:23 v0 broadcast: `Receiving state dict 44/44`, `POST /update_weights 200 OK`, and **zero**
  `memory allocation failed` allocator warnings across all 64 ranks (attempt 5 had one per rank). Orchestrator
  loop started 04:44:24. Next: orchestrator step 1 ~05:00, trainer step 1 + v1 update ~05:05.
- 04:57:44 orchestrator Step 1: `13m 16s | Reward 0.7812 | Trainable 64/64 | Turns 31.1 | Error 0.0% | Truncation 0.0%`.
- 05:02:21 **trainer Step 1, the run is alive end to end**:
  `18m 23s | Loss -0.0003 | Entropy 0.3717 | Mismatch KL 0.0249 | Grad. Norm 0.0560 | LR 1.00e-06 | Throughput 8712
  tokens/s | MFU 5.4% | Peak Mem. 89.6 GiB | Max Vio 6.0787 | Routing Conf. 0.0992`.
  - **Gate 2 cleared**: 89.6 GiB peak on a 139.8 GiB card at 8 trainer nodes, cp = 4, 64 x 131k batch. The
    handoff expected ~66 GiB weight-shaped state + ~24 GiB broadcast transient + ~11 GiB activations, so this is
    right on the arithmetic. ~50 GiB of headroom; the memory ladder was not needed.
  - v1 `POST /update_weights`: 200 OK on all 8 replicas, zero OOM or allocator warnings. The 0.75 fix holds.
  - Step 1 wall time includes the first-call TileLang / torch.compile warmup, so 18 min is an upper bound.
  - Mismatch KL 0.0249 is in line with the FP8 measurement in `FP8_MISMATCH_RESULTS.md` (0.0307 at lr = 0),
    i.e. above the 0.015 bar that document discusses. Expected consequence of the FP8 decision, not a bug.
- 05:04:29 trainer Step 2: `2m 8s | Peak Mem. 71.8 GiB | Mismatch KL 0.0277`; Step 3 `Peak Mem` similar.
- 05:12 **Gate 3 (who waits on whom)** from `monitors/file/metrics.jsonl`, steps 1-3:

  | step | trainer time/step | wait_for_batch | forward_backward | broadcast_weights | orch time/step | orch wait_for_policy |
  |---|---|---|---|---|---|---|
  | 1 | 1104 s | 799 s | 241 s | 37 s | 957 s | 0 |
  | 2 | 128 s | 12 s | 93 s | 22 s | 293 s | 0 |
  | 3 | 306 s | 169 s | 112 s | 24 s | 285 s | 0 |

  Rollout-bound: the trainer spends most of each step waiting for a batch and the orchestrator never waits on
  the policy. Steady state is ~5 min per step, set by the 64-episode rollout collection. Mean episode length is
  10-27k tokens (`num_total_tokens/mean`), so 131k is a ceiling, not the typical case. At this pace step 20
  (first checkpoint) lands around 06:35.
- 05:17-05:46 trainer steps 5-15 at 2.5-4 min each, Peak Mem 72-93 GiB, mismatch KL 0.023-0.032, grad norm
  0.005-0.09. Every step carries the `Trainer waited ... add more inference nodes` warning; rollout-bound as
  measured above, and 16 nodes is the cap, so left alone.
- 05:48:23 orchestrator: `Discarded 72/136 episodes (52.9%): stale=0, errored=0, no_signal=72`. Groups whose 8
  rewards are identical (mostly all-pass, given mean reward 0.78-0.94) have zero advantage and are dropped.
  A throughput cost, not a failure; the batch still reached 64 effective episodes. See open questions.
- 06:01:16 **first checkpoint, step 20**: trainer `Saving checkpoint at step 20` at 05:57:44, step 20 logged at
  06:01:16 with `6m 27s` total versus 1.5-4 min for neighbouring steps, so the 3.2 TB DCP write took roughly
  3.5 min. `checkpoints/step_20/` has both `trainer/` (3.2T) and `orchestrator/`, so a bare `[resume]` is safe
  from here. Step 21 followed at 06:02:45, so nothing stalled after the save. Next checkpoint at step 40 will
  test `keep_last = 2` cleanup at step 60.
- 06:12:38 second per-image harness failure class: `SandboxError: prime exec failed: Connect RPC failed
  (invalid_argument): resolve command "/bin/bash": stat /bin/bash: ...` (one group; the task image has no bash).
  Same bucket as the `uv --script` failure: image-level, a few percent of episodes, left alone.
- 06:02-06:14 steps 21-27 at 1-3 min each, Peak Mem 72-92 GiB, mismatch KL 0.020-0.025. Nothing new.
- 06:18:54 orchestrator Step 30: `Max Off-Policy 26` against `max_off_policy_steps = 32`. Steps 17-30 range
  6-26 with no trend and `stale=0` in every `Discarded` warning so far. Because trainer steps are ~2.5 min and
  a long SWE rollout can run over an hour, the longest episodes sit close to the stale cutoff. Watch item: if
  `stale=` becomes non-zero, the fix is a higher `max_off_policy_steps` (a `[orchestrator]` knob), not a guard.
- 06:26:28 orchestrator: `Cancelled 2 train episodes past max_off_policy_steps=32. Consider increasing it`.
  The watch item above fired. Decision: leave the run alone. Raising the knob needs a relaunch (~35 min boot
  plus resume from step 20, losing ~15 steps) to save a handful of the longest episodes per hour. Will revisit
  if cancellations become a large fraction of a batch. Recommend `max_off_policy_steps = 64` on the next
  relaunch; see open questions.
- 06:34:19 `Cancelled 8 train episodes past max_off_policy_steps=32`: 10 cancelled in total by step 40, under
  0.5% of ~2,500 episodes. Holding the no-restart decision.
- 06:46:08 **checkpoint step 40**: trainer save started 06:43:05, `Step 40 | 5m 12s` (roughly 3 min of write).
  `checkpoints/` now holds `step_20` and `step_40`, both complete, 5.4 TB total; `/home` at 18% used, 207 TB
  free. `keep_last = 2` should delete `step_20` after the step 60 save (~07:30).
- 06:47 hourly summary: steps 21-40 in 45 min (~2.3 min/step), reward 0.72-0.95, mismatch KL 0.020-0.027,
  grad norm 0.006-0.09, Peak Mem 72-92 GiB, no errors beyond the two image-level harness failures.
- 07:00:36 orchestrator Step 50: `Reward 0.9844 | Cancelled 15.3%`. Per-batch cancelled share since the first
  hit: 4.5% (34), 4.0% (42), 6.8% (44), 5.1% (49), 15.3% (50), 0% elsewhere; 19 episodes total out of ~3,200
  dispatched (0.6%). Reward over steps 46-50 is 0.83, 0.94, 0.95, 0.94, 0.98 versus 0.69-0.89 for steps 29-43,
  which fits a selection effect: the episodes that outlive 32 steps are the long, hard ones. Still holding
  (the run is alive; the knob changes the data mix). If Garrett wants it changed, the cheapest moment is right
  after a checkpoint: `scancel`, set `max_off_policy_steps = 64`, relaunch with the bare `[resume]`.
- 07:25:09 **checkpoint step 60 and first `keep_last` cleanup**: save 07:21:13, `Step 60 | 4m 51s`; `checkpoints/`
  now holds `step_40` and `step_60` only (6.3 TB), so `step_20` was deleted after the save as designed.
- 07:26 hourly summary: steps 41-61 in ~40 min (~2 min/step), reward 0.70-0.98, mismatch KL 0.020-0.027, grad
  norm 0.006-0.09, Peak Mem 79-92 GiB. Off-policy cancellations 42 total by step 61 (~1% of episodes, spiky per
  batch, 0-15%). No new failure classes.
- 07:56:00 one `SandboxError: prime exec failed: Connect RPC failed (unavailable): Bad Gateway` (transient
  sandbox API blip, single trace, did not recur).
- 08:05:22 **checkpoint step 80**: `Step 80 | 6m 46s`; `checkpoints/` holds `step_60` and `step_80` (6.3 TB).
- 08:06 hourly summary through step 80: ~2 min/step, reward 0.70-0.98, mismatch KL 0.020-0.028, Peak Mem 72-92
  GiB. Failure tally since 04:44: 73 image-level harness failures (`uv --script` / no `/bin/bash`), 68 off-policy
  cancellations, 1 Bad Gateway, out of ~5,100 episodes. No trainer or inference errors since the v0 broadcast.
- 08:06-08:09 **broadcast stall, self-resolved**: trainer `Still waiting for the broadcast receiver after 60s /
  120s / 180s` for v81. Timeline from `broadcasts/step_81/` markers: `.sender_ready` 08:05:59, `.receiver_ready`
  and `.started` 08:09:28, `.finished` 08:09:48; trainer `Step 81 | 4m 26s` at 08:09:49. Inference was healthy
  throughout (all 8 replicas serving, no errors) and the orchestrator process was alive (25% CPU, 64 threads in
  futex wait, no tracebacks). Cause: `WeightWatcher.apply_policy_update` runs
  `Dispatcher.on_version_pending` before acknowledging, and that first does `async with self.scheduling_lock`
  to let an in-flight `fill_inflight` finish; with ~300 live sandboxes that scheduling pass took ~3.5 min.
  The 8 stale cancels then landed at 08:09:28 and the ack followed immediately. So the stall length tracks
  sandbox provisioning latency, and `weight_broadcast.timeout = 3600` bounds it. py-spy was denied (ptrace).
  No action; noted in open questions as something that would get worse with more inflight rollouts.
- 08:14-08:42 three new one-off sandbox failure classes, one trace each: `prime sandbox provisioning failed:
  Sandbox ... is not running`, `Read file failed: HTTP 503 ... sandbox_not_placed`, and `harness 'bash' exited
  137` (OOM kill inside a sandbox). All task-level; none recurred.
- 08:50:36 **step 100, checkpoint saved** (`Step 100 | 5m 51s`); `checkpoints/` holds `step_80` and `step_100`.
- 08:51 hourly summary through step 100: 4h 42m of job time, ~2 min/step, reward 0.70-0.97 (last 10 steps
  mean ~0.87), mismatch KL 0.020-0.034, grad norm 0.03-0.09, Peak Mem 72-92 GiB. 10,577 episodes finished.
  Failure tally: 88 `uv --script` image failures, 95 off-policy cancellations, 5 assorted sandbox one-offs.
  One broadcast stall (3.5 min at step 81), none since.
- 09:33:14 step 120 checkpoint saved (`Step 120 | 5m 41s`); `checkpoints/` holds `step_100` and `step_120`.
  Steps 101-120 uneventful: ~2 min/step, mismatch KL 0.021-0.027, Peak Mem 72-92 GiB.
- 09:48 hourly summary through step 130: 5h 39m job time, ~2 min/step, reward 0.73-0.98 (last 10 mean ~0.85),
  mismatch KL 0.020-0.034, grad norm 0.008-0.09, Peak Mem 72-92 GiB. 13,662 episodes finished. Cumulative:
  132 off-policy cancellations, 100 trace failures (88 `uv --script`, 6 sandbox OOM exit 137, 6 sandbox API
  one-offs). No further broadcast stalls after step 81. Cluster is otherwise full (1 idle node).
- 09:56:56 **burst of 29 `HarnessError: harness 'bash' exited 1`** within one second, all with the same stderr
  tail (a long list of numeric coordinate pairs, evidently tool output from one task that leaked into a shared
  log). Identical text across 29 traces in mid-rollout (turns 11-97) means one shared harness worker process
  died and took every rollout it hosted with it. Confined to that second; no inference errors, sandbox API
  responsive (512 live sandboxes, ~2 s list). The orchestrator kept collecting (`Train batch 21/64`). New
  failure class, watching for recurrence; a repeat pattern would point at the harness runner rather than tasks.
- 10:13:28 step 140 checkpoint saved (`Step 140 | 5m 47s`); `checkpoints/` holds `step_120` and `step_140`.
  No repeat of the 09:56 harness burst. Steps 131-140 routine.

## Open questions for Garrett

- KV pool is 11.55x concurrency per FP8 replica at 131k (1.51M tokens), not ~1.4x. Worth re-checking whether
  bf16 serving now fits too (the 0.74x figure may have predated `kv_cache_dtype = "fp8"`), which would reopen
  the FP8-vs-bf16 mismatch trade. Also 8 inference replicas may be more than the 8-node trainer can consume.
- The `scaleswe` bash harness fails setup on some task images with `uv ... unexpected argument '--script'`
  (old `uv` in the image), and on others with no `/bin/bash`. Whole groups die with reward 0 and no signal (task 41 in attempt 5; 7.1% error on
  attempt 7 step 2). Fix belongs in the harness setup or the images, not this config. Left alone.
- Mismatch KL is ~0.025 at step 1 under FP8 serving with the indexer excluded, consistent with the
  `FP8_MISMATCH_RESULTS.md` numbers and above the 0.015 bar discussed there. Not touched (FP8 on/off is yours).
- `wait_for_ready_timeout = 7200` and `gpu_memory_utilization = 0.75` are now in the config. The second costs
  ~18% of the KV pool (11.55x to 9.46x concurrency per replica). An alternative that keeps 0.85 would be
  sub-chunking the NCCL receive buffer in `inference/vllm/worker/nccl.py`; I did not make that code change.
- Roughly half of all episodes are `no_signal` (whole group same reward) because scaleswe pass rates are high
  for this model (`train/agg/effective/agent/reward/mean` 0.78-0.94). That halves effective rollout throughput
  on an already rollout-bound run. Options for later: a curriculum/difficulty filter on the source, a larger
  `group_size`, or a harder taskset mix. Not a bring-up concern.
- `max_off_policy_steps = 32` is being hit: the longest SWE rollouts outlive 32 trainer steps at ~2.5 min each
  (first 2 cancellations at 06:26, step ~33; 19 by step 50, up to 15% of a single batch). The episodes it drops are the hardest, longest ones, which biases
  the batch toward short tasks. Suggest 64 on the next relaunch. I did not restart the run for this.
- Weight updates can stall for minutes because the watcher waits on the dispatcher's `scheduling_lock` before
  acknowledging the trainer's broadcast (3.5 min at step 81 with ~300 live sandboxes). The trainer idles for
  that whole time. Worth a look at whether the stale-drain barrier needs the full scheduling pass, or whether
  `fill_inflight` should yield more often; `weight_broadcast.timeout = 3600` is the only guard today.
