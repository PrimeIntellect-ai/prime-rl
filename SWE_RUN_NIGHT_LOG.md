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
  log). Identical text across 29 traces in mid-rollout (turns 11-97) spanning ~25 different tasks. Confined to that
  second; no inference errors, sandbox API responsive (512 live sandboxes, ~2 s list). The orchestrator kept
  collecting (`Train batch 21/64`). **Correction (18:05, seen again in the GLM run with byte-identical stderr
  across dozens of tasks)**: the text is not task or model output and not one worker's crash; it is a shared
  failure path in the harness tooling that flakes for ~30 sandboxes at once, about once per run. Task- and
  model-independent, self-clearing.
- 10:13:28 step 140 checkpoint saved (`Step 140 | 5m 47s`); `checkpoints/` holds `step_120` and `step_140`.
  No repeat of the 09:56 harness burst. Steps 131-140 routine.
- 10:30 hourly summary through step 150: 6h 22m job time, ~2 min/step, reward 0.53-0.98 (last 10 mean ~0.82),
  mismatch KL 0.024-0.034, grad norm 0.009-0.10, Peak Mem 72-92 GiB. 15,958 episodes finished. Cumulative: 160
  off-policy cancellations, 129 trace failures (of which 29 were the single 09:56 harness burst; no repeat).
- 10:52:24 step 160 checkpoint saved (`Step 160 | 6m 10s`); `checkpoints/` holds `step_140` and `step_160`.
  Steps 151-160 routine.
- 11:18 hourly summary through step 172: 7h 09m job time, ~2 min/step, reward 0.72-0.92 (last 10 mean ~0.85),
  entropy 0.29-0.50, grad norm 0.01-0.10, Peak Mem 72-92 GiB. 18,354 episodes finished; 198 cancellations, 131
  trace failures, no further harness bursts. `checkpoints/` holds `step_140`, `step_160`.
  - **Mismatch KL is drifting up.** Per step: 142-150 in 0.025-0.034, 151-160 in 0.027-0.040, 161-172 in
    0.033-0.054 (peak 0.054 at step 167). Steps 1-100 sat in 0.020-0.034. Reward and entropy are flat, so this
    is not a divergence, but it is a monotone-ish trend over ~70 steps. Plausible cause: the trainer's bf16
    weights drift from the checkpoint that the FP8 quantization was calibrated against on every update, so
    the online per-block FP8 re-quantization of the broadcast weights disagrees more as the policy moves.
    No action taken (FP8 on/off is Garrett's call); see open questions.
- 11:19:16 one `HarnessError: harness setup: IndexError: list index out of range` (single trace, new class).
- 11:38:03 step 180 checkpoint saved (`Step 180 | 6m 2s`), but **`Mismatch KL 0.0780`**, the run's highest, and
  the drift is accelerating. From `monitors/file/metrics.jsonl`:

  | step | mismatch_kl mean | is_masked mean | off_policy mean | entropy | grad norm | reward |
  |---|---|---|---|---|---|---|
  | 1 | 0.025 | 0.0003 | - | 0.37 | 0.056 | 0.63 |
  | 100 | 0.025 | 0.0009 | 7.1 | 0.46 | 0.055 | 0.95 |
  | 160 | 0.029 | 0.0053 | 9.5 | 0.45 | 0.050 | 0.83 |
  | 170 | 0.042 | 0.0066 | 5.8 | 0.36 | 0.037 | 0.89 |
  | 178 | 0.052 | 0.0115 | - | 0.39 | 0.044 | - |
  | 180 | 0.078 | 0.0188 | 4.9 | 0.42 | 0.071 | 0.92 |

  Reading: the trust-region masked fraction is up 60x, mean KL 3x, while staleness, entropy, grad norm and
  reward are flat. Flat staleness rules out off-policy lag as the driver; a growing trainer-vs-inference
  logprob gap with a slowly moving policy points at the FP8 serving path (online per-block re-quantization of
  the broadcast weights, indexer kept bf16) disagreeing more with the bf16 trainer as weights leave the
  original checkpoint. Not yet harming training (loss ~0, reward stable), and the IPO mask is absorbing it, but
  at this slope the masked fraction is a few percent within another 50 steps.
  **Not acting**: turning FP8 off is on the handoff's do-not-decide-alone list, and a bf16 relaunch also needs
  the KV-pool question answered (bf16 was measured at 0.74x concurrency per replica before `kv_cache_dtype =
  "fp8"`; with FP8 KV it is likely fine). The run stays up. If Garrett wants the switch, the recipe is: `scancel`,
  drop `quantization`, `quantization_config` and `use_deep_gemm` in `[inference]` and `[inference.vllm]`, keep
  `kv_cache_dtype = "fp8"`, relaunch with bare `[resume]` from `step_180`, and confirm the KV pool clears 131k.
- 12:16:16 step 200 checkpoint saved (`Step 200 | 5m 10s`); `checkpoints/` holds `step_180` and `step_200`.
- 12:17 hourly summary through step 200: 8h 08m job time, ~2 min/step, reward 0.69-1.00 (last 10 mean ~0.87),
  Peak Mem 72-92 GiB. 21,216 episodes finished; 230 cancellations, 134 trace failures, one env-server
  `interception: unauthorized request` warning (11:41, single). No harness bursts, no broadcast stalls.
  - **Mismatch KL update, correcting the 11:38 entry**: the rise was a transient bump, not a runaway. Per step
    180-200: 0.078 0.057 **0.100** 0.071 0.077 0.069 0.060 0.056 0.051 0.050 0.051 0.043 0.049 0.045 0.047
    0.043 0.047 0.050 0.053 0.046 0.048. Masked fraction peaked at 2.9% (step 182) and is back to 0.7-1.0%.
    So the run now sits at roughly 2x its first-100-step mismatch (0.045-0.05 vs 0.025) with a plateau rather
    than a slope. Still worth Garrett's eyes, but it did not warrant an overnight intervention.
- 12:19-12:21 two `Dropped 8 queued traces past max_off_policy_steps=32` right after the step 200 checkpoint
  pause; transient, same staleness family.
- 12:40 **tunnel outage, ongoing, degrading rollouts by ~70%**. Burst of `HarnessError: harness 'bash' exited 1`
  whose traceback ends in `openai.InternalServerError: <html>... 502 Bad Gateway ... nginx/1.27.5`, raised from
  `client.chat.completions.create` inside the sandboxed bash harness. Sustained 56-80 failures/min against
  18-29 completions/min from 12:40 on; orchestrator step 216 reported `Error 36.2%`. Steps still complete
  (batches fill from the surviving 3/4), so no guard has fired.
  - Path: sandbox harness -> `https://<tunnel-id>.tunnel.pinfra.io` (Prime frps behind nginx) -> `frpc` on the
    orchestrator node -> local interception server (aiohttp, 4 in the elastic pool) -> vllm-router. Inference
    engines and router: zero errors. Sandbox API: fine (512 live, 2-3 s list).
  - Diagnosis: probed every tunnel's public URL from the head node (`/v1/models`, expecting 401 behind basic
    auth). Active pool = local ports 34457, 38669, 37619, 40447. Three answer 401 in 0.2 s. The fourth,
    `t-2-aada33686cd8ced8` (port 37619), alternates 401-instant / 502-after-30 s. Its local server answers 401
    in 45 ms from the node, so the break is between frps and frpc. Least-loaded balancing keeps assigning new
    sessions to the broken server, so it never drains.
  - Surgical attempt: started a second `frpc` for the same toml (rejected: `proxy already exists`), then SIGKILLed
    the wedged frpc (pid 336786, 6h47m old; needed -9) and relaunched -> `start proxy success` at 12:50:01, but
    public probes still alternate 401 / 502-30s. So the fault is on Prime's side for this tunnel id (looks like
    round-robin over one healthy and one stale frontend route), not in our client. Only a fresh tunnel id fixes
    it, and only the env server's pool can mint one.
  - Side finding: 20 `frpc` processes alive for a 4-server pool; the elastic pool leaks the old tunnels when it
    resizes (every earlier tunnel from 03:45 on still had a live frpc and answered 401). Not harmful tonight.
  - **Decision: restart the run right after the step 220 checkpoint lands** (both halves), with the config
    unchanged, so nothing but boot time is lost and the pool mints four new tunnels. Fallback path is well
    exercised (attempt 5 -> 7). Fits "kill, fix and relaunch"; changes nothing about what the run measures.
- 13:01:00 trainer `Step 220 | 5m 2s | Mismatch KL 0.0573` with the checkpoint complete (64 shards). Attempt 7
  final tally: steps 1-220 in 8h 17m of training, 220 steps, 2 checkpoints kept (`step_200`, `step_220`).
- 13:01:21 `scancel 837` (job gone 13:03:42). The orchestrator's exit did not leave a stray orchestrator-only
  checkpoint; `checkpoints/` is exactly `step_200` and `step_220`, both halves each.

### Attempt 9 (SLURM job 907), submitted 13:05, 16 nodes, resume from step 220

Command: `uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml` (config unchanged at `f7ceea189`; run-dir attempt
8 was the dry run). Three-way resolved-config check: all OK. Launcher: `Resuming from step 220, cleaning future
rollouts and broadcasts`. Same node set as attempt 7 (`prime-nebius-puku-h200-gpu-[005-006,013,015-016,018,020,
024,027,033,036,038,042,046,052,055]`), so weight loads should be page-cache warm on all 8 inference nodes.
Reason for the restart: the 12:40 tunnel outage (see above); a fresh launch mints four new tunnels.
Logs: `/home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_9/`.

#### Attempt 9 progress

- 13:05 reaper: `Successfully deleted 444 sandbox(es)` from attempt 7.
- 13:12:13 `Policy inference pool ready after 3m 52s` (all 8 nodes page-cache warm: `Loading weights took 27 s`).
- 13:13:46 trainer `Resuming from step 220 (total_tokens=213484544, total_samples=245)`; checkpoint load plus v220
  broadcast succeeded (`POST /update_weights 200 OK`). Total downtime 13:01 to 13:14, about 13 minutes. So a
  restart on warm nodes costs ~13 min, not the ~40 min a cold boot costs; the same node set is worth asking for.
- 13:19:40 orchestrator Step 221 `Reward 0.9844 | Error 0.0% | Cancelled 0.0%` and 13:22:02 trainer Step 221
  (`8m 16s`, includes refilling the rollout pipeline; `Mismatch KL 0.0589`). v221 update 200 OK. Interception pool
  back at 4 servers with fresh tunnels; zero harness failures in the first post-resume batch. Steady state again.
- 13:46-14:15 steps 222-239 at ~2-3 min each, reward 0.64-1.00, but **mismatch KL kept climbing: 0.059 (221),
  0.075 (231-235), 0.078 (236), 0.106 (237), 0.090 (238), 0.119 (239)**, with grad-norm spikes 0.34 (223), 0.16
  (227, 231), 0.22 (235), 0.28 (236), 0.19 (239) and the masked-token fraction 1.8% -> 3.4%. Entropy 0.24-0.43.
- 14:15 onward, **rollout throughput collapsed and the trainer starved**. Trainer's last step is 239 (14:15:20);
  orchestrator batch 240 crawled 8 -> 48 of 64 over 22 minutes. Diagnosis, in order of what I ruled out:
  - Not inference or router health: all 8 engines serving (~45-49 running, 0 waiting, 0 preemptions,
    ~1700 generated tok/s each), zero engine errors, router fine. Not the tunnel: only 2 x 504 all hour.
  - Not the concurrency controller as such: it restarted its inflight cap at 75 on resume and had ramped to
    416 (below attempt 7's 512), so the dispatcher sat at its permit ceiling; that explains dispatch pacing,
    not the collapse.
  - Not signal starvation: raw pass rate 64% (323 pass / 179 fail in 20 min), curriculum admits everything.
  - **Cause: runaway generations.** Inference `e2e_request_latency p99` went from 40-150 s to 870-1720 s at
    ~14:15. On one engine, 45 s of metrics showed 5 completions against 79k generated tokens (~16k tokens per
    completing request) at ~35 tok/s per request. One of 7 sampled live traces had a **65,716-token single
    turn**; another's reasoning tail reads "current hidden hidden interdependenciesholidays" (token salad).
    A minority of such turns (30-60 min each) hog engine slots, so normal turns slow to a crawl, completions
    fall from ~30/min to ~5/min, and the harness's OpenAI client starts raising `openai.APITimeoutError:
    Request timed out` (8 by 14:36, rising to 8-9 failures/min). Router traffic fell 10x (5000 -> 500
    lines/min). Completed episodes through step 239 still look normal (mean 4-13k output tokens, one at
    126k at step 238), so the damage is in-flight, not yet in the metrics.
  - Timing matches the trainer metrics: the KL/grad-norm spikes at 237-239 are the policy moving fast right
    before generation quality broke. Whether FP8 serving drift caused it or just amplified it, I cannot tell
    from here. The config has `num_output_tokens_weight = 0.0`, so nothing in the reward opposes long outputs.
- 14:37:27 **stopped the run (`scancel 907`)**, job gone 14:39:17. Reasoning: the job was alive but not
  producing training steps (22 min without one and slowing), every path forward is a recipe change (bf16
  serving, an output-length cap or penalty, lower lr, resume point), which the handoff reserves for Garrett,
  and the alternative was 16 nodes generating timeouts until the zero-output guard fired and held them for
  another hour of grace. `checkpoints/` is exactly `step_200` and `step_220`, both halves, 64 shards each; no
  stray orchestrator-only checkpoint was written. Deleted the 425 leftover sandboxes. No jobs of mine remain.
- **Resume recipe** (whatever Garrett changes): the bare `[resume]` picks `step_220` (KL 0.057, pre-collapse);
  `--resume.step 200` (KL 0.048) is the safer point if the drift is judged to have started earlier. Same node
  set gives a ~13 min boot. Run the three-way resolved-config check first.

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
- **The run collapsed at step ~237-240 and I stopped it at 14:37.** Mismatch KL went 0.025 (steps 1-100) ->
  0.045-0.05 (150-200) -> 0.06-0.08 (221-236) -> 0.10-0.12 (237-239), then generations ran away (65k-token
  turns), inference clogged, and the trainer starved. Full diagnosis in the 14:15 entry. Decisions that are
  yours: bf16 serving (recipe in the 11:38 entry), an output-length cap or non-zero `num_output_tokens_weight`,
  a lower lr, and whether to resume from `step_220` or `step_200`.
- The sandbox-to-inference tunnel (`prime_tunnel` / frps) is a single point of failure per interception server,
  and the pool has no health check: a broken tunnel became a black hole for new sessions (least-loaded picks
  it) and cost ~70% of rollouts until a run restart. Worth a tunnel health probe in the pool, or a retry on 5xx
  in the harness's model client, or both. Also: the pool leaks frpc processes when it resizes (20 alive for 4).

## GLM-4.5-Air comparison run (Garrett's request, 15:10)

Same task and recipe with the model swapped: `configs/advanced/glm-4.5-air/swe.toml` at `3e448e362`, a copy of the
DeepSeek config with only the model block changed (`zai-org/GLM-4.5-Air`, `attn = "flash_attention_3"`,
`cp_style = "ulysses"`, `renderer.name = "glm-4.5"`, `enable_return_routed_experts = true`; the DeepSeek-only indexer
exclusion and `block_size` dropped). Kept identical on purpose: 131k, batch 64 / group 8, AdamW 1e-6, length
penalty, ckpt every 20 keep 2, online `fp8_per_block` serving with `kv_cache_dtype = "fp8"` and DeepGEMM,
`gpu_memory_utilization = 0.75`, 8 trainer + 8 inference nodes, the 7200 s ready timeout. wandb: same project
(`primeintellect/deepseek-v4-flash`) so the two runs sit side by side, name suffixed `-glm45air`, tag `glm-4.5-air`.
Sandbox label `glm45air-swe`. Run dir `/home/garrett/prl_output_dir/glm45air-swe-131k`.

- The checkpoint lives only in the shared `/home/huggingface/hub` (412 GB with a complete PrimeRL conversion
  cache and `.prime-v1` marker, world-writable). Rather than switch `HF_HOME` and hit the lock-file problem,
  symlinked that one model directory into `~/.cache/huggingface/hub/`.
- Unit test (5 passed), dry run, and the three-way resolved-config check all pass; resolved runtime is `prime`
  with label `glm45air-swe`, renderer `glm-4.5`.

### Attempt 2 (SLURM job 915), submitted 15:25, started 16:07, 16 nodes

Command: `uv run rl @ configs/advanced/glm-4.5-air/swe.toml`. Queued on `(Resources)`: 46 nodes allocated to
others, 14 planned for this job, 1 idle. Starts when two more free up. Run-dir attempt 1 was the dry run.
Nodes: `prime-nebius-puku-h200-gpu-[013-016,018,020,024,036,038,042,046,050,052,055,057-058]`. GLM weights are cold
on every node (different model), so expect a longer first load than the warm DeepSeek resumes.
Logs: `/home/garrett/prl_output_dir/glm45air-swe-131k/logs/attempt_2/`.
- 16:06 job 915 started (SLURM estimate was 16:48). 16:16 **died at inference startup**: every replica's workers
  raised `torch._dynamo.exc.ObservedAssertionErrorError` inside `profile_run` -> `glm4_moe.py:602` `down_proj` ->
  `fp8_linear.apply_weights` -> `ops.cutlass_scaled_mm` -> `triton_scaled_mm` (`_custom_ops.py:867`). Log:
  `logs/attempt_2/inference/node_0.log`. Weight load itself was fine (337 s cold).
  - Cause: blockwise FP8 (128 x 128 blocks) needs every sharded weight dimension to be a multiple of 128.
    GLM-4.5-Air's layer-0 dense MLP has `intermediate_size = 10944` (10944 / 8 = 1368, not even 16-aligned, so
    the call falls to the Triton path whose block-shape assertion fires) and its shared expert has
    `moe_intermediate_size = 1408`, which TP=8 splits to 176. Routed experts stay whole under expert parallelism
    (1408 = 11 x 128) and attention shards cleanly (q 1536, kv 128, o 1536 per rank). DeepSeek V4 Flash never hit
    this because all of its dimensions tile. The `fp8_per_block` literal is the only quantization prime-rl's
    inference config accepts, so "same FP8" for GLM means blockwise with exclusions.
  - `scancel 915` at 16:18 (first attempt got `Connection reset by peer` from slurmctld; retry worked).
  - Fix: `quantization_config = { ignore = ["re:.*\\.layers\\.0\\.mlp\\..*", "re:.*\\.shared_experts\\..*"] }`,
    the same mechanism as the DeepSeek indexer exclusion (vLLM's online-quant `should_ignore_layer` takes
    `re:` regexes and expands fused `gate_up_proj` into its `gate_proj` / `up_proj` shards). Layer 0 and the 45
    shared experts run in bf16; the 128 x 45 routed experts and all attention stay FP8. Not a perfect match to
    the DeepSeek run, but the closest one that boots. Noted for Garrett below.

### Attempt 4 (SLURM job 918), submitted 16:24, started immediately, 16 nodes

Config at `e70a6fce6` (adds the two FP8 `ignore` patterns). Run-dir attempt 3 was the dry run. Same node set as
attempt 2, so GLM's weights are now page-cache warm. Logs: `/home/garrett/prl_output_dir/glm45air-swe-131k/logs/attempt_4/`.

#### Attempt 4 progress

- 16:27:29 replica 0 `Loading weights took 14.07 seconds` (warm). Profiling passed with the FP8 exclusions.
- 16:29:56 KV pool: `GPU KV cache size: 7,923,472 tokens, Maximum concurrency for 131,072 tokens per request: 60.45x`
  (6.4x the DeepSeek pool; GLM-4.5-Air is ~1/3 the parameters and has 8 KV heads).
- 16:34:30 `Policy inference pool ready after 8m 0s`; v0 `POST /update_weights 200 OK` shortly after.
- 16:37:01 `No admitted train payload after 64 finalized units (consecutive zero-output batch equivalents: 1/10)`
  two minutes into rollouts, and 560 of the first 574 episodes finished with reward 0 after 1-3 turns
  (`stop=agent_completed`). **Cause: GLM-4.5-Air calls tools the harness does not offer.** The harness exposes
  exactly `bash` and `edit`; decoding the sampled tokens of a one-turn episode shows a well-formed
  `<tool_call>read\n<arg_key>path</arg_key><arg_value>reconcile/github_org.py</arg_value></tool_call><|observation|>`.
  `parse_glm` (`deps/renderers/renderers/parsing.py`) validates names against the offered tools and silently
  drops unknown ones, mirroring vLLM's GLM parser, so the assistant message arrives with no tool call, the
  harness treats it as a final answer, and the episode ends. Not a quantization problem: the prose is coherent
  and the tool-call syntax is exact; the model is reaching for a Claude-Code-style `read`. DeepSeek V4 never
  did this. The guard counter reset once batch 1 shipped.
  - RL is already correcting it, so I left it alone: turns per episode 3.8 / 9.1 / 14.5 and reward 0.33 / 0.42
    / 0.27 over orchestrator steps 1-3, with 56-78% of episodes dropped as no_signal (all-zero groups).
- 16:38:44 **trainer Step 1** `4m 13s | Loss -0.0064 | Entropy 0.1826 | Mismatch KL 0.0044 | Grad. Norm 0.4696 |
  Peak Mem. 43.8 GiB`; steps 2-3 at 20-46 s, Peak Mem 33.6 GiB. Mismatch KL is ~6x lower than DeepSeek's
  0.025 at step 1 under the same FP8 serving (bf16 dense MLP and shared experts aside). Steps are trainer-cheap
  and rollout-bound as expected for a 106B model on 8 nodes.
- 16:44 step 10: reward 0.53, turns 15.8; 16:52 step 20: reward 0.86, turns 24.4. The tool-hallucination
  hurdle is gone within 20 steps. Trainer steps 20-60 s, mismatch KL 0.004-0.006, Peak Mem 34-49 GiB.
- 16:53:52 **first checkpoint, step 20**: `Step 20 | 2m 7s`, ~75 s of write, trainer half 1.2T (vs 3.2 TB for
  DeepSeek); both halves present.
- 17:13 hourly summary through step 40 (50 min of job time): steps 20-90 s each (~1 min average, 3x faster
  than DeepSeek), reward 0.23-0.33 for steps 1-8 then 0.5-0.89 from step 16 on, turns 4 -> 25-35, mismatch KL
  flat at 0.0043-0.0070, grad norm 0.09-0.47, Peak Mem 34-49 GiB. 6,130 episodes finished; 14 off-policy
  cancellations, 41 trace failures (40 `uv --script`, 1 sandbox 503). No-signal discards 55-62% per batch.
  Checkpoints `step_20`, `step_40` (1.2 TB each). Nothing new to escalate.
- 17:31:47 step 60 checkpoint saved (`Step 60 | 2m 12s`); `checkpoints/` now `step_40 step_60` so `keep_last = 2` cleanup
  works here too. Steps 41-60: reward 0.34-0.89, turns 21-45, KL 0.0045-0.0061, `Max Off-Policy` up to 23.
- 17:53:41 step 80 checkpoint saved (`Step 80 | 2m 27s`); `checkpoints/` holds `step_60`, `step_80`. Steps 61-80
  routine: reward 0.5-0.8, KL 0.005-0.006, Peak Mem 34-49 GiB.
- 18:05:12 the coordinate-list `harness 'bash' exited 1` burst again: 31 traces across ~25 tasks in two seconds,
  same stderr as the DeepSeek run's 09:56 burst (corrected there). Self-cleared.
- 18:06 hourly summary through step 92 (1h 42m of job time): ~1 min/step, reward 0.52-0.81 over steps 81-92,
  turns 27-38, mismatch KL 0.0044-0.0061, Peak Mem 34-49 GiB. 12,117 episodes finished; 55 off-policy
  cancellations, 122 trace failures (88 `uv --script`, 31 the burst, 2 OOM-137, 1 sandbox 503). Checkpoints
  `step_60`, `step_80`. At this pace step 200 lands around 20:00 and step 300 around 21:45.
- 18:15:41 **step 100 checkpoint saved** (`Step 100 | 2m 13s`, `Mismatch KL 0.0060 | Peak Mem. 49.0 GiB`);
  `checkpoints/` holds `step_80`, `step_100`. 1h 50m from launch to step 100 versus ~4h 40m for DeepSeek.
- 18:20-18:22 five `prime: failed to delete sandbox ...: HTTP 500: Failed to dispatch sandbox cleanup request`
  warnings from the env server. Cleanup-only: provisioning had zero failures, sandboxes came up at 50-150/min,
  the API listed 509 sandboxes in 4.5 s. Leaked sandboxes fall back on the 3600 s idle timeout.
- 18:38:36 step 120 checkpoint saved (`Step 120 | 3m 29s`); `checkpoints/` holds `step_100`, `step_120`.
  Steps 101-120: reward 0.66-0.88, turns 21-39, KL 0.0056-0.0062.
- 19:00:36 step 140 checkpoint saved (`Step 140 | 3m 33s`); `checkpoints/` holds `step_120`, `step_140`.
- 19:01 hourly summary through step 140 (2h 37m of job time): ~1 min/step, reward 0.47-0.89 over steps 121-140,
  turns 21-39, mismatch KL 0.0046-0.0062, Peak Mem 34-49 GiB, inference p99 request latency ~40 s (no runaway
  generations). 18,256 episodes finished; 98 off-policy cancellations, 128 trace failures (88 `uv --script`,
  31 the coordinate-list burst, 4 OOM-137, 5 assorted sandbox one-offs), 44 sandbox cleanup 500s (cleanup only).
- 19:12:01 one `prime sandbox provisioning failed: Connect RPC failed (unavailable): Service Unavailable` (2 total);
  sandbox creation and completions stayed at 50-115/min, so a blip, not a degradation. Cleanup 500s stopped
  after 19:02 (44 total); live sandboxes matched inflight (510 vs 512), so nothing leaked.
- 19:23:15 step 160 checkpoint saved (`Step 160 | 2m 18s | Mismatch KL 0.0045`); `checkpoints/` holds `step_140`,
  `step_160`. Steps 141-160: reward 0.53-0.89, turns 28-40.
- 19:44:48 step 180 checkpoint saved (`Step 180 | 2m 32s | Mismatch KL 0.0052`); `checkpoints/` holds `step_160`,
  `step_180`. Steps 161-180 routine: reward 0.53-0.88, turns 30-40, cancellations 5-7% of a batch at times.
- 20:08:01 **step 200 checkpoint saved** (`Step 200 | 2m 32s | Mismatch KL 0.0054 | Peak Mem. 49.1 GiB`);
  `checkpoints/` holds `step_180`, `step_200`. 3h 44m of job time to step 200 (DeepSeek needed ~7h 20m).
- 20:08 hourly summary through step 200: ~1 min/step, reward 0.55-0.88 over steps 181-200, mean reward by
  50-step window 0.58 / 0.70 / 0.70 / 0.73, turns 29-40, mismatch KL 0.0043-0.0062 with no trend, entropy
  0.18-0.26, grad norm 0.08-0.13, Peak Mem 34-49 GiB, inference p99 ~30 s. 24,982 episodes finished; 150
  off-policy cancellations (5-7% of some batches, the 1 min/step cadence makes the 32-step limit ~35 min),
  129 trace failures (no new classes), 2 provisioning blips, 44 cleanup 500s (all before 19:02).
- 20:28:12 step 220 checkpoint saved (`Step 220 | 2m 26s | Mismatch KL 0.0049`); `checkpoints/` holds `step_200`,
  `step_220`. Steps 201-220: reward 0.75-0.88, turns 30-39, one batch with 13.7% cancellations (step 210).
- 20:53:32 step 240 checkpoint saved (`Step 240 | 2m 37s | Mismatch KL 0.0064`); `checkpoints/` holds `step_220`,
  `step_240`. Steps 221-240: reward 0.59-0.88, turns 30-43.

## Open questions for Garrett (GLM run)

- The GLM comparison is not a perfect "same FP8": layer 0's dense MLP and every layer's shared expert run in bf16
  because their dimensions do not tile into 128 x 128 blocks under TP=8 (or at all, for 10944). Routed experts
  and attention are FP8 as in the DeepSeek run. If you want the shared experts in FP8 too, the option is TP=1
  with more replicas per node (1408 tiles at TP=1, 10944 never does), which changes the serving topology.
- wandb project is `deepseek-v4-flash` for side-by-side comparison; rename if you want a model-neutral project.
- GLM-4.5-Air's first action on the bash harness is usually a hallucinated `read`/`write`-style tool call, which
  the renderer drops (unknown tool), so ~70% of early episodes end after one turn with reward 0. RL is fixing it
  within a few steps, but a system-prompt note or exposing a `read` tool would make the comparison with
  DeepSeek fairer at step 0. Left unchanged since it alters the task.
