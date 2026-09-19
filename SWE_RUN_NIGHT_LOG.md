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

## Open questions for Garrett

- KV pool is 11.55x concurrency per FP8 replica at 131k (1.51M tokens), not ~1.4x. Worth re-checking whether
  bf16 serving now fits too (the 0.74x figure may have predated `kv_cache_dtype = "fp8"`), which would reopen
  the FP8-vs-bf16 mismatch trade. Also 8 inference replicas may be more than the 8-node trainer can consume.

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
