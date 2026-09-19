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
