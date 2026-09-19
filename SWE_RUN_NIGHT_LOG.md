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

(none yet)
