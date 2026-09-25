# Handoff: get the DeepSeek V4 Flash SWE run alive overnight

You are taking over an operational task, not a design task. Garrett is asleep and will review in
the morning. You are autonomous until then.

## Goal

**A minimal working run.** Success is that `configs/advanced/deepseek-v4-flash/swe.toml` is still
running and producing training steps when he wakes up. Not throughput, not reward, not a tuned
recipe. If you have to shrink something to keep it alive, shrink it and write down what and why.

Failure to reach a working state is an acceptable outcome *if* the night log explains what you
tried and what blocked you. Silently idling 16 nodes is not.

## Your authority

- **Edit anything you need**, including `deps/` submodules and `.venv`-adjacent pins, to get this
  running. Prefer the smallest change that works.
- **Submit, kill, fix and relaunch** as many times as you need.
- **Up to 16 nodes**, matching the current config (8 trainer + 8 inference). Do not exceed 16.
  Release the allocation whenever you are not actively booting or running (`scancel <jobid>`), and
  check `sinfo` before resubmitting; other users share this cluster.
- **Never `git push`.** Commit freely on `feat/ds-v4-fp8-rl`; pushing needs verbal approval.
- Do not modify anything under `skills/` without asking.
- Do not change `optimization_dtype` or `reduce_dtype` anywhere. They are load-bearing numerical
  knobs and this run does not need them touched.

## Log your work

Append to `SWE_RUN_NIGHT_LOG.md` as you go. Garrett reads this first in the morning, so it should
stand alone. For each attempt record:

- timestamp, SLURM job id, node count, and the exact command
- what changed since the previous attempt, and the commit SHA if you committed
- how it failed, with the actual error text and the log path
- the reasoning behind the fix you chose, including options you rejected
- anything surprising, even if unrelated, and anything you could not explain

Commit the log periodically so it survives. Keep a short **Open questions for Garrett** section at
the bottom for judgement calls you made that he might want to reverse.

## Launch

```
uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml
```

That is the whole command. `HF_HOME` and every JIT cache path are already pinned inside the config,
so no shell prefix is needed. A `[slurm]` table is present, so this submits a batch job rather than
running locally; it takes about 4 minutes to resolve the `scaleswe` environment before
`Submitted batch job N` appears. Run directory is `/home/garrett/prl_output_dir/dsv4-swe-131k`.

Logs:
```
tail -F /home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_1/trainer.log
tail -F /home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_1/inference.log
tail -F /home/garrett/prl_output_dir/dsv4-swe-131k/logs/attempt_1/orchestrator.log
```
Note the attempt number increments on each launch into the same run dir. Per-node logs are under
`logs/attempt_N/{trainer,inference}/node_*.log`, and the orchestrator log is where config
validation failures land.

## State at handoff

Config is committed at `caf74354b` and has been submitted once. `tests/unit/test_configs.py` is
green (196 passed).

**Job 819 was submitted and died during orchestrator startup**, then the allocation was cancelled.
Cause, now fixed in `caf74354b`: the `deepseek-v4` renderer defaults `drop_thinking = true`, which
implies `thinking_retention = "tool_cycle"` and conflicted with the explicit `"all"`.

**Read this next part carefully, because it will bite you again.** That failure passed both
`--dry-run` and the config unit tests. The orchestrator, trainer and inference each re-validate
their own resolved JSON in a subprocess, and in that round trip every serialized default counts as
*explicitly set*, so validators keyed on `__pydantic_fields_set__` fire at launch but not at
dry-run time. **Before every relaunch, run this instead of trusting `--dry-run`:**

```
uv run rl @ configs/advanced/deepseek-v4-flash/swe.toml --dry-run >/dev/null 2>&1
uv run python -c "
import json
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.configs.inference import InferenceConfig
base='/home/garrett/prl_output_dir/dsv4-swe-131k/configs/latest/resolved'
for name, cls in [('orchestrator',OrchestratorConfig),('trainer',TrainerConfig),('inference',InferenceConfig)]:
    try: cls(**json.load(open(f'{base}/{name}.json'))); print(name,'OK')
    except Exception as e: print(name,'FAIL',str(e).splitlines()[:4])
"
```
All three must print OK. This costs a minute and would have saved a 16-node submission.

## What to watch, in order

Boot is roughly 30 minutes: ~4 min to submit, then FP8 quantizes every weight at load. Three gates:

1. **Inference KV pool.** In `inference.log`, look for `GPU KV cache size` / `Maximum concurrency`.
   It must clear **131,072 tokens**, or a single full-length request cannot be scheduled. This is
   the number the whole FP8 decision rests on and it has not been measured at this `max_model_len`.
   If it comes in short, raise `gpu_memory_utilization` toward 0.92 first; if still short, drop
   `seq_len` and `max_model_len` together to 98304 and record that you did.
2. **Trainer `Peak Mem.` at step 1.** Expect roughly 66 GiB of weight-shaped state on a 141 GiB
   card at 8 nodes, plus ~24 GiB broadcast transient and ~11 GiB activations. **This is the
   genuinely unproven number: nothing has ever run this model above `seq_len = 4096`.** If it OOMs,
   the ladder in order is `fused_lm_head_token_chunk_size` 1024 to 512, then `cp = 4` to `8` (valid
   values are `{1,2,4,8}` given `ep = 8`), then `seq_len` down, then more trainer nodes, then
   `full_offload` as a last resort.
3. **First completed step** in `trainer.log`, and `time/wait_for_batch` vs `time/wait_for_policy`
   in `monitors/file/metrics.jsonl` to see whether it is rollout-bound or trainer-bound.

## Known hazards

- **`scaleswe` + prime sandbox is unexercised with this model and renderer.** Nothing about that
  path has been tested. If the run dies early and it is not memory, look here first.
  `PRIME_API_KEY` is set in the environment. The `pre_run_command` reaps sandboxes labelled
  `dsv4-swe`, which must match `env.agent.runtime.labels` on every source.
- **`env.agent.runtime.type` must stay absent.** SWE tasksets set a per-task container image and
  the `subprocess` runtime raises unconditionally on that. The resolved config should show
  `runtime.type = "prime"`.
- **`cp_style` must stay unset.** DeepSeek V4 is ring-only; `"ulysses"` is a startup `ValueError`
  despite what `docs/scaling.md` recommends.
- **Two guards will kill a long run**: `MAX_CONSECUTIVE_EMPTY_BATCHES = 10`
  (`src/prime_rl/orchestrator/orchestrator.py:93`) and
  `MAX_CONSECUTIVE_ZERO_OUTPUT_BATCH_EQUIVALENTS = 10`
  (`src/prime_rl/orchestrator/train_sink.py:28`). Both are module constants, not config. They fire
  on empty or zero-output batches, meaning rollouts are *failing*, not merely slow. Deliberate
  decision: **leave them alone.** Nothing auto-restarts, so a wedge idles 16 nodes until morning
  whereas the guard exits and frees them. If you do raise them, say why in the log and only after
  you understand the failures they were catching.
- **Checkpoints are ~3.2 TB** with `keep_last = 2`, so ~6.4 TB peak on disk; `/home` has ~226 TB.
  Write time is unmeasured. `interval = 20`.
- **A crash can break bare `[resume]`.** The orchestrator writes a checkpoint on every exit while
  the trainer only writes on interval boundaries, so a crash can leave an orchestrator-only
  `step_N` that `resolve_latest_ckpt_step` picks and the trainer then fails to load. Before
  resuming, confirm both `trainer/` and `orchestrator/` exist under the latest
  `checkpoints/step_N/`; otherwise pass `--resume.step <last complete step>`.
- **Nothing auto-restarts.** `srun --kill-on-bad-exit=1` reaps every node on any component's
  non-zero exit. That is why a failed attempt is cheap: the allocation is released.

## Decisions already taken, with reasons. Do not relitigate without cause.

- **Online FP8 serving, with the Lightning Indexer excluded from quantization**
  (`quantization_config = { ignore = ["re:.*indexer.*"] }`). At 131k a bf16 KV pool gives under 1x
  concurrency, so FP8 is close to required. The indexer exclusion matters *only* above ~2048
  tokens: below that a short-context shortcut selects all candidates and quantizing `indexer.wq_b`
  is bitwise irrelevant; above it, measured 0.31-0.36% of tokens move probability by more than 0.3,
  worst token KL 47. Unlike routed experts, the indexer's top-512 selection is not replayed to the
  trainer. Note the `ignore` list takes exact strings or `re:`-prefixed regexes only; a bare `*`
  glob silently matches nothing.
- **8 trainer + 8 inference nodes.** 4 trainer nodes is the arithmetic minimum (measured 133.9 GiB
  peak of 141) and violates "sure not to OOM". 8 inference replicas because an FP8 replica holds
  only ~1.4 full-length requests at 131k, and slow rollouts are the likeliest way this trips the
  zero-output guard.
- **AdamW at 1e-6**, Garrett's call. Repo precedent for SWE; Muon has never been exercised on this
  model. Do not switch optimizers.
- **No eval source.** Removes a separate code path and an hour-long SWE-Bench block at step 0 from
  a bring-up run. Add it once the 131k path is proven.
- **`batch_size = 64`, `group_size = 8`.** Smaller than the plan's 256/16 so steps actually
  complete; this is a bring-up run.
- Cache paths and `HF_HOME` are hardcoded to Garrett's user with a `TODO`. The inherited
  `HF_HOME=/home/huggingface` is not writable. `TILELANG_CACHE_DIR` is included deliberately: it
  defaults to `~/.tilelang/cache` on shared `/home`, nothing else in the repo redirects it, and
  both the trainer's sparse-attention kernels and inference's mHC pre-norm compile through TileLang.

## Escalate vs. fix

Fix autonomously: anything mechanical (config validation, memory ladder, cache paths, dependency
pins, submodule edits, taskset plumbing). Record it.

Do not decide alone; write it in **Open questions** and pick the conservative option meanwhile:
changing the optimizer or learning rate, turning FP8 off entirely, exceeding 16 nodes, raising the
consecutive-failure guards, or anything that changes what the run is measuring rather than whether
it runs.

## Background reading, in priority order

- `SWE_131K_PLAN.md` — the full design rationale, memory arithmetic, and per-setting justification.
- `FP8_MISMATCH_RESULTS.md` — why FP8 was chosen and what it costs; the indexer numbers.
- `AGENTS.md` — repo conventions. `uv run` always, never bare `python`.
- `skills/training/start-run` and `skills/training/monitor-run` — launch and monitoring mechanics.
