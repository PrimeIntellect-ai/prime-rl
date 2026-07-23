---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the output dir and read the resolved configs at `{output_dir}/configs/` (start with `rl.toml`).
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{output_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{output_dir}/STATUS.md` (never overwrite):

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation, eval scores, env-specific metrics.
**Stability**: entropy, mismatch_kl, grad_norm — flag spikes.
**Performance**: trainer vs orchestrator step time, env lag, inference pressure.

**Notes**: anything unusual (errors, restarts, hangs). Omit if nothing notable.
```

In W&B, each project auto-gets an **"overview" saved view** (train / eval / stability / performance sections) on its first run — use it for a quick check instead of the auto-generated default workspace.

### Restarting a run

**Never restart unless the researcher explicitly asked.** Confirm the exact restart command and the conditions that warrant one.

**Never** run kill or launch commands from your own shell. Dispatch them to the tmux **Launcher** window so the researcher sees what was executed:

```bash
SESSION=$(tmux display-message -p '#S')
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `scripts/tmux.sh` launches the run with a `Launcher` window in the named tmux session. The Claude window receives the output dir and session name in its appended prompt — if either is missing, **ask** rather than guess.
- `{output_dir}/configs/` — resolved TOMLs (`rl.toml` has the full picture).
- `{output_dir}/logs/` — see below.
- `{output_dir}/rollouts/step_N/{train,eval}/` — saved rollout traces (see Traces below).

### Logs

```
{output_dir}/logs/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router_0.log           # vllm-router per replica (multi-node only)
└── envs/{train,eval}/{env_name}.log    # one log file per env
```

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {output_dir}/logs/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/{train,eval}/*.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

**Progress** — orchestrator log. Rollout metrics are keyed `{scope}/{subset}/<metric>/<stat>`: `scope` is `train/agg` (all train envs) or `train/<env>` (`eval/<env>` for eval); `subset` is `all` (every rollout) or `effective` (post-filter).

| Metric | Description |
|--------|-------------|
| `train/agg/all/reward/mean` | mean training reward (per env: `train/<env>/all/reward/mean`) |
| `train/agg/all/num_total_tokens/mean` | avg tokens per rollout (also `num_input_tokens`, `num_output_tokens`) |
| `train/agg/all/num_turns/mean` | avg turns per rollout (multi-turn only) |
| `train/agg/all/is_truncated/mean` | fraction truncated |
| `train/agg/all/has_error/mean` | fraction errored (per-type under `train/agg/all/error/<type>`; also `dispatcher/errored/{train,eval}`) |
| `train/<env>/all/metrics/<name>/mean` | env-specific metrics (e.g. pass rate) |
| `eval/<env>/all/{avg@k,pass@k}` | eval scores when configured |

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `masked_advantage_{positive,negative}/mean` | fraction of DPPO-masked tokens with +/- advantage |
| `optim/grad_norm` | spikes may precede divergence |

**Performance** — trainer and orchestrator step independently, so comparing step times shows who's waiting on whom.

| Source | Metric | Description |
|--------|--------|-------------|
| trainer | `time/step` | total trainer step |
| trainer | `time/wait_for_batch` | **high → orchestrator is bottleneck** |
| trainer | `time/forward_backward`, `time/broadcast_weights`, `time/save_ckpt` | phase timings |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU % |
| orchestrator | `time/step`, `time/save_ckpt` | phase timings |
| orchestrator | `time/wait_for_policy` | **high → trainer is bottleneck** |
| orchestrator | `dispatcher/off_policy_level_{mean,max}`, `dispatcher/inflight_{train,eval}`, `dispatcher/groups_in_flight`, `dispatcher/queued/eval` | dispatcher / async state |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Traces

```
{output_dir}/rollouts/step_N/{train,eval}/all/traces.jsonl        # appended per rollout as it completes
{output_dir}/rollouts/step_N/{train,eval}/effective/traces.jsonl  # written per finalized batch / eval epoch
```

JSONL files of `vf.Trace` records (training tensors excluded), one line per trace — a
multi-agent env's episode contributes several lines sharing one `info.episode_id`. `all`
gets every completed rollout the moment it arrives — errored, filtered, and never-batched
ones included — so it's crash-durable; `effective` gets the clean trainable subset that went
into the step's train batch (eval: the non-errored trainable epoch cohort; multiple eval envs
share the step file) — untrainable traces (a frozen judge's) appear only in `all`. Each record carries `run` (`{type, id, step}`; for eval, `step` is the trigger step),
`verifiers` (producing build), `agent` (model, sampling, harness, `name`, `trainable`), `ok`
(the success sentinel — `errors` alone keeps retry history even after a recovery), and
`runtime` (config + provisioned resource id, e.g. the sandbox id), plus `env_name`,
`group_id`, `episode_id`, and `policy_version` under `info`.

```bash
wc -l {output_dir}/rollouts/step_42/train/{all,effective}/traces.jsonl
jq '.rewards' {output_dir}/rollouts/step_42/train/effective/traces.jsonl
jq 'select(.ok | not) | {id, env: .info.env_name, runtime}' {output_dir}/rollouts/step_*/train/all/traces.jsonl
```

The binary batches consumed by the trainer still live at `{output_dir}/rollouts/step_N/train_rollouts.bin`, next to the trace subtrees.

### Common failure modes

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.

### Multimodal image offload checks

v1 multimodal RL offloads images exactly once, at verifiers ingress: every image
content part is rewritten to a `file://` asset under the run image directory
before rendering, and renderers/inference/trainer all work from those refs. A
`data:` URL reaching a renderer ("requires offloaded file:// image assets")
means ingress was bypassed.

- The image directory comes from `[multimodal].offload_dir` in the resolved
  config; unset, it defaults to a run-scoped path (`{output_dir}/assets/images`
  or the hosted `RUN_ID` path). The launcher exports it to the orchestrator as
  `VF_RENDERER_IMAGE_OFFLOAD_DIR` (protected — not settable via `env_vars`).
- While multimodal rollouts are in flight, image files should accumulate under
  that directory. Zero files means the offload path isn't being exercised or
  image preparation failed before request submission.
- Inference rejects bad refs with `invalid_mm_image_ref` 400s (hash mismatch,
  fingerprint mismatch, unreadable asset) — grep the inference log.
- Inference caches materialized refs and logs
  `mm materialize cache: hits=X misses=Y hit_rate=Z% bytes=A/B evictions=C`
  every 1000 lookups. Hit rate should climb after turn 1 of multi-turn
  multimodal rollouts; a stuck-at-zero hit rate with repeat images means the
  cache is disabled or thrashing (sized by `PRIME_RL_MM_MATERIALIZE_CACHE_GB`,
  default 2.0, `0` disables).
- The orchestrator raises on placeholder/token drift ("does not cover
  image-typed tokens") before a sample ships — treat any occurrence as a bug,
  not noise.
- Trainer metrics: `mm/images_materialized`, `time/mm_materialize`, and
  `mm/images_placeholdered`. A nonzero placeholder count means image files
  disappeared before materialization (the batch trains with zero-loss
  placeholders and logs "raw image materialization missing image(s)") — check
  whether something cleaned the offload directory mid-run.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::Orchestrator       (CPU-only)
│   └── Verifiers::EnvServer     (ZMQ env server per environment)
│       └── Verifiers::EnvWorker0..N
├── torchrun
│   └── PRIME-RL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.
