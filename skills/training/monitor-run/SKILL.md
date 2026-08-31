---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the run dir and read the resolved configs at `{run_dir}/configs/latest/resolved/` (start with `rl.json`, or `orchestrator.json` on local runs). Read the launch command from `{run_dir}/configs/latest/command.txt`. The launch TOML is copied verbatim to `{run_dir}/configs/latest/rl.toml`. The run dir is `{output_dir}/{run_name}` — `run.name` auto-generates as `<envs>--<model>--<short-id>`, so if you only know the output dir, pick the most recently modified subdirectory (`ls -t {output_dir} | head -1`) or read `run.name` from the launch command.
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{run_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{run_dir}/STATUS.md` (never overwrite):

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

**Never** run kill or launch commands yourself. Hand the researcher the exact command and let them run it; after a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `{run_dir}/configs/latest/` — the current attempt's command, launch TOML, and `resolved/` JSON files. Each launch stays under `configs/attempt_<n>/`.
- `{run_dir}/logs/latest/` — the current attempt's logs (each launch gets `logs/attempt_<n>/`; resumes never overwrite earlier attempts). See below.
- `{run_dir}/rollouts/step_{n}/{train,eval}/` — saved episodes (see Episodes below).

### Dashboard

`uv run dashboard [output_dir ...]` (default `outputs/`, or `$PRL_OUTPUT_DIR` if set; several dirs can be tracked at
once) serves a local web dashboard at `http://localhost:7788` with four views per run:
metrics (the W&B overview sections, read from `metrics.jsonl`), per-attempt config
files, a rollout trace viewer with a per-token advantage/logprob view, and merged
component logs. It only reads the run dirs — safe to run against a live run.
`--port`/`--host` pick the bind address; a taken port automatically bumps to the next
free one, so several dashboards run side by side without coordination. GPU deps live
behind the `gpu` extra, so `uv sync --extra dashboard && uv run dashboard` works
without the training stack (e.g. on a head node).

**Daemon (auto-start)**: launchers auto-start one dashboard per host per user and a
live one absorbs each new run's output dir automatically — see the `dashboard` skill
for discovery, kill/restart commands, and `--isolated`. The short version: the live
port can differ from 7788 (a taken port bumps), so read the discovery file:

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json   # {"pid": ..., "url": "http://localhost:<actual port>"}
ps aux | grep PRL::Dashboard                  # the daemon's process title
```

Verify liveness with `curl -sf <url>/api/runs` and hand the researcher the `url`.

### Logs

```
{run_dir}/logs/latest/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── evals.log                  # SFT online-eval evals stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router.log             # the single global router (multi-node only; single-node logs it in inference.log)
└── envs/{train,eval}/{env_name}.log    # one log file per env
```

SLURM batch logs are under `{run_dir}/launcher/logs/*job_*.log`.

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {run_dir}/logs/latest/{trainer,orchestrator,evals,inference}.log
grep -E "WARNING|ERROR" {run_dir}/logs/latest/envs/{train,eval}/*.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

**Progress** — orchestrator log. Rollout metrics mirror the episode/trace hierarchy, at two levels:

- `{scope}/{subset}/<metric>/<stat>` — episode-level facts only: the token/turn/branch counts, summed over an episode's traces.
- `{scope}/{subset}/<agent>/<metric>/<stat>` — every trace-level metric (reward, truncation, errors, timing, env metrics, curriculum admission, eval scores), keyed by agent name so seats never mix. Flat over that agent's traces: one sample is one trace, so an in-episode fan-out like n solvers contributes n samples.

`scope` is `train/agg` (all train envs) or `train/<env>` (`eval/<env>` for eval); `subset` is `all` (every rollout) or `effective` (admitted, clean, and trainable). Single-agent envs have one agent — usually `agent` — and one trace per episode, so both levels agree; multi-agent envs name each seat (`proposer`, `solver`, `judge`, …).

| Metric | Description |
|--------|-------------|
| `train/agg/effective/<agent>/reward/mean` | mean training reward for that agent (per env: `train/<env>/effective/<agent>/reward/mean`) |
| `train/agg/effective/num_total_tokens/mean` | avg tokens per episode, summed over its agents (also `num_input_tokens`, `num_output_tokens`) |
| `train/agg/effective/num_turns/mean` | avg turns per episode, summed over its agents |
| `train/<env>/effective/<agent>/num_turns/mean` | avg turns for that agent alone (also token counts, `num_branches`) |
| `train/agg/effective/<agent>/is_truncated/mean` | fraction of that agent's rollouts truncated |
| `train/agg/all/<agent>/has_error/mean` | fraction of that agent's rollouts errored (per-type under `train/agg/all/<agent>/error/<type>`; also `dispatcher/errored/{train,eval}`) |
| `train/agg/all/<agent>/is_trainable/mean` | fraction carrying a training signal — 0.0 for a frozen seat like a judge |
| `train/agg/all/<agent>/is_admitted/mean` | fraction accepted by the source curriculum; per-source counters and custom policy metrics live under `curriculum/<env>/` |
| `train/<env>/effective/<agent>/metrics/<name>/mean` | env-specific metrics for that agent (e.g. pass rate) |
| `train/<env>/effective/<agent>/timing/agent/model/mean` | model vs harness share of that agent's phase |
| `eval/<env>/effective/<agent>/{avg@k,pass@k}` | eval scores for that agent, when configured |

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `is_masked/mean` | fraction of tokens masked by the IPO trust region |
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
| orchestrator | `dispatcher/off_policy/{mean,max}`, `dispatcher/inflight/{train,eval}`, `dispatcher/queued/eval` | dispatcher / async state |
| orchestrator | `off_policy/{mean,max}`, `off_policy/{in_flight,in_queue}/{mean,max}`, `off_policy/dropped` | per-step staleness of trained rollouts |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8100/metrics | grep -E "num_requests|gpu_cache_usage"  # engine port (8000 is the router)
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Episodes

```
{run_dir}/rollouts/step_{n}/{train,eval}/all/traces.jsonl        # appended per episode as it completes
{run_dir}/rollouts/step_{n}/{train,eval}/effective/traces.jsonl  # written per finalized batch / eval epoch
```

JSONL files of native `vf.Episode` records (training tensors excluded), one line per episode.
`all` gets every completed episode as it arrives — including trace-less failures,
curriculum-rejected work, and work that never enters a batch — so it is crash-durable.
`effective` contains the admitted clean trainable traces grouped into their original episodes
(eval: the non-errored trainable epoch cohort). Each record carries its provenance at the
episode level: `env` (`id` plus the orchestrator's `name`), full `task`, `group` (`id`),
and `run`. Training-run records discriminate train/eval work and include dispatch
step plus an optional live-policy version span. Traces retain their own task,
verifiers, agent, and runtime fields.

Each trainable trace also carries `info.advantage`, the scalar view of its assigned credit.
It is only populated in `train/effective` (written at ship time, after credit assignment).
It resolves to `null` everywhere else — in `all` (written on arrival, before the sink scores
it), for eval traces, and for algorithms that assign no rl credit (sft, opd, opsd). `null` is
distinct from `0.0`: the latter means the trace *was* scored and its group cancelled out, so it
carried no gradient.

```bash
wc -l {run_dir}/rollouts/step_42/train/{all,effective}/traces.jsonl
jq '.traces[].rewards' {run_dir}/rollouts/step_42/train/effective/traces.jsonl
jq 'select(.ok | not) | {id, env: .env.id, errors}' {run_dir}/rollouts/step_*/train/all/traces.jsonl
# traces that scored but carried no gradient (group rewards were all equal)
jq '.traces[] | select(.info.advantage == 0)' {run_dir}/rollouts/step_42/train/effective/traces.jsonl
```

The batches consumed by the trainer are shipped over ZMQ by default, so nothing binary is written. With `rollout_transport.type = "filesystem"` they land at `{run_dir}/rollouts/step_{n}/rank_<rank>.bin` (one packed micro-batch file per trainer DP rank), next to the episode subtrees.

### Common failure modes

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRL::Launcher
├── PRL::Inference          (vLLM server, GPU 0)
├── PRL::EnvServer          (verifiers' ZMQ env server, run in-process; one per train/eval source)
│   └── Verifiers::EnvWorker0..N
├── PRL::Orchestrator       (CPU-only; connects to each env server)
├── torchrun
│   └── PRL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.
