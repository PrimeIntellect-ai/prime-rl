---
name: monitor-run
description: How to monitor ongoing training runs — find output directories, check logs, diagnose performance, and inspect SLURM jobs. Use when asked to check on a run, debug training issues, or investigate performance.
---

# Monitor RL Run

## Runbook

### On launch

Immediately gather context and write a summary of the run into `{output_dir}/STATUS.md`:

1. Identify the output directory and read the resolved configs to understand the experiment (model, envs, hyperparameters, deployment details).
2. Make sure that the run started successfully and that all processes are alive.
3. Read the logs and note the current training step and health.

### Recurring check-ins

After the initial overview, schedule recurring check-ins. By default, check in every **1 hour** (the researcher can override this).

At each check-in:

1. Check that all processes are alive.
2. Read the logs — look for errors, warnings, hangs, or degraded performance
3. Note the current training step, key metrics, and checkpoint progress.
4. **Append an entry to `{output_dir}/STATUS.md`**:

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation rate, eval scores (if available), notable env-specific metrics.
**Stability**: entropy, mismatch KL, grad norm — flag any spikes or concerning trends.
**Performance**: trainer and orchestrator step times, who is waiting on whom, env server lag, inference pressure.

**Notes**: (anything unusual — errors, restarts, hangs, etc. Omit if nothing notable.)
```

Always append — never overwrite previous entries.

### Restarting a run

**IMPORTANT**: Never restart a run unless you were explicitly instructed by the researcher. If you were given permission, make sure to ask the researcher for the exact command to resume a run and under what conditions a restart is necessary.

**IMPORTANT**: Never run kill or launch commands directly from your shell. Instead, send them to the tmux **Launcher** window so the researcher can see exactly what was executed.

Use `tmux send-keys` to dispatch commands to the Launcher pane:

```bash
# Get your current tmux session name
SESSION=$(tmux display-message -p '#S')

# Send a command to the Launcher window
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify that all processes are back up and healthy before resuming periodic check-ins. Check the process tree and tail the logs to confirm the run is making progress again.

---

## Reference

### Output directory and tmux session

The output directory and tmux session name are typically provided by the researcher in the appended system prompt (see `scripts/tmux.sh` — the Claude window is launched with this context). If not provided, **ask the researcher** which output directory to monitor and which tmux session the run is in.

The tmux session contains the **Launcher** window where the researcher runs launch commands — this is where you should send any restart commands (see [Restarting a run](#restarting-a-run)).

Once you have the output directory, the resolved configs are at `{output_dir}/configs/`.

### Configs

The launcher writes resolved configs as TOML files to `{output_dir}/configs/`. Read `rl.toml` to get the full picture of the experiment (model, envs, hyperparameters, wandb, deployment).

### Logs

Logs are usually the most informative place to monitor a run.

```
{output_dir}/logs/
├── trainer.log                  # trainer stdout (rank 0)
├── orchestrator.log             # orchestrator stdout
├── inference.log                # vLLM inference server stdout
├── trainer/
│   ├── node_*.log               # per-node logs (multi-node only)
│   └── torchrun/                # per-rank stdout/stderr (all ranks)
├── inference/
│   ├── node_*.log               # per-node logs (multi-node only)
│   └── router_0.log             # vllm-router per replica (multi-node only)
└── envs/
    ├── train/{env_name}/
    │   ├── env_server.log
    │   └── env_worker_{id}.log
    └── eval/{env_name}/
        └── ...
```

Usually it's sufficient to tail `trainer.log`, `orchestrator.log`, and `inference.log`. For debugging, it may be necessary to check the per-node logs (`node_*.log`) or per-rank trainer logs under `torchrun/`.

```bash
tail {output_dir}/logs/trainer.log                     # training progress — kl mismatch, entropy, grad norm, trainer step time
tail {output_dir}/logs/orchestrator.log                # orchestrator — reward, rollouts, env execution, inference step time
tail {output_dir}/logs/inference.log                   # inference — completed HTTP requests, engine stats, OOM errors
tail {output_dir}/logs/envs/train/*/env_server.log     # env server — aggregated stats across its workers (lag, task distribution)
tail {output_dir}/logs/envs/train/*/env_worker_*.log   # env workers — individual env logs
```

All logs use loguru with the format `HH:mm:ss  LEVEL message`. Log levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`. To scan for problems:

```bash
grep -E "WARNING|ERROR" {output_dir}/logs/trainer.log
grep -E "WARNING|ERROR" {output_dir}/logs/orchestrator.log
grep -E "WARNING|ERROR" {output_dir}/logs/inference.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_server.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_worker_*.log
```

### Metrics

All metrics below are logged to the console. The source column indicates which log file to check.

#### Progress

These tell you whether the model is learning and how the run is progressing.

| Metric | Source | Description |
|--------|--------|-------------|
| `reward/{all,env}/mean` | orchestrator | mean training reward |
| `seq_len/{all,env}/mean` | orchestrator | average sequence length in tokens |
| `num_turns/{all,env}/mean` | orchestrator | average turns per rollout (ignore for single-turn envs) |
| `is_truncated/{all,env}/mean` | orchestrator | fraction of truncated rollouts |
| `empty_rollouts/{all,env}` | orchestrator | fraction of empty rollouts |
| `errored_rollouts/{all,env}` | orchestrator | fraction of errored rollouts |
| `metrics/{env}/{metric}` | orchestrator | env-specific metrics (e.g. pass rate for unit tests) |
| `eval/{env}/{avg@k,pass@k}` | orchestrator | eval scores (if eval is configured) |

#### Stability

These tell you whether training is healthy or diverging.

| Metric | Source | Description |
|--------|--------|-------------|
| `mismatch_kl/mean` | trainer | KL divergence between trainer and (old) inference policy  |
| `entropy/mean` | trainer | policy entropy |
| `optim/grad_norm` | trainer | gradient norm — spikes may precede divergence |

#### Performance

These tell you how fast the run is and where the bottlenecks are. Trainer and orchestrator step independently — compare their step times to identify who is waiting on whom.

**Trainer** (in `trainer.log`):

| Metric | Description |
|--------|-------------|
| `time/step` | total trainer step time |
| `time/wait_for_batch` | time waiting for orchestrator to deliver a batch — **high = orchestrator is the bottleneck** |
| `time/forward_backward` | forward/backward pass time |
| `time/broadcast_weights` | time broadcasting weights to inference |
| `time/save_ckpt` | checkpoint save time |
| `perf/throughput` | tokens/s throughput |
| `perf/mfu` | model FLOPs utilization % |

**Orchestrator** (in `orchestrator.log`):

| Metric | Description |
|--------|-------------|
| `time/step` | total orchestrator step time |
| `time/generate_completions` | rollout generation time |
| `time/wait_for_ckpt` | time waiting for trainer checkpoint — **high = trainer is the bottleneck** |
| `time/update_weights` | weight update time |
| `scheduler/async_level` | current async level |
| `scheduler/inflight_rollouts` | number of in-flight rollouts |

**Env servers** (in `envs/train/{env_name}/env_server.log`):

| Metric | Description |
|--------|-------------|
| event loop lag (min/mean/p90/p99/max) | server and worker lag stats, logged periodically |
| active task distribution | per-worker and per-env task counts  |

**Inference** (in `inference.log`):

vLLM logs completed HTTP requests and occasionally engine stats. For live inference metrics, query the vLLM metrics endpoint directly:

```bash
# Get vLLM Prometheus metrics (num running/queued reqs, KV cache usage, etc.)
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
```

Key vLLM metrics to watch:
- `vllm:num_requests_running` — requests currently being processed
- `vllm:num_requests_waiting` — requests queued waiting for KV cache space
- `vllm:gpu_cache_usage_perc` — KV cache pressure (approaching 1.0 = requests will queue)

### Rollouts

Plain-text rollouts (`verifiers` format) are saved every step alongside the binary training batch inside the run directory. For single-run (local) runs this is typically `{output_dir}/run_default`.

```
{output_dir}/{run_dir}/rollouts/
└── step_{N}/
    ├── train_rollouts.jsonl   # all train rollouts
    ├── eval_rollouts.jsonl    # all eval rollouts (only present when eval ran)
    └── train_rollouts.bin     # binary-encoded training batch (consumed by the trainer)
```

Each line in the `.jsonl` files is a JSON-serialized `vf.RolloutOutput` dict with fields like `example_id`, `task`, `prompt`, `completion`, `reward`, `trajectory`, `metrics`, etc. To inspect rollouts for a given step:

```bash
# Count rollouts at a step
wc -l {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl

# Preview first rollout (pretty-printed)
head -1 {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl | python -m json.tool

# Extract rewards
jq '.reward' {output_dir}/{run_dir}/rollouts/step_42/train_rollouts.jsonl
```

### Errors and warnings

As part of every check-in, grep all logs for `WARNING` and `ERROR` level messages. Pay special attention to env server and env worker logs — these are the most common source of issues since they run user-provided code.

Common things to look for:
- **Env workers**: exceptions in environment execution, timeouts, sandbox errors, OOM kills
- **Orchestrator**: empty/errored rollout spikes, weight broadcast failures, checkpoint errors
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients
- **Inference**: NCCL/CUDA errors, OOM, request timeouts

A small number of warnings is normal (e.g. occasional env timeouts). Escalate to the researcher if you see errors that are persistent, increasing, or affect a large fraction of rollouts.

### Processes

All processes have custom process names, making them easy to identify in `ps`, `htop`, and `pstree`. Use this in case you need to debug a process that is not responding or behaving unexpectedly.

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::Orchestrator       (CPU-only, data/scheduling)
│   ├── Verifiers::EnvServer     (ZMQ env server per environment)
│   │   ├── Verifiers::EnvWorker0
│   │   ├── Verifiers::EnvWorker1
│   │   └── ...
│   └── ...
├── torchrun
│   └── PRIME-RL::Trainer        (RL trainer, GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are distributed across separate nodes. Use `srun` or `ssh` to inspect processes on other nodes directly.

