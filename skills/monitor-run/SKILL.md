---
name: monitor-run
description: How to monitor ongoing training runs — find output directories, check logs, diagnose performance, and inspect SLURM jobs. Use when asked to check on a run, debug training issues, or investigate performance.
---

# Monitor a Run

## Find the output directory

The output directory is set in the config (`output_dir`). To find it:

- **Local run**: check the resolved configs at `{output_dir}/configs/`
- **SLURM run**: check `squeue -u $USER` to find the job, then look at `{output_dir}/configs/`

## RL

### Check GPU allocation

#### Single-node

GPUs are assigned in order: inference first, then trainer, then teacher (if any).

```
GPU 0..N-1     → inference (vLLM)
GPU N..M-1     → trainer (torchrun)
GPU M..K-1     → teacher inference (optional)
```

The exact split is controlled by `deployment.num_infer_gpus`, `deployment.num_train_gpus`, and `deployment.num_teacher_gpus`. The orchestrator runs as a separate process (no GPU). Check the resolved configs at `{output_dir}/configs/` for the actual values.

#### Multi-node (SLURM)

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

Nodes from the SLURM allocation are split in order: inference nodes first, then trainer nodes.

```
Nodes 0..I-1   → inference (vLLM, first node of each replica also runs vllm-router)
Nodes I..I+T-1 → trainer (torchrun, rank 0 node also runs the orchestrator)
```

The node assignment is visible in the generated sbatch script at `{output_dir}/rl.sbatch`.

### Check logs

Log paths are consistent across deployment types — `logs/trainer.log` and `logs/inference.log` always exist (real files for local, symlinks for multi-node SLURM).

#### Local runs

```
{output_dir}/logs/
├── trainer.log                  # trainer stdout (rank 0 only)
├── orchestrator.log             # orchestrator stdout
├── inference.log                # vLLM inference server stdout
├── trainer/
│   └── torchrun/                # per-rank stdout/stderr (all ranks)
└── envs/
    ├── train/{env_name}/
    │   ├── env_server.log
    │   └── env_worker_{id}.log
    └── eval/{env_name}/
        └── ...
```

#### SLURM runs

```
{output_dir}/logs/
├── trainer.log                  -> trainer/node_0.log (symlink)
├── inference.log                -> inference/node_0.log (symlink)
├── orchestrator.log
├── trainer/
│   ├── node_0.log
│   ├── node_1.log
│   └── torchrun/               # per-rank stdout/stderr (all ranks)
├── inference/
│   ├── node_0.log
│   ├── node_1.log
│   └── router_0.log            # vllm-router per replica
└── envs/
    └── ...
```

### Check performance

#### Trainer

Check `{output_dir}/logs/trainer.log` (rank 0, node 0). For multi-node, per-node logs are at `logs/trainer/node_*.log`. Per-rank logs from all ranks are under `logs/trainer/torchrun/`.

Key metrics per step:
- `time/step` — total step time
- `time/wait_for_batch` — time waiting for the orchestrator to deliver a batch
- `time/forward_backward` — forward/backward pass time
- `time/broadcast_weights` — time broadcasting weights to inference servers
- `time/save_ckpt` — checkpoint save time

High `wait_for_batch` means the orchestrator is the bottleneck (slow rollouts, slow envs, or too few inference replicas).

#### Orchestrator

Check `{output_dir}/logs/orchestrator.log`.

Key metrics per step:
- `time/step` — total orchestrator step time
- `time/generate_completions` — rollout generation time
- `time/wait_for_ckpt` — time waiting for trainer checkpoint
- `time/update_weights` — weight update time
- `scheduler/async_level` — current async level
- `empty_rollouts/all` — fraction of empty rollouts
- `errored_rollouts/all` — fraction of errored rollouts

High `wait_for_ckpt` means the trainer is the bottleneck. The orchestrator logs when it pauses/resumes:
```
"Orchestrator paused: waiting for trainer process to complete checkpoint ..."
"Orchestrator resumed: checkpoint ... ready (after ...s)"
```

#### Env servers

Check `{output_dir}/logs/envs/train/{env_name}/env_server.log` and `{output_dir}/logs/envs/train/{env_name}/env_worker_{id}.log`.

Key things to look for:
- **Event loop lag**: server logs aggregate lag stats (min/mean/median/p90/p99/max) of itself and all workers periodically. check that neither is overloaded
- **Active task distribution**: check if tasks are distributed as expected across workers per-env and across envs. uneven distribution suggests some workers/envs are slower. heavily skewed distribution can indicate that a env is bottlenecking the trainer or has stopped being responsive.
