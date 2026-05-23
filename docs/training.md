# Training

This page covers everything you need to launch, observe, checkpoint, and recover a `prime-rl` training run — RL, SFT, and the related on-policy distillation mode. For multi-node and cluster layouts, see [Scaling](scaling.md). For the loss math and algorithm knobs, see [Algorithms](algorithms.md).

## Table of Contents

- [Entrypoints](#entrypoints)
- [RL training](#rl-training)
  - [Launch](#launch)
  - [Useful CLI flags](#useful-cli-flags)
  - [What each process does at runtime](#what-each-process-does-at-runtime)
  - [Key knobs](#key-knobs)
- [SFT training](#sft-training)
  - [Dataset format](#dataset-format)
  - [Launch](#launch-1)
  - [SFT-specific knobs](#sft-specific-knobs)
- [Training modes (RL / OPD / SFT-via-orchestrator)](#training-modes-rl--opd--sft-via-orchestrator)
- [Evaluations](#evaluations)
- [Checkpointing](#checkpointing)
  - [Enabling checkpoints](#enabling-checkpoints)
  - [Resuming a run](#resuming-a-run)
  - [Saving HF weights for serving](#saving-hf-weights-for-serving)
- [Observability](#observability)
  - [Log files](#log-files)
  - [Console output and the tmux helper](#console-output-and-the-tmux-helper)
  - [Weights & Biases](#weights--biases)
  - [Platform monitoring](#platform-monitoring)
  - [Prometheus and BetterStack](#prometheus-and-betterstack)
- [Important metrics](#important-metrics)
- [Rules of thumb](#rules-of-thumb)

## Entrypoints

| Command | Purpose | Notes |
|---|---|---|
| `uv run rl` | Co-launches inference + orchestrator + trainer on one node | The default for any single-node RL run. Mirrors a `[trainer]` + `[orchestrator]` + `[inference]` TOML. |
| `uv run sft` | Supervised fine-tuning on a HF dataset | Launches torchrun internally; never call torchrun directly. |
| `uv run inference` | vLLM server | Always use this entrypoint over `vllm serve` — it adds `/update_weights`, `/load_lora_adapter`, and `/init_broadcaster`. |
| `uv run trainer` | Standalone trainer process group | Use only when launching the trainer separately from the orchestrator (e.g. multi-node RL without the `rl` wrapper). |
| `uv run orchestrator` | Standalone orchestrator process | Pair with a separately-launched trainer + inference. |

`rl` is a convenience wrapper — it parses one merged TOML, splits it across `[trainer]` / `[orchestrator]` / `[inference]` tables, picks GPUs, sets up logging, and spawns the three children. Standalone entrypoints exist for the multi-node case where each process lives on a different host.

## RL training

### Launch

The minimal RL run trains an SFT-warmed `Qwen3-0.6B` on the `reverse-text` task — the env is bundled with the `verifiers` submodule, so nothing else needs to be installed. From the project root, on two GPUs (one for inference, one for the trainer):

```bash
uv run rl @ examples/reverse_text/rl.toml \
  --wandb.project my-project \
  --wandb.name reverse-text-smoke \
  --ckpt
```

GPU placement: by default `rl` uses 1 trainer GPU and 1 inference GPU on the local node. To run on (say) 8 GPUs with 4 inference + 4 trainer, set the deployment counts:

```bash
uv run rl @ rl.toml \
  --deployment.num-infer-gpus 4 \
  --deployment.num-train-gpus 4 \
  --inference.parallel.dp 4
```

The launcher assigns physical GPUs from `CUDA_VISIBLE_DEVICES` (or all visible GPUs if unset) — inference takes the first `num_infer_gpus`, the trainer takes the next `num_train_gpus`, and any teacher gets the remainder. To run on a specific subset of physical GPUs, pin `CUDA_VISIBLE_DEVICES` before launching.

For multi-node and SLURM, see [Scaling § RL training](scaling.md#rl-training).

### Useful CLI flags

Commonly-used flags every RL launch should know about:

| Flag | What it does |
|---|---|
| `--ckpt` | Enable end-of-training checkpoint. See [Checkpointing](#checkpointing) for interval / keep-last / resume variants. |
| `--wandb` | Enable Weights & Biases logging with defaults. Pair with `--wandb.project` / `--wandb.name`. |
| `--orchestrator.prime-monitor` | Register the run on the Prime Intellect platform (Lab) and stream metrics there. See [Platform monitoring](#platform-monitoring). |
| `--clean-output-dir` | Wipe `<output_dir>` before starting. Useful when re-running an experiment with the same name during iteration. |
| `--output-dir outputs/<name>` | Per-run output directory. Always set this when running more than one experiment in parallel. |
| `--max-steps N` | Stop after `N` trainer steps. Overrides whatever the config sets. |
| `--dry-run` | Resolve + validate the full config, write per-process TOMLs to `<output_dir>/configs/`, and exit without launching. The fastest way to debug a misbehaving config. |

### What each process does at runtime

- **Inference** (vLLM) holds the current policy and serves OpenAI-compatible completions. Receives a new HF checkpoint via `POST /update_weights` after each trainer step (or batched into one update per `max_async_level` steps).
- **Orchestrator** samples a prompt batch from the configured `[[orchestrator.train.env]]` envs, drives them against the inference server (multi-turn, tool calls, etc.), packs the completed rollouts into a binary batch, writes it under `outputs/rollouts/step_N/`, and notifies the trainer. The orchestrator talks to one **env server** per train/eval env (sidecar `vf.EnvServer` subprocess by default), and each env server holds a pool of **env workers** that run user code concurrently — that's where most rollout-time CPU work lives.
- **Trainer** waits for the binary batch, runs forward/backward/optimizer step under FSDP2, writes new weights to the broadcast transport, and signals the orchestrator that step `N+1` is in flight.

The orchestrator is the only stateful CPU process; the trainer is GPU-bound; the inference server is stateless apart from KV cache. On restart the orchestrator pushes the latest checkpoint into inference automatically — you don't need to checkpoint inference state.

### Key knobs

The orchestrator owns the data-side knobs that most directly shape what the trainer sees. For trainer-side parallelism, sampling, optimizer, and loss knobs see [Scaling](scaling.md) and [Algorithms](algorithms.md); for the full field reference see [Reference](reference.md).

| Knob | What it controls |
|---|---|
| `orchestrator.batch_size` | Prompts per trainer step. |
| `orchestrator.rollouts_per_example` | Group size — rollouts generated per prompt. Used for advantage normalization and pass@k estimation. |
| `orchestrator.max_off_policy_steps` | How many distinct policies may have contributed to one rollout before it gets discarded (default 8). The main throughput-vs-noise dial on long agentic rollouts — bump for throughput, lower for tighter on-policyness. Watch `errored_rollouts` and `mismatch_kl/all/mean` when tuning. |
| `orchestrator.training_mode` | Picks the training-mode dispatch: `rl` (default), `opd`, or `sft`. See [Training modes](#training-modes-rl--opd--sft-via-orchestrator). |

## SFT training

`uv run sft` runs supervised fine-tuning from a HF dataset. It shares model loaders, FSDP setup, checkpointing, and the chat-template plumbing with the RL trainer, so a typical workflow is _SFT → RL → SFT → …_ without any reformatting.

### Dataset format

Two accepted layouts:

- **Prompt-completion**: a HF dataset with `prompt` and `completion` columns ([TRL format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion)). The trainer masks out the prompt and computes loss only over the completion.
- **Messages**: a HF dataset with a single `messages` column containing a list of chat turns. The trainer interprets the whole conversation as one sample, applies role-based loss masking, and trains over all assistant turns.

If both columns are present, `messages` takes precedence.

**Chat-template prefix property.** Multi-turn SFT requires that tokenizing the first _k_ turns of a conversation be a strict prefix of tokenizing all _n ≥ k_ turns. Qwen3's default template _violates_ this (it strips past `<think>` blocks), so use either the prime-rl–patched checkpoints (e.g. `PrimeIntellect/Qwen3-0.6B`) or a custom chat template that preserves thinking. See [Algorithms § Multi-turn trajectories](algorithms.md#multi-turn-trajectories).

### Launch

The minimal SFT run trains `Qwen3-0.6B` on the `reverse-text` SFT dataset:

```bash
uv run sft @ examples/reverse_text/sft.toml --wandb
```

Multi-GPU and multi-node use torchrun under the hood (the `sft` entrypoint manages this for you — see [Scaling § SFT training](scaling.md#sft-training) for non-default layouts).

### SFT-specific knobs

| Knob | What it controls |
|---|---|
| `data.type = "sft"` and `data.path` | HF dataset name or local path |
| `data.batch_size` | Tokens per trainer step (packed) |
| `data.seq_len` | Per-sample sequence length |
| `loss_mask.*` | Which roles contribute to loss; see [Reference § `sft.data.loss_mask`](reference.md#sft-data) |
| `val.interval` | Run validation every N steps; `val.data` mirrors `data` |

## Training modes (RL / OPD / SFT-via-orchestrator)

The RL entrypoint supports three training modes, switched via `orchestrator.training_mode`:

| Mode | Student | Teacher | Use case |
|---|---|---|---|
| `rl` | Required | Forbidden | Standard RL |
| `opd` | Required | Required, must be vLLM (needs `prompt_logprobs`) | [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): student generates rollouts, trainer minimizes KL to teacher logprobs |
| `sft` | Required | Required, any OpenAI-compatible endpoint | Hard-distill: teacher generates rollouts, student trains on them |

For OPD and SFT-via-orchestrator, set `deployment.num_teacher_gpus` to auto-launch a teacher vLLM server, or hand-launch one and pass its URL via `orchestrator.client.base_url`. Debug configs for all variants ship under [`configs/debug/training_modes/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/configs/debug/training_modes).

The standalone `uv run sft` entrypoint is the more traditional SFT path — pure dataset-based, no teacher, no orchestrator. Use `orchestrator.training_mode = "sft"` only when you want a teacher to generate the supervision on the fly.

## Evaluations

Evals run inside the orchestrator on a separate set of envs declared under `[[orchestrator.eval.env]]`:

```toml
[orchestrator.eval]
interval = 25            # evaluate every 25 trainer steps
rollouts_per_example = 4

[[orchestrator.eval.env]]
id = "math-env"
name = "gsm8k-eval"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main", split = "test" }
```

Eval scores land in the trainer logs as `eval/{env}/{avg@k,pass@k}` and in W&B under the same keys. For one-off evaluations outside of training, use `prime eval` (from the [`prime` CLI](https://docs.primeintellect.ai/cli-reference/introduction)) — it defaults to Prime Inference but talks to any OpenAI-compatible endpoint via `--provider vllm --api-base-url ...`:

```bash
prime eval run math-env \
  --env-args '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}' \
  --model PrimeIntellect/Qwen3-0.6B \
  --provider vllm --api-base-url http://localhost:8000/v1 \
  --num-examples 50 --max-tokens 2048
```

## Checkpointing

Checkpointing is split across processes because the orchestrator and trainer can be on different machines and on different steps at any given time. Inference is stateless.

| Process | What's saved | Where |
|---|---|---|
| Trainer | FSDP-sharded model (DCP), optimizer, scheduler, progress | `<output_dir>/checkpoints/step_N/trainer/` |
| Orchestrator | Step counter, total tokens / samples / problems | `<output_dir>/checkpoints/step_N/orchestrator/` |
| Inference | _nothing_ — re-pushed from the latest checkpoint on restart | n/a |
| Trainer (HF weights) | HF-compatible weight snapshot for serving | `<output_dir>/weights/step_N/` |

### Enabling checkpoints

Checkpointing is **off by default** to save disk. Enable it with `--ckpt`:

```bash
uv run rl @ rl.toml --ckpt                              # default: end-of-training only
uv run rl @ rl.toml --ckpt.interval 25                  # every 25 steps
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-last 3  # rolling window of 3
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-interval 100  # …plus permanent every 100
```

### Resuming a run

Re-run the same launch command and pass `--ckpt.resume-step <N>` (or `-1` for "latest"). Make sure `--max-steps` is at least the target final step, not the remaining delta:

```bash
# First run: steps 0–10
uv run rl @ rl.toml --max-steps 10 --ckpt

# Resume: continue to step 20
uv run rl @ rl.toml --max-steps 20 --ckpt.resume-step 10
```

### Saving HF weights for serving

HF-compatible weight snapshots are written under `<output_dir>/weights/step_N/` whenever a full checkpoint runs (or you can write weights-only via `--ckpt.weights-only` for cheaper snapshots). Upload directly:

```bash
uv run hf upload <user>/<model>-RL outputs/weights/step_100
```

For LoRA runs, set `ckpt.weights.save_adapter_separately = true` to also write the raw adapter alongside the merged weights — useful when serving the adapter through a separate `/load_lora_adapter` call.

## Observability

### Log files

The launcher tees every process's stdout/stderr into `<output_dir>/logs/`. The full layout (single-node runs skip the `node_*.log` and `router_*.log` files):

```
<output_dir>/logs/
├── trainer.log                  # rank 0 only; symlink → trainer/node_0.log on multi-node
├── orchestrator.log             # single instance, single file
├── inference.log                # symlink → inference/node_0.log on multi-node
├── trainer/
│   ├── node_*.log               # per-node trainer stdout (multi-node only)
│   └── torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log   # per-rank
├── inference/
│   ├── node_*.log               # per-node inference stdout (multi-node only)
│   └── router_*.log             # vllm-router per replica (multi-node only)
└── envs/{train,eval}/<env_name>/
    ├── env_server.log
    └── env_worker_<id>.log
```

Env worker logs are the first place to look for env-side errors (most user code lives there). Verbosity is controlled by `orchestrator.log.vf_level`. For multi-rank trainer debugging, drop into `logs/trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log` — verbose and per-rank.

Live tailing from a single point (works on the head node for multi-node runs over a shared filesystem):

```bash
tail -F <output_dir>/logs/{trainer,orchestrator,inference}.log
tail -F <output_dir>/logs/trainer/node_*.log     # multi-node only
tail -F <output_dir>/logs/inference/router_*.log # multi-node only
```

### Console output and the tmux helper

`scripts/tmux.sh` opens a 4-pane tmux session that follows `trainer.log`, `orchestrator.log`, `inference.log`, and the union of env worker logs. Start it before launching:

```bash
bash scripts/tmux.sh
# then in the Launcher window:
uv run rl @ ... --output-dir outputs/my-run
```

Pass `-s <session>` and `-o <output_dir>` to run multiple parallel experiments side-by-side in different sessions. The helper also works on a SLURM head node — `bash scripts/tmux.sh my-rl-job /shared/outputs/my-rl-job`.

### Weights & Biases

W&B is off by default. Enable with `--wandb`:

```bash
uv run rl @ rl.toml --wandb                               # default project, random name
uv run rl @ rl.toml --wandb.project my-proj --wandb.name run-42
```

By default (`wandb.shared = true`) the trainer and orchestrator log into a **single shared W&B run**, so all metrics from both processes land in one place. Set `wandb.shared = false` (or pass `--no-wandb.shared`) to fall back to the legacy split — two runs suffixed `-trainer` and `-orchestrator`. Shared mode requires the W&B SDK ≥ 0.19.9 and is incompatible with `wandb.offline = true`.

By default, every 10 steps each process also logs a sample of prompts/completions (with rewards and advantages) and reward/advantage/entropy distributions as W&B tables. Tune via `--wandb.log-extras.interval` and `--wandb.log-extras.sample-ratio`, or disable subsets:

```bash
uv run rl @ rl.toml --wandb \
  --orchestrator.wandb.log-extras.interval 50 \
  --no-trainer.wandb.log-extras.distributions
```

### Platform monitoring

Register a run on the Prime Intellect platform (Prime Lab) and stream training metrics, samples, and distributions to the platform dashboard. Bare flag uses defaults:

```bash
uv run rl @ rl.toml --orchestrator.prime-monitor
```

Or set it in TOML:

```toml
[orchestrator.prime_monitor]
run_name = "my-experiment"
```

Requires `PRIME_API_KEY` (set via `prime login` or env var) and an allowlisted team. Currently internal-only.

### Prometheus and BetterStack

For long-running production training:

- **Prometheus**: set `trainer.metrics_server.port` to expose `/metrics` on each trainer process. vLLM also exposes `/metrics` natively — useful for KV-cache saturation and pending-request counts.
- **BetterStack heartbeats**: set `trainer.heartbeat.url` (and the matching orchestrator field) to ping a heartbeat URL each step. Pair with a BetterStack monitor to page on stalls.

## Important metrics

Pulled from the three console logs (and mirrored to W&B):

**Progress** (orchestrator):

- `reward/{all,env}/mean` — main signal. Should trend upward over hundreds of steps.
- `seq_len/{all,env}/mean` and `is_truncated/{all,env}/mean` — rollout length and truncation rate.
- `num_turns/{all,env}/mean` — for multi-turn envs.
- `empty_rollouts/{all,env}`, `errored_rollouts/{all,env}` — non-zero is fine in small numbers; sustained > 5% is a smell.
- `eval/{env}/{avg@k,pass@k}` — eval scores when `[orchestrator.eval]` is set.

**Stability** (trainer):

- `mismatch_kl/{all,env}/{mean,std,max}` — KL between trainer's current policy and the (older) inference policy that generated the rollouts. A sustained, growing mean is the early-warning sign for off-policy collapse.
- `entropy/{all,env}/mean` — too low means mode-collapse; too high means the model isn't committing.
- `masked_advantage_{positive,negative}/mean` — fraction of DPPO-masked tokens, split by sign.
- `optim/grad_norm` — spikes precede divergence; check the loss config or lower the LR.

**Performance** (trainer + orchestrator step independently):

| Source | Metric | Reading |
|---|---|---|
| trainer | `time/wait_for_batch` | **high → orchestrator bottleneck** |
| orchestrator | `time/wait_for_ckpt` | **high → trainer bottleneck** |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU |
| orchestrator | `scheduler/async_level`, `scheduler/inflight_rollouts` | current async lag |
| vLLM | `vllm:gpu_cache_usage_perc` | → 1.0 means KV cache saturated, slow generation |

## Rules of thumb

- **Start small.** Run `examples/reverse_text/rl.toml` end-to-end on 2 GPUs before scaling. If the smoke run finishes cleanly, your install is good.
- **Batch size ≥ 64.** Smaller batches give noisy gradient estimates and the trainer's overhead-per-step dominates throughput. 64 is the practical floor; 128–512 is typical for production RL.
- **Group size ≥ 8.** Bigger groups (`orchestrator.rollouts_per_example`) make it more likely that a prompt produces a mix of high- and low-reward rollouts, which is what gives the trainer a usable signal — if all rollouts in a group succeed or all fail, the within-group advantage collapses to zero and the trainer learns nothing from that prompt. Bigger groups also tighten advantage normalization. 8 is the floor; 16–32 is common.
- **Pin `output_dir` per run.** Sharing a directory across runs will mix rollouts and break resumes. `--output-dir outputs/<unique-name>` is the simplest discipline.
- **Use `--dry-run` before SLURM.** Validators (CP needs flash-attention, NCCL broadcast needs `max_async_level=1`, etc.) fail fast in dry-run and slow in queue.
- **Don't change `optimization_dtype` / `reduce_dtype`.** These are load-bearing — flipping bfloat16/float32 silently changes training dynamics. Stick with defaults unless you know what you're doing.
