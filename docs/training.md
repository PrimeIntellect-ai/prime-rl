# Training

This page covers everything you need to launch, observe, checkpoint, and recover a `prime-rl` training run — RL, SFT, and the related on-policy distillation mode. For multi-node and cluster layouts, see [Scaling](scaling.md). For the loss math and algorithm knobs, see [Algorithms](algorithms.md).

## Table of Contents

- [Entrypoints](#entrypoints)
- [RL training](#rl-training)
  - [Launch](#launch)
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
  - [Prometheus and BetterStack](#prometheus-and-betterstack)
  - [Platform monitoring](#platform-monitoring)
- [Metrics that matter](#metrics-that-matter)
- [Rules of thumb](#rules-of-thumb)
- [Common issues](#common-issues)

## Entrypoints

| Command | Purpose | Notes |
|---|---|---|
| `uv run rl` | Co-launches inference + orchestrator + trainer on one node | The default for any single-node RL run. Mirrors a `[trainer]` + `[orchestrator]` + `[inference]` TOML. |
| `uv run sft` | Supervised fine-tuning on a HF dataset | Launches torchrun internally; never call torchrun directly. |
| `uv run inference` | OpenAI-compatible vLLM server | Always use this entrypoint over `vllm serve` — it adds `/update_weights`, `/load_lora_adapter`, and `/init_broadcaster`. |
| `uv run trainer` | Standalone trainer process group | Use only when launching the trainer separately from the orchestrator (e.g. multi-node RL without the `rl` wrapper). |
| `uv run orchestrator` | Standalone orchestrator process | Pair with a separately-launched trainer + inference. |

`rl` is a convenience wrapper — it parses one merged TOML, splits it across `[trainer]` / `[orchestrator]` / `[inference]` tables, picks GPUs, sets up logging, and spawns the three children. Standalone entrypoints exist for the multi-node case where each process lives on a different host.

## RL training

### Launch

The minimal single-node RL run uses a shipped example config. From the project root:

```bash
prime env install primeintellect/math-env   # install the env once
bash scripts/tmux.sh                         # 4-pane tmux that tails the logs

uv run rl @ configs/gsm8k/rl.toml \
  --wandb.project my-project \
  --wandb.name gsm8k-smoke \
  --ckpt
```

GPU placement: by default `rl` puts inference on GPU 0 and the trainer on GPU 1. Override with `--inference-gpu-ids` / `--trainer-gpu-ids`:

```bash
uv run rl @ rl.toml \
  --inference-gpu-ids 0,1,2,3 \
  --trainer-gpu-ids 4,5,6,7 \
  --inference.parallel.dp 4
```

For multi-node and SLURM, see [Scaling § RL training](scaling.md#rl-training).

### What each process does at runtime

- **Inference** (vLLM) holds the current policy and serves OpenAI-compatible completions. Receives a new HF checkpoint via `POST /update_weights` after each trainer step (or batched into one update per `max_async_level` steps).
- **Orchestrator** samples a prompt batch from the configured `[[orchestrator.train.env]]` envs, drives them against the inference server (multi-turn, tool calls, etc.), packs the completed rollouts into a binary batch, writes it under `outputs/rollouts/step_N/`, and notifies the trainer.
- **Trainer** waits for the binary batch, runs forward/backward/optimizer step under FSDP2, writes new weights to the broadcast transport, and signals the orchestrator that step `N+1` is in flight.

The orchestrator is the only stateful CPU process; the trainer is GPU-bound; the inference server is stateless apart from KV cache. On restart the orchestrator pushes the latest checkpoint into inference automatically — you don't need to checkpoint inference state.

### Key knobs

These are the knobs you'll touch most often. The full field reference for each lives in [Reference](reference.md).

| Knob | Where | What it controls |
|---|---|---|
| `model.name` | top-level | HF model ID or local path. Auto-fans-out to trainer/orchestrator/inference. |
| `max_steps` | top-level | Number of trainer steps before exit. |
| `seq_len` | top-level | Max sequence length per training sample; also enforced by the orchestrator when packing. |
| `max_async_level` | top-level | How many steps inference can run ahead of the trainer. 1 = fully overlapped; >1 = more off-policy, higher throughput. See [Algorithms § Async](algorithms.md#async--off-policy-training). |
| `orchestrator.batch_size` | orchestrator | Prompts per trainer step. |
| `orchestrator.rollouts_per_example` | orchestrator | Rollouts per prompt (the group size used for advantage normalization). |
| `orchestrator.train.sampling.max_completion_tokens` | orchestrator | Max tokens per turn at sampling time. |
| `inference.parallel.tp` / `inference.parallel.dp` | inference | Tensor and data parallelism for the inference server. |
| `inference.gpu_memory_utilization` | inference | Fraction of GPU memory vLLM may use. Tighten on co-located single-GPU runs. |
| `trainer.optim.lr` | trainer | Learning rate. Default optimizer is AdamW. |
| `trainer.loss.type` | trainer | Pick the loss variant (default AIPO vs custom). See [Algorithms § Loss](algorithms.md#loss). |

## SFT training

`uv run sft` runs supervised fine-tuning from a HF dataset. It shares model loaders, FSDP setup, checkpointing, and the chat-template plumbing with the RL trainer, so a typical workflow is _SFT → RL → SFT → …_ without any reformatting.

### Dataset format

Two accepted layouts:

- **Prompt-completion**: a HF dataset with `prompt` and `completion` columns ([TRL format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion)). The trainer masks out the prompt and computes loss only over the completion.
- **Messages**: a HF dataset with a single `messages` column containing a list of chat turns. The trainer interprets the whole conversation as one sample, applies role-based loss masking, and trains over all assistant turns.

If both columns are present, `messages` takes precedence.

**Chat-template prefix property.** Multi-turn SFT requires that tokenizing the first _k_ turns of a conversation be a strict prefix of tokenizing all _n ≥ k_ turns. Qwen3's default template _violates_ this (it strips past `<think>` blocks), so use either the prime-rl–patched checkpoints (e.g. `PrimeIntellect/Qwen3-0.6B`) or a custom chat template that preserves thinking. See [Algorithms § Multi-turn trajectories](algorithms.md#multi-turn-trajectories).

### Launch

Single GPU:

```bash
uv run sft @ configs/<your-sft-config>.toml --wandb
```

Multi-GPU and multi-node use torchrun under the hood (the `sft` entrypoint manages this for you — see [Scaling § SFT training](scaling.md#sft-training) for non-default layouts).

A CPU-friendly smoke run with fake data:

```bash
uv run sft @ configs/debug/sft/train.toml
```

### SFT-specific knobs

| Knob | What it controls |
|---|---|
| `data.type = "sft"` and `data.path` | HF dataset name or local path |
| `data.batch_size` | Tokens per trainer step (packed) |
| `data.seq_len` | Per-sample sequence length |
| `loss_mask.*` | Which roles contribute to loss; see [Reference § `sft.data.loss_mask`](reference.md#sft-data) |
| `val.interval` | Run validation every N steps; `val.data` mirrors `data` |

## Training modes (RL / OPD / SFT-via-orchestrator)

The RL entrypoint also supports two distillation modes, switched via `orchestrator.training_mode`:

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

Eval scores land in the trainer logs as `eval/{env}/{avg@k,pass@k}` and in W&B under the same keys. For one-off evaluations outside of training, use `vf-eval`:

```bash
uv run vf-eval math-env \
  -a '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}' \
  -m PrimeIntellect/Qwen3-0.6B \
  -b http://localhost:8000/v1 \
  -n 50 -t 2048
```

`vf-eval` talks to any OpenAI-compatible endpoint, so it works against `uv run inference`, hosted endpoints, or a stale checkpoint mid-run.

## Checkpointing

Checkpointing is split across processes because the orchestrator and trainer can be on different machines and on different steps at any given time. Inference is stateless.

| Process | What's saved | Where |
|---|---|---|
| Trainer | FSDP-sharded model (DCP), optimizer, scheduler, progress | `<output_dir>/checkpoints/step_N/` |
| Orchestrator | Step counter, total tokens / samples / problems | `<output_dir>/checkpoints/orchestrator/step_N/` |
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

Common combo for long runs: `--ckpt.interval 50 --ckpt.keep-last 3 --ckpt.keep-interval 500` — rolling 3-checkpoint window for fast recovery, plus a permanent snapshot every 500 steps.

### Resuming a run

Re-run the same launch command and pass `--ckpt.resume-step <N>` (or `-1` for "latest"). Make sure `--max-steps` is at least the target final step, not the remaining delta:

```bash
# First run: steps 0–10
uv run rl @ rl.toml --max-steps 10 --ckpt

# Resume: continue to step 20
uv run rl @ rl.toml --max-steps 20 --ckpt.resume-step 10
```

Trainer + orchestrator step counters are kept in lockstep — both rewind to the same resume step. The inference server can stay running across restarts; the orchestrator pushes the resumed weights on reconnect.

### Saving HF weights for serving

HF-compatible weight snapshots are written under `<output_dir>/weights/step_N/` whenever a full checkpoint runs (or you can write weights-only via `--ckpt.weights-only` for cheaper snapshots). Upload directly:

```bash
uv run hf upload <user>/<model>-RL outputs/weights/step_100
```

For LoRA runs, set `ckpt.weights.save_adapter_separately = true` to also write the raw adapter alongside the merged weights — useful when serving the adapter through a separate `/load_lora_adapter` call.

## Observability

### Log files

The launcher tees every process's stdout/stderr into `<output_dir>/logs/`:

```
<output_dir>/logs/
├── trainer.log                  # rank 0 only
├── orchestrator.log
├── inference.log
├── trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log
└── envs/{train,eval}/<env_name>/
    ├── env_server.log
    └── env_worker_<id>.log
```

Multi-node runs add `trainer/node_*.log` and `inference/node_*.log` — `trainer.log` and `inference.log` at the top level symlink to node 0 for convenience. See [Scaling § Multi-node logs](scaling.md#multi-node-logs).

Env worker logs are the first place to look for env-side errors (most user code lives there). Verbosity is controlled by `orchestrator.log.vf_level`.

### Console output and the tmux helper

`scripts/tmux.sh` opens a 4-pane tmux session that follows `trainer.log`, `orchestrator.log`, `inference.log`, and the union of env worker logs. Start it before launching:

```bash
bash scripts/tmux.sh
# then in the Launcher window:
uv run rl @ ... --output-dir outputs/my-run
```

Pass `-s <session>` and `-o <output_dir>` to run multiple parallel experiments side-by-side in different sessions.

For multi-node SLURM runs, follow the head-node logs via `tail -f` on the shared filesystem — see [Scaling § SLURM](scaling.md#slurm).

### Weights & Biases

W&B is off by default. Enable with `--wandb`:

```bash
uv run rl @ rl.toml --wandb                               # default project, random name
uv run rl @ rl.toml --wandb.project my-proj --wandb.name run-42
```

For RL runs the trainer and orchestrator log as **two separate runs** with the same name: `<name>-trainer` and `<name>-orchestrator`. You'll usually want both grouped in a W&B group.

By default, every 10 steps each process also logs a sample of prompts/completions (with rewards and advantages) and reward/advantage/entropy distributions as W&B tables. Tune via `--wandb.log-extras.interval` and `--wandb.log-extras.sample-ratio`, or disable subsets:

```bash
uv run rl @ rl.toml --wandb \
  --orchestrator.wandb.log-extras.interval 50 \
  --no-trainer.wandb.log-extras.distributions
```

### Prometheus and BetterStack

For long-running production training:

- **Prometheus**: set `trainer.metrics_server.port` to expose `/metrics` on each trainer process. vLLM also exposes `/metrics` natively — useful for KV-cache saturation and pending-request counts.
- **BetterStack heartbeats**: set `trainer.heartbeat.url` (and the matching orchestrator field) to ping a heartbeat URL each step. Pair with a BetterStack monitor to page on stalls.

### Platform monitoring

Internal teams can register runs on the Prime Intellect platform:

```toml
[orchestrator.prime_monitor]
run_name = "my-experiment"
```

This streams training metrics, samples, and distributions to the platform dashboard. Requires `PRIME_API_KEY` (set via `prime login` or env var) and an allowlisted team. Currently internal-only.

## Metrics that matter

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

Live vLLM stats (Prometheus):

```bash
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
```

## Rules of thumb

- **Start small.** Run `configs/gsm8k/rl.toml` end-to-end on 2 GPUs before scaling. If GSM8K runs cleanly, your install is good.
- **Eyeball the reward distribution.** If `reward/all/std` collapses to ~0 within a few steps, the env is too easy or rewards are degenerate — increase difficulty or check the rubric.
- **Match `inference.parallel.tp` to model layout.** TP > num attention heads / 2 starts losing efficiency. For dense models keep TP small and use DP for throughput. For MoE-heavy models prefer EP.
- **Set `max_async_level` deliberately.** `1` = fully synced overlap (lowest off-policy drift). `2` = default, suited for cross-WAN weight broadcast. Higher values trade more drift for throughput; watch `mismatch_kl/all/mean`.
- **Pin `output_dir` per run.** Sharing a directory across runs will mix rollouts and break resumes. `--output-dir outputs/<unique-name>` is the simplest discipline.
- **Use `--dry-run` before SLURM.** Validators (CP needs flash-attention, NCCL broadcast needs `max_async_level=1`, etc.) fail fast in dry-run and slow in queue.
- **Don't change `optimization_dtype` / `reduce_dtype`.** These are load-bearing — flipping bfloat16/float32 silently changes training dynamics. Stick with defaults unless you know what you're doing.

## Common issues

**`@ path/to/x.toml` fails to load.** Leave a space between `@` and the path — `@ rl.toml`, not `@rl.toml`. If the error mentions Pydantic, your TOML doesn't match the schema; `--dry-run` will pinpoint the offending field.

**API timeouts under load.** Bump file descriptors: `ulimit -n 32000`. Our defaults are already generous, so a real timeout usually means inference is saturated — check `time/generate_completions` and vLLM's `gpu_cache_usage_perc`.

**CUDA OOM in the trainer.** In order, try:

1. Full activation checkpointing: `--model.ac` (the bare flag enables defaults).
2. Lower `seq_len` or `data.micro_batch_size`.
3. FSDP CPU offload: `--model.fsdp-cpu-offload` (or `--model.optim-cpu-offload` for optimizer states only).
4. Context parallelism: `--model.cp 2` (requires flash-attention; see [Scaling § CP](scaling.md#context-parallelism)).

**CUDA OOM in inference.** Tighten `inference.gpu_memory_utilization` (start around 0.85), reduce `inference.model.max_model_len`, or split inference across more GPUs via `inference.parallel.dp`.

**Eval scores frozen but training reward rising.** Likely a chat-template prefix violation eating the model's outputs. Check `orchestrator.renderer` settings (`preserve_all_thinking`, etc.) and use the prime-rl–patched model checkpoint if available.

**Trainer hangs on weight broadcast.** NCCL transport requires `max_async_level=1` and is incompatible with LoRA — the run will fail at config-validate time if either is set. Otherwise check that all trainer ranks survived the previous step (`grep ERROR logs/trainer/torchrun/`).

**Run dies mid-step with no traceback.** Look in `<output_dir>/logs/envs/train/<env>/env_worker_*.log` first — most silent kills come from OOM-killed env workers running user code. Set `orchestrator.log.vf_level = "debug"` for more verbose env logging.
