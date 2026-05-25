# Training

This page covers everything you need to launch, observe, checkpoint, and recover a `prime-rl` training run — the RL trainer, the SFT trainer, and the related on-policy distillation mode. For multi-node and cluster layouts, see [Scaling](scaling.md). For the loss math and algorithm knobs, see [Algorithms](algorithms.md).

> **AI agents working in this repo:** the equivalent runbooks are at [`skills/training/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/skills/training) — top-level routing in [`skills/training/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/SKILL.md), launch details in [`skills/training/start-run/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/start-run/SKILL.md), and check-in / restart procedures in [`skills/training/monitor-run/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/monitor-run/SKILL.md).

## Table of Contents

- [Entrypoints](#entrypoints)
- [RL trainer](#rl-trainer)
  - [Launch](#launch)
  - [Useful knobs](#useful-knobs)
  - [Training modes (RL / OPD / SFT)](#training-modes-rl--opd--sft)
  - [Important metrics](#important-metrics)
- [SFT trainer](#sft-trainer)
  - [Dataset format](#dataset-format)
  - [Launch](#launch-1)
  - [SFT-specific knobs](#sft-specific-knobs)
  - [Important metrics](#important-metrics-1)
- [Checkpointing](#checkpointing)
  - [Enabling checkpoints](#enabling-checkpoints)
  - [Resuming a run](#resuming-a-run)
  - [Serving checkpoints](#serving-checkpoints)
- [Observability](#observability)
  - [Log files](#log-files)
  - [Console output](#console-output)
  - [Weights & Biases](#weights--biases)
  - [Platform monitoring](#platform-monitoring)
- [Rules of thumb](#rules-of-thumb)

## Entrypoints

| Command | Purpose | Notes |
|---|---|---|
| `uv run rl` | Wraps the trainer, orchestrator, and inference server in one launch from a merged TOML. | The default for any RL run. Runs locally for single-node experiments; submits to SLURM for single- or multi-node when `[slurm]` is set (see [Scaling § SLURM](scaling.md#slurm)). |
| `uv run sft` | Supervised fine-tuning on a HF dataset. | Launches torchrun internally; never call torchrun directly. |
| `uv run inference` | vLLM server. | Always use this entrypoint over `vllm serve` — it adds `/update_weights`, `/load_lora_adapter`, and `/init_broadcaster`. |
| `uv run trainer` | Standalone trainer process group. | Use only when launching the trainer separately from the orchestrator (e.g. multi-node RL without the `rl` wrapper). |
| `uv run orchestrator` | Standalone orchestrator process. | Pair with a separately-launched trainer + inference. |

## RL trainer

### Launch

The minimal RL run trains an SFT-warmed `Qwen3-0.6B` on the `reverse-text` task — the env is bundled with the `verifiers` submodule, so nothing else needs to be installed:

```bash
uv run rl @ examples/reverse_text/rl.toml
```

### Useful knobs

A condensed view of the knobs you'll most often tune. For trainer-side parallelism, sampling, optimizer, and loss knobs see [Scaling](scaling.md) and [Algorithms](algorithms.md); for the full field reference see [Reference](reference.md).

**Data and algorithm:**

| Knob | What it does |
|---|---|
| `orchestrator.batch_size` | Tasks per trainer step. |
| `orchestrator.group_size` | Rollouts generated per task. |
| `orchestrator.max_off_policy_steps` | How many distinct policies may have contributed to one rollout before it's discarded (default 8). The main off-policy dial on long agentic rollouts — bump for throughput, lower for tighter on-policyness. Watch `errored_rollouts` and `mismatch_kl/all/mean` when tuning. |
| `orchestrator.training_mode` | `rl` (default), `opd`, or `sft`. See [Training modes](#training-modes-rl--opd--sft). |
| `[[orchestrator.train.env]]` | Training environments. List multiple tables for multi-env training; weight them via `ratio`. See [Configuration § Environments](configuration.md#environments-orchestratortrainenv). |
| `[[orchestrator.eval.env]]` + `orchestrator.eval.interval` | Eval environments and cadence (default every 100 steps). |

**Monitoring:**

| Knob | What it does |
|---|---|
| `log.level` | Process log level for trainer + orchestrator (`info` default; falls back to `$PRIME_LOG_LEVEL`). Set per-process via `trainer.log.level` / `orchestrator.log.level`, or globally on the `rl` entrypoint to propagate to both. |
| `orchestrator.log.vf_level` | Env-worker / verifiers log level (`info` default; `debug` is noisy but useful for env debugging). |
| `--wandb` (+ `--wandb.project`, `--wandb.name`) | Enable Weights & Biases logging. See [Weights & Biases](#weights--biases). |
| `--orchestrator.prime-monitor` | Stream metrics to the Prime Intellect platform (Prime Lab). See [Platform monitoring](#platform-monitoring). |

**Run management:**

| Knob | What it does |
|---|---|
| `--clean-output-dir` | Wipe `<output_dir>` before starting. Useful when re-running an experiment with the same name during iteration. |
| `--output-dir outputs/<name>` | Per-run output directory. Always set this when running more than one experiment in parallel. |
| `--max-steps N` | Stop after `N` trainer steps. Overrides the config value. |
| `--dry-run` | Resolve + validate the full config, write per-process TOMLs to `<output_dir>/configs/`, and exit without launching. The fastest way to debug a misbehaving config. |

### Training modes (RL / OPD / SFT)

The RL entrypoint supports three training modes, switched via `orchestrator.training_mode`:

| Mode | Student | Teacher | Use case |
|---|---|---|---|
| `rl` | Required | Forbidden | Standard RL |
| `opd` | Required | Required, must be vLLM (needs `prompt_logprobs`) | [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): student generates rollouts, trainer minimizes KL to teacher logprobs |
| `sft` | Required | Required, any OpenAI-compatible endpoint | Hard-distill: teacher generates rollouts, student trains on them |

The `rl` entrypoint only manages student-policy inference. For OPD and (local-vLLM) SFT, start the teacher inference server manually and point `[orchestrator.teacher.client]` at it:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name <teacher> --server.port 8001
```

The standalone `uv run sft` entrypoint is the more traditional SFT path — pure dataset-based, no teacher, no orchestrator. Use `orchestrator.training_mode = "sft"` only when you want a teacher to generate the supervision on the fly.

### Important metrics

Pulled from the console logs and mirrored to W&B.

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

## SFT trainer

`uv run sft` runs supervised fine-tuning from a HF dataset. It shares model loaders, FSDP setup, checkpointing, and the chat-template plumbing with the RL trainer, so a typical workflow is _SFT → RL → SFT → …_ without any reformatting.

### Dataset format

Two accepted layouts:

- **Prompt-completion**: a HF dataset with `prompt` and `completion` columns ([TRL format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion)). The trainer masks out the prompt and computes loss only over the completion.
- **Messages**: a HF dataset with a single `messages` column containing a list of chat turns. The trainer interprets the whole conversation as one sample, applies role-based loss masking, and trains over all assistant turns.

If both columns are present, `messages` takes precedence.

**Tool definitions.** For tool-use SFT, add a `tools` column (OpenAI function-calling format) or `tool_defs` (verifiers rollout format). Each row's value can be either a list of dicts or a JSON-encoded string of a list — both are accepted, and `tool_defs` rows are auto-converted to OAI shape before being passed into the chat template's `tools=...` argument. The `chat_template_kwargs` column, if present, is forwarded verbatim into `apply_chat_template`.

**Position-dependent chat templates.** Multi-turn SFT under the default tokenization path (`build_incremental_token_mask`) requires that tokenizing the first _k_ turns of a conversation be a strict prefix of tokenizing all _n ≥ k_ turns. Qwen3's upstream template _violates_ this — it strips past `<think>` blocks across user turns, silently corrupting the loss mask. Two fixes:

- **Enable the renderer** (`use_renderer = true`, recommended). The [`renderers`](algorithms.md#renderers) package owns tokenization end-to-end and is robust to position-dependent templates. Hand-coded renderers ship for Qwen3, Qwen3.5, GLM-5, GLM-4.5, Kimi K2/K2.5, MiniMax M2, DeepSeek V3, Nemotron 3, GPT-OSS. Not supported for VLMs.
- **Patched chat template** — the prime-rl–patched checkpoints (e.g. `PrimeIntellect/Qwen3-0.6B`, used in `examples/reverse_text/sft.toml`) ship a chat template that preserves thinking. Or supply your own.

See [Algorithms § Multi-turn trajectories](algorithms.md#multi-turn-trajectories) for the full picture.

### Launch

The minimal SFT run trains `Qwen3-0.6B` on the `reverse-text` SFT dataset:

```bash
uv run sft @ examples/reverse_text/sft.toml --wandb
```

Multi-GPU and multi-node use torchrun under the hood (the `sft` entrypoint manages this for you — see [Scaling § SFT training](scaling.md#sft-training) for non-default layouts).

### SFT-specific knobs

| Knob | What it controls |
|---|---|
| `data.name` | HF dataset name or local path |
| `data.batch_size` | Tokens per trainer step (packed) |
| `data.seq_len` | Per-sample sequence length |
| `loss_mask.*` | Which roles contribute to loss; see [Reference § `sft.data.loss_mask`](reference.md#sft-data) |
| `val.interval` | Run validation every N steps; `val.data` mirrors `data` |

### Important metrics

Pulled from the console log and mirrored to W&B.

**Progress and loss:**

- `loss/mean` — main signal. Should decrease through the run.
- `val/loss` — validation loss when `[val]` is set, logged every `val.interval` steps.
- `progress/epoch`, `progress/num_samples`, `progress/num_tokens` — dataset progress.
- `progress/<subset>/ratio_{samples,tokens}` — when training on multiple HF subsets/splits, the realized mixing ratio.

**Stability and optimization:**

- `optim/grad_norm` — spikes precede divergence.
- `optim/lr`, `optim/zero_grad_ratio` — LR schedule and the fraction of params that received zero gradients (high → dead path or wrong loss masking).
- For MoE: `max_vio/mean` (load-balancing violation), `routing_confidence/mean` — both are logged when non-zero.

**Performance:**

| Metric | Reading |
|---|---|
| `perf/throughput`, `perf/throughput_per_gpu` | tokens/s overall and per GPU |
| `perf/mfu` | MFU |
| `perf/peak_memory` | peak GPU memory (GiB) |
| `time/step`, `time/forward_backward`, `time/save_ckpt` | step breakdown |

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

### Serving checkpoints

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

### Console output

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

## Rules of thumb

- **Start small.** Run `examples/reverse_text/rl.toml` end-to-end on 2 GPUs before scaling. If the smoke run finishes cleanly, your install is good.
- **Batch size ≥ 64.** Smaller batches give noisy gradient estimates and the trainer's overhead-per-step dominates throughput. 64 is the practical floor; 128–512 is typical for production RL.
- **Group size ≥ 8.** Bigger groups (`orchestrator.group_size`) make it more likely that a task produces a mix of high- and low-reward rollouts, which is what gives the trainer a usable signal — if all rollouts in a group succeed or all fail, the within-group advantage collapses to zero and the trainer learns nothing from that task. Bigger groups also tighten advantage normalization. 8 is the floor; 16–32 is common.
- **Pin `output_dir` per run.** Sharing a directory across runs will mix rollouts and break resumes. `--output-dir outputs/<unique-name>` is the simplest discipline.
- **Use `--dry-run` before SLURM.** Validators (CP needs flash-attention, NCCL broadcast needs `max_async_level=1`, etc.) fail fast in dry-run and slow in queue.
