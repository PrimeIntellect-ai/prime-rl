# Training

This page covers everything you need to launch, observe, checkpoint, and recover a `prime-rl` training run — the RL trainer (and the distillation algorithms that run through it) and the SFT trainer. For multi-node and cluster layouts, see [Scaling](scaling.md). For the loss math and algorithm knobs, see [Algorithms](algorithms.md).

> **AI agents working in this repo:** the equivalent runbooks are at [`skills/training/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/skills/training) — top-level routing in [`skills/training/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/SKILL.md), launch details in [`skills/training/start-run/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/start-run/SKILL.md), and check-in / restart procedures in [`skills/training/monitor-run/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/training/monitor-run/SKILL.md).

## Table of Contents

- [Entrypoints](#entrypoints)
- [RL Trainer](#rl-trainer)
  - [Launch](#launch)
  - [Useful Knobs](#useful-knobs)
  - [Algorithms](#algorithms)
  - [Important Metrics](#important-metrics)
- [SFT Trainer](#sft-trainer)
  - [Dataset Format](#dataset-format)
  - [Launch](#launch-1)
  - [SFT-Specific Knobs](#sft-specific-knobs)
  - [Important Metrics](#important-metrics-1)
- [Evals](#evals)
- [Checkpointing](#checkpointing)
  - [Enabling Checkpoints](#enabling-checkpoints)
  - [Resuming a Run](#resuming-a-run)
  - [Serving Checkpoints](#serving-checkpoints)
- [Observability](#observability)
  - [Log Files](#log-files)
  - [Console Output](#console-output)
  - [Weights & Biases](#weights--biases)
  - [Platform Monitoring](#platform-monitoring)
- [Rules of Thumb](#rules-of-thumb)

## Entrypoints

| Command | Purpose | Notes |
|---|---|---|
| `uv run rl` | Wraps the trainer, orchestrator, and inference server in one launch from a merged TOML. | The default for any RL run. Runs locally for single-node experiments; submits to SLURM for single- or multi-node when `[slurm]` is set (see [Scaling § SLURM](scaling.md#slurm)). |
| `uv run sft` | Supervised fine-tuning on a HF dataset. | Launches torchrun internally; never call torchrun directly. |
| `uv run inference` | vLLM server. | Always use this entrypoint over `vllm serve` — it adds `/update_weights`, `/load_lora_adapter`, and `/init_broadcaster`. |
| `uv run trainer` | Standalone trainer process group. | Use only when launching the trainer separately from the orchestrator (e.g. multi-node RL without the `rl` wrapper). |
| `uv run orchestrator` | Standalone orchestrator process. | Pair with a separately-launched trainer, inference, and one `env-server` per source. |
| `uv run eval` | Multi-env evals against a live inference server. | One epoch per source, adaptive concurrency, cursor checkpoints + `--resume`, dashboard + optional platform upload; see [Evals](#evals). |
| `uv run online-eval` | Evals per trainer weight broadcast. | Spawned by `uv run sft` for [Online Evals](#online-eval); standalone only with filesystem broadcasts. |
| `uv run env-server` | Standalone env server for one environment. | The `rl` launcher starts these automatically (one per train/eval source, at a derived loopback address); only needed when running the orchestrator standalone, or for sources with an explicit `serve.address` — those are externally managed (e.g. their own k8s pod) and the launcher expects the server to already run there. |

## RL Trainer

### Launch

The minimal RL run trains an SFT-warmed `Qwen3-0.6B` on the `reverse-text` task — the env is bundled with the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) submodule, so nothing else needs to be installed:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml
```

### Useful Knobs

A condensed view of the knobs you'll most often tune. For trainer-side parallelism, sampling, optimizer, and loss knobs see [Scaling](scaling.md) and [Algorithms](algorithms.md).

**Data and algorithm:**

| Knob | What it does |
|---|---|
| `orchestrator.batch_size` | Tasks per trainer step. |
| `orchestrator.constant_trainer_batch_size` | Keep trainer batches constant when samples have no training signal, such as zero advantage on all tokens. Enabled by default. Disable it for faster collection with variable trainer batch sizes. |
| `orchestrator.group_size` | Rollouts generated per task. |
| `orchestrator.max_off_policy_steps` | Maximum staleness of a trained rollout (default 8): the version a batch trains on minus the oldest version that generated the rollout, queue time included. Episodes past the bound are dropped; a group shares one dispatch version, so its episodes age out together. The main off-policy dial on long agentic rollouts — bump for throughput, lower for tighter on-policyness. Watch `off_policy/*` and `mismatch_kl/all/mean` when tuning. |
| `[orchestrator.algo]` | Training algorithm — its `type` names it (`grpo` default, `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `opsd`, `sft`, `echo`). See [Algorithms](#algorithms). |
| `[[orchestrator.train.source]]` | Training sources. List multiple tables for multi-env training; weight them via `ratio`. See [Configuration § Training sources](configuration.md#training-sources-orchestratortrainsource). |
| `[[orchestrator.eval.source]]` + `orchestrator.eval.interval` | Eval environments and cadence (default every 100 steps). |

**Monitoring:**

| Knob | What it does |
|---|---|
| `log.level` | Process log level for trainer + orchestrator (`info` default; falls back to `$PRIME_LOG_LEVEL`). Set per-process via `trainer.log.level` / `orchestrator.log.level`, or globally on the `rl` entrypoint to propagate to both. |
| `orchestrator.log.vf_level` | Env-worker / [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) log level (`info` default; `debug` is noisy but useful for env debugging). |
| `--monitors.wandb` (+ `--monitors.wandb.project`, `--monitors.wandb.name`) | Enable Weights & Biases logging. See [Weights & Biases](#weights--biases). |
| `--monitors.prime` | Stream metrics and episodes to the Prime Intellect platform (Prime Lab). See [Platform monitoring](#platform-monitoring). |

**Run management:**

| Knob | What it does |
|---|---|
| `--output-dir outputs` | Directory that groups related runs. Each run writes its artifacts to its own run directory `<output_dir>/<run_name>` (`<run_dir>` below). Defaults to `$PRL_OUTPUT_DIR` if set, else `outputs`. |
| `--run.name <name>` | Run name, also the run directory name under `<output_dir>` (override the directory separately via `--run.dir`). Auto-generated as `<envs>--<model>--<short-id>` when unset, so every launch gets a fresh, readable run directory. Set an explicit name for a predictable path — required to resume the run later. |
| `--clean` | Wipe the run directory before starting. Useful when re-running a named run during iteration. |
| `--max-steps N` | Stop after `N` trainer steps. Overrides the config value. |
| `--dry-run` | Resolve + validate the full config, write per-process configs to `<run_dir>/configs/latest/resolved/`, and exit without launching. The fastest way to debug a misbehaving config. |

### Algorithms

The RL entrypoint supports several training algorithms, switched via `[orchestrator.algo]`'s `type` (see [Algorithms](algorithms.md#the-algorithm-abstraction) for the full reference, model references, and per-algorithm customization):

| `algo.type` | Frozen model | Use case |
|---|---|---|
| `grpo` (default) | None | Standard group-relative RL |
| `max_rl` | None | [MaxRL](https://arxiv.org/abs/2602.02710): GRPO with mean-normalized advantages (maximum-likelihood RL) |
| `rae` | None | [SPIRAL](https://arxiv.org/abs/2506.24119)'s role-conditioned advantage estimation: reward minus a per-agent EMA baseline, for multi-agent self-play envs (e.g. `kuhn-poker`) |
| `hierarchical_grpo` | None | GRPO for proposer-solver envs: compare solvers only with attempts on the same proposed problem, and compare proposers with the other proposals in the group |
| `opd` | Required, must be vLLM (needs `prompt_logprobs`) | [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): the policy generates rollouts, the trainer minimizes per-token reverse KL to a reference model |
| `sft` | Required, any OpenAI-compatible endpoint | Hard-distill: a frozen model generates rollouts, the policy trains on its tokens |
| `opsd` | None — the live policy is its own reference (no deployment) | [SDFT](https://arxiv.org/abs/2601.19897): the model is its own reference conditioned on expert demonstrations |
| `echo` | None | GRPO plus cross-entropy on env-observation tokens |

A new algorithm is a named class in code, not a config — see [Algorithms § Authoring an Algorithm](algorithms.md#authoring-an-algorithm).

Frozen models are declared inline on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for `opd` (the frozen model scored against), `[orchestrator.algo.sampling.source]` for `sft` (the model it samples from) — each with `name` + `base_url`. `opsd` declares no frozen model: it self-distills against the live policy. The `rl` entrypoint only manages policy inference — start frozen-model servers yourself and point `base_url` at them:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --vllm.model <frozen-model> --server.port 8001
```

The standalone `uv run sft` entrypoint is the more traditional SFT path — pure dataset-based, no orchestrator. Use the `sft` algorithm only when you want a frozen model to generate the supervision on the fly.

### Important Metrics

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
- `is_masked/mean` — fraction of tokens masked by the IPO trust region.
- `optim/grad_norm` — spikes precede divergence; check the loss config or lower the LR.

**Performance** (trainer + orchestrator step independently):

| Source | Metric | Reading |
|---|---|---|
| trainer | `time/wait_for_batch` | **high → orchestrator bottleneck** |
| orchestrator | `time/wait_for_policy` | **high → trainer bottleneck** |

The trainer warns when batch wait time exceeds active trainer time. Add inference nodes when this warning persists. The orchestrator warns when policy wait time exceeds active orchestrator time. Add trainer nodes when this warning persists. The orchestrator also warns when it discards more than half of an episode window. This warning reports stale, errored, and no-signal counts.

## SFT Trainer

`uv run sft` runs supervised fine-tuning from a HF dataset. It shares model loaders, FSDP setup, checkpointing, and the chat-template plumbing with the RL trainer, so a typical workflow is _SFT → RL → SFT → …_ without any reformatting.

### Dataset Format

Two accepted layouts:

- **Prompt-completion**: a HF dataset with `prompt` and `completion` columns ([TRL format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion)). The trainer masks out the prompt and computes loss only over the completion.
- **Messages**: a HF dataset with a single `messages` column containing a list of chat turns. The trainer interprets the whole conversation as one sample, applies role-based loss masking, and trains over all assistant turns.

If both columns are present, `messages` takes precedence.

**Tool definitions and renderer controls.** For tool-use SFT, add a `tools` column (OpenAI function-calling format) or `tool_defs` ([`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) rollout format). Each row's value can be either a list of dicts or a JSON-encoded string of a list — both are accepted, and `tool_defs` rows are auto-converted to OAI shape before being passed into the renderer.

Renderer-backed SFT reads template controls from the typed `[renderer]` config in the SFT TOML. For example:

```toml
[renderer]
name = "qwen3"
enable_thinking = false
```

If a model needs another template control, add it to that model's renderer config in `renderers` (for example a new field on the relevant `*RendererConfig`) and consume it in the renderer implementation.

**Renderer-backed tokenization.** SFT tokenization is renderer-only. The [`renderers`](algorithms.md#renderers) package owns message-to-token conversion and loss attribution end-to-end, so position-dependent chat templates (for example templates that strip past `<think>` blocks across user turns) do not corrupt the loss mask. `[renderer]` defaults to `name = "auto"`; set a typed renderer config only when you need model-specific template controls. Hand-coded renderers ship for Qwen3, Qwen3.5, GLM-5, GLM-4.5, Kimi K2/K2.5, MiniMax M2, DeepSeek V3, Nemotron 3, GPT-OSS, and VLM families such as Qwen3-VL/Qwen3.5.

**VLM training requires a custom PrimeRL implementation.** Training a model with `[model.vlm]` set (SFT or RL) requires `model.impl = "custom"` and only works for models with a registered PrimeRL VLM class (currently Qwen3.5 dense and MoE).

See [Algorithms § Multi-Turn Trajectories](algorithms.md#multi-turn-trajectories) for the full picture.

### Launch

The minimal SFT run trains `Qwen3-0.6B` on the `reverse-text` SFT dataset:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --monitors.wandb
```

Multi-GPU and multi-node use torchrun under the hood (the `sft` entrypoint manages this for you — see [Scaling § SFT and Torchrun](scaling.md#sft-and-torchrun) for non-default layouts; multi-node SFT goes through [SLURM](scaling.md#slurm)).

### Online Evals

`uv run sft` can evaluate the model on rollout-based envs as it trains, reusing the RL orchestrator's eval machinery. Configure an `[eval]` block — the same shape as `[orchestrator.eval]`: multiple `[[eval.source]]` envs with per-source `interval` / `num_examples` / `group_size` / sampling overrides — plus an `[inference]` block for the vLLM server:

```toml
[eval]
interval = 25
num_examples = 32

[[eval.source]]
name = "reverse-text"
env.taskset.id = "reverse-text"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"

[inference]

[deployment]
num_train_gpus = 1  # trainer
num_infer_gpus = 1  # inference
```

The launcher starts the inference server, one env server per eval source, and an `online-eval` process next to the trainer. NCCL is the default weight transport. The trainer broadcasts weights at startup (fail-fast) and at every step an eval env is due, Every broadcast runs the same four-stage handshake in `broadcasts/step_{n}`: the trainer offers the version (`.sender_ready`) and blocks, the online-eval process acknowledges (`.receiver_ready`), then the trainer transfers (`.started`) and commits (`.finished`). It runs the due envs sequentially per broadcast, so every epoch measures exactly one policy version. Set `[weight_broadcast] type = "filesystem"` to reload weights from disk instead. LoRA and externally managed inference use filesystem broadcast automatically. The base model is evaluated before the first step (disable with `eval.skip_first_step`), and the final broadcast always fires every env. In-flight eval episodes are cancelled by default when the next checkpoint is ready, so stale evals do not delay a weight update. Set `eval.cancel_on_new_checkpoint = false` to drain every triggered epoch instead. The trainer can idle while it waits for slow evals. They are sized by the same adaptive concurrency controller as the orchestrator; bound it with `[eval.concurrency]` (`min_inflight` / `max_inflight`; set them equal for fixed concurrency).

#### Multi-Node Trainer and Inference Pool

On a `multi_node` deployment, one SLURM job reserves `deployment.num_train_nodes + deployment.num_infer_nodes` nodes. The first `num_infer_nodes` run the inference pool, router, env servers, and online-eval process. The remaining nodes run the trainer. The inference pool runs one vLLM engine per DP rank behind one router, with `gpus_per_node / inference.vllm.tensor_parallel_size` engines per node:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2  # trainer nodes
num_infer_nodes = 1  # inference pool + online evals

[inference.vllm]
tensor_parallel_size = 8

[slurm]
job_name = "my-run"
```

The shared script passes the trainer rank-0 hostname directly to the online-eval process for NCCL weight broadcasts. Each transfer is synchronous, but eval rollout execution overlaps with later training steps. The allocation remains active while the final eval finishes. Without `max_steps`, online-eval never sees a final broadcast, so the job remains active until walltime. Trainer and online-eval log to one shared W&B run. The trainer creates it, and online-eval finalizes it.

### SFT-Specific Knobs

| Knob | What it controls |
|---|---|
| `data.name` | HF dataset name or local path |
| `data.batch_size` | Tokens per trainer step (packed) |
| `data.seq_len` | Per-sample sequence length |
| `loss_mask.*` | Which roles contribute to loss (system / user / assistant / tool). |
| `val.interval` | Run validation every N steps; `val.data` mirrors `data` |
| `eval.interval` | Run online evals every N steps; see [Online Evals](#online-eval) |

### Important Metrics

Pulled from the console log and mirrored to W&B.

**Progress and loss:**

- `loss/mean`, `loss/perplexity` — main signal. Should decrease through the run.
- `val/loss`, `val/perplexity` — validation metrics when `[val]` is set, logged every `val.interval` steps.
- `eval/{env}/...` — online eval metrics when `[eval]` is set, logged at each evaluated checkpoint step.
- `progress/epoch`, `progress/num_samples`, `progress/num_tokens` — dataset progress.
- `progress/<subset>/ratio_{samples,tokens}` — when training on multiple HF subsets/splits, the realized mixing ratio.

**Stability and optimization:**

- `optim/grad_norm` — spikes precede divergence.
- `optim/lr` — LR schedule.
- For MoE: `max_vio/mean` (load-balancing violation), `routing_confidence/mean` — both are logged when non-zero.

**Performance:**

| Metric | Reading |
|---|---|
| `perf/throughput`, `perf/throughput_per_gpu` | tokens/s overall and per GPU |
| `perf/mfu` | MFU |
| `perf/peak_memory` | peak GPU memory (GiB) |
| `time/step`, `time/forward_backward`, `time/save_ckpt` | step breakdown |

## Evals

`uv run eval` evaluates one or more environments against a live inference server (a `uv run inference` vLLM server or an external OpenAI-compatible API) and exits after one epoch per source. It reuses the orchestrator's eval pipeline: env servers are spawned per source, episodes are admitted under the adaptive concurrency controller, and every episode streams into the run's trace stream and metrics.

```bash
uv run inference --vllm.model Qwen/Qwen3-4B
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B                      # single source
uv run eval gsm8k -c 8 --env.agent.harness.id bash \
  --client.base_url https://api.pinference.ai/api/v1 --client.api_key_var PRIME_API_KEY
uv run eval @ eval.toml --run.name my-eval                          # several [[source]] blocks
```

Single-source shorthands: `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block, `-n`/`-r` set `num_examples`/`group_size`, `-m` the model, `-o` the output dir, and `-c N` pins the concurrency band. Against an endpoint without vLLM `/metrics` (an external API) the band must be pinned; against vLLM it adapts to KV usage like the orchestrator's. Multi-source runs use a TOML:

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]
max_inflight = 128

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
```

The run writes to `output_dir / run.name` with the same layout as training runs (`configs/attempt_<n>/`, `logs/attempt_<n>/eval.log`, `monitors/file/`), shows up in the [dashboard](#dashboard), and checkpoints its task cursor after every completed group. Relaunch with the same `--run.name` and `--resume` to skip the completed prefix. `--monitors.prime` uploads each source's finished epoch as an evaluation on the Prime Intellect platform (`PRIME_API_KEY` or `prime login`). Examples live in `examples/eval/`.

## Checkpointing

Checkpointing is split across processes because the orchestrator and trainer can be on different machines and on different steps at any given time. Inference is stateless.

| Process | What's saved | Where |
|---|---|---|
| Trainer | FSDP-sharded model (DCP), optimizer, scheduler, progress | `<run_dir>/checkpoints/step_{n}/trainer/` |
| Orchestrator | Progress, per-env data state | `<run_dir>/checkpoints/step_{n}/orchestrator/` |
| Inference | _nothing_ — re-pushed from the latest checkpoint on restart | n/a |

### Enabling Checkpoints

Checkpointing is **off by default** to save disk. Enable it with `--ckpt`:

```bash
uv run rl @ rl.toml --ckpt                              # default: end-of-training only
uv run rl @ rl.toml --ckpt.interval 25                  # every 25 steps
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-last 3  # rolling window of 3
uv run rl @ rl.toml --ckpt.interval 25 --ckpt.keep-interval 100  # …plus permanent every 100
```

### Resuming a Run

Re-run the same launch command and pass `--resume` (latest checkpoint) or `--resume.step <N>`. Resuming reuses the run directory, so the run needs a name you can point back at — launch with `--run.name` (or pass the first run's auto-generated name). Make sure `--max-steps` is at least the target final step, not the remaining delta:

```bash
# First run: steps 1–10
uv run rl @ rl.toml --max-steps 10 --ckpt --run.name my-run

# Resume from the latest checkpoint: continue to step 20
uv run rl @ rl.toml --max-steps 20 --ckpt --resume --run.name my-run

# ...or from a specific step
uv run rl @ rl.toml --max-steps 20 --ckpt --resume.step 10 --run.name my-run

# ...or fork another run's checkpoint into a fresh run
uv run rl @ rl.toml --max-steps 20 --ckpt --run.name my-fork \
  --resume.dir outputs/my-run/checkpoints/step_10
```

### Exporting Checkpoints

Trainer checkpoints are DCP-sharded; export them to HF-format safetensors with `tools/convert_dcp_to_bf16.py`. The script reads the model config from the run's resolved config and writes sharded safetensors plus config/tokenizer assets to `<ckpt_dir>/weights` (or a second positional arg). It exports full fine-tunes only — LoRA checkpoints are rejected.

```bash
# single process (1 GPU)
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_10

# multi-rank for faster gathers and models that don't fit one GPU
uv run torchrun --nproc-per-node 8 tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_10
```

The exported directory loads directly into `uv run inference --vllm.model <dir>` or any HF consumer. Quantize it to blockwise FP8 (DeepSeek/GLM format, loads natively in vLLM) with `tools/convert_bf16_to_fp8.py <dir>`, or go straight from the checkpoint with `tools/convert_dcp_to_fp8.py <ckpt_dir>` (each rank quantizes its gathered slice, writes only `<ckpt_dir>/weights-FP8` — no intermediate bf16 export); dequantize an fp8-only release (e.g. GLM-5-FP8) for training with `tools/convert_fp8_to_bf16.py <dir>`.

## Observability

### Config Files

Each launch writes its command, input TOML, and resolved JSON files to
`<run_dir>/configs/attempt_<n>/`. Resumed runs keep the earlier configs.
`command.txt` uses shell-safe quoting.
`configs/latest` points to the current attempt.

### Log Files

The launcher tees every process's stdout/stderr into `<run_dir>/logs/attempt_<n>/` — every launch (fresh or resumed) gets its own numbered attempt directory, and `logs/latest` symlinks to the current one. The full layout (single-node runs skip the `node_*.log` and `router.log` files — there the router logs into `inference.log`):

```
<run_dir>/logs/latest/     # symlink -> attempt_<n>, one per launch
├── trainer.log                  # rank 0 only; symlink → trainer/node_0.log on multi-node
├── orchestrator.log             # single instance, single file
├── eval.log                    # `uv run eval` process
├── online-eval.log             # SFT online-eval process
├── inference.log                # symlink → inference/node_0.log on multi-node
├── trainer/
│   ├── node_*.log               # per-node trainer stdout (multi-node only)
│   └── torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log   # per-rank
├── inference/
│   ├── node_*.log               # per-node inference stdout (multi-node only)
│   └── router.log               # the single global router (multi-node only)
└── envs/{train,eval}/<env_name>.log # one env server process per source (broker + its workers)
```

Env logs are the first place to look for env-side errors (most user code lives there). Verbosity is controlled by `orchestrator.log.vf_level`. For multi-rank trainer debugging, drop into `logs/latest/trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log` — verbose and per-rank.

Live tailing from a single point (works on the head node for multi-node runs over a shared filesystem):

```bash
tail -F <run_dir>/logs/latest/{trainer,orchestrator,evals,online-eval,inference}.log
tail -F <run_dir>/logs/latest/trainer/node_*.log   # multi-node only
tail -F <run_dir>/logs/latest/inference/router.log # multi-node only
```

### Dashboard

`uv run dashboard [output_dir ...]` (default `outputs/`) serves a local web dashboard at `http://localhost:7788` with five views per run: metrics (the W&B overview sections, read from the file monitor's `metrics.jsonl`), the resolved configs, a rollout trace viewer with per-token overlays (advantage, entropy, trainer/sampling mismatch, IPO stable mask, loss and content masks), merged component logs, and markdown reports from `<run>/reports/`. It only reads the run dirs, so it is safe to point at a live run; pass several output directories to track parallel experiments. A taken port automatically bumps to the next free one, so several dashboards coexist on one node.

A coding agent on the same machine can drive the open dashboard: `POST /api/view` with an on-disk address (`{"run", "tab", "step", "kind", "subset", "episode", "highlight": [...]}`) navigates every connected tab there and paints quote-anchored highlights in the trace viewer. Reports cite traces with `[^id]` markers whose JSON definitions carry the same address plus a verbatim quote; the dashboard re-checks each quote against the trace files and marks the citation verified or broken, so answers stay grounded in what is actually on disk. The `dashboard` skill documents the full contract.

### Weights & Biases

W&B is off by default (the file monitor, which writes the metrics and the trace stream under the run's `monitors/file/` directory, is on by default):

```bash
uv run rl @ rl.toml --monitors.wandb                      # default project, random name
uv run rl @ rl.toml --monitors.wandb.project my-proj --monitors.wandb.name run-42
uv run rl @ rl.toml --no-monitors.file                    # disable the local metric/trace files
```

The trainer and orchestrator log into a **single shared W&B run**, so all metrics from both processes land in one place. Shared mode requires the W&B SDK ≥ 0.19.9 and is incompatible with `monitors.wandb.offline = true`.

prime-rl deliberately logs a **large number of metrics** for maximum observability: every rollout metric is emitted per subset (`all`/`effective`), per statistic (`mean`/`max`/`min`/`p10`/`p90`), and per environment alongside a cross-env aggregate, so a multi-env run can emit thousands of series. To keep that navigable, every training run (RL and SFT) gets an **auto-created `overview` saved view** curating the handful of metrics that matter into `train`, `eval`, `stability`, and `performance` sections (with per-env breakdowns). The view is created once per project and adapts to the run's environments; if a later run uses a different set of environments, a new versioned view (`overview-v2`, …) is created instead of overwriting the first.

### Platform Monitoring

Register a run on the Prime Intellect platform (Prime Lab) and stream training metrics and episodes to the platform dashboard. Bare flag uses defaults:

```bash
uv run rl @ rl.toml --monitors.prime
```

Or set it in TOML:

```toml
[monitors.prime]
name = "my-experiment"
```

Every 10th step the orchestrator uploads the step's episodes (full conversations with rewards and advantages) to the run's sample viewer.

Requires `PRIME_API_KEY` (set via `prime login` or env var) and an allowlisted team. Currently internal-only.

## Rules of Thumb

- **Start small.** Run `examples/basic/reverse-text/rl.toml` end-to-end on 2 GPUs before scaling. If the smoke run finishes cleanly, your install is good.
- **Batch size ≥ 64.** Smaller batches give noisy gradient estimates and the trainer's overhead-per-step dominates throughput. 64 is the practical floor; 128–512 is the range for quick ablations; production RL often runs at 1024+.
- **Group size ≥ 8.** Bigger groups (`orchestrator.group_size`) make it more likely that a task produces a mix of high- and low-reward rollouts, which is what gives the trainer a usable signal — if all rollouts in a group succeed or all fail, the within-group advantage collapses to zero and the trainer learns nothing from that task. Bigger groups also tighten advantage normalization. 8 is the floor; 16–32 is common.
- **Runs never share a directory.** Every launch writes to its own run directory `<output_dir>/<run_name>`, auto-named `<envs>--<model>--<short-id>` by default. Name runs you want to find again or resume with `--run.name <name>`; re-using a name blocks unless you resume or pass `--clean`.
- **Use `--dry-run` before SLURM.** Validators (e.g. CP needs flash-attention) fail fast in dry-run and slow in queue.
