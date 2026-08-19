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
| `uv run evals` | Multi-env evals against a live inference server. | Standalone one-shot evals (see [Evals](#evals)), or checkpoint-watching online evals with an `[online]` block (the `sft` launcher configures this). Spawns its own env servers. |
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
| `orchestrator.group_size` | Rollouts generated per task. |
| `orchestrator.max_off_policy_steps` | How many distinct policies may have contributed to one rollout before it's discarded (default 8). The main off-policy dial on long agentic rollouts — bump for throughput, lower for tighter on-policyness. Watch `errored_rollouts` and `mismatch_kl/all/mean` when tuning. |
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
| `--output-dir outputs` | Directory that groups related runs. Each run writes its artifacts to its own run directory `<output_dir>/<run_name>` (`<run_dir>` below). |
| `--run.name <name>` | Run name, also the run directory name under `<output_dir>` (override the directory separately via `--run.dir`). Auto-generated as `<envs>--<model>--<short-id>` when unset, so every launch gets a fresh, readable run directory. Set an explicit name for a predictable path — required to resume the run later. |
| `--clean` | Wipe the run directory before starting. Useful when re-running a named run during iteration. |
| `--max-steps N` | Stop after `N` trainer steps. Overrides the config value. |
| `--dry-run` | Resolve + validate the full config, write per-process TOMLs to `<run_dir>/configs/`, and exit without launching. The fastest way to debug a misbehaving config. |

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
- `masked_advantage_{positive,negative}/mean` — fraction of DPPO-masked tokens, split by sign.
- `optim/grad_norm` — spikes precede divergence; check the loss config or lower the LR.

**Performance** (trainer + orchestrator step independently):

| Source | Metric | Reading |
|---|---|---|
| trainer | `time/wait_for_batch` | **high → orchestrator bottleneck** |
| orchestrator | `time/wait_for_ckpt` | **high → trainer bottleneck** |

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

[eval.source.env.taskset]
id = "reverse-text"

[eval.source.env.agent.harness]
id = "null"

[eval.source.env.agent.runtime]
type = "subprocess"

[inference]

[deployment]
num_train_gpus = 1  # trainer
num_infer_gpus = 1  # inference
```

The launcher starts the inference server, one env server per eval source, and an `evaluator` process next to the trainer. The handoff is the filesystem, not NCCL: the trainer writes an HF weight checkpoint at every step an eval env is due (in addition to `ckpt.interval`), and the evaluator watches `weights/step_{n}`, points the inference server at each stable checkpoint (`/update_weights` reload from disk), and runs the due envs against it — sequentially per checkpoint, so every epoch measures exactly one policy version. The base model is evaluated before the first step (disable with `eval.skip_first_step`), and the final checkpoint always fires every env. In-flight eval episodes are sized by the same adaptive concurrency controller as the orchestrator; bound it with `[eval.concurrency]` (see [Evals](#evals)).

#### Multi-Node (Decoupled Trainer and Inference Pool)

On a `multi_node` deployment (SLURM), the trainer and the eval deployment are **two independent SLURM jobs**. `deployment.num_nodes` sizes the trainer job; `deployment.num_infer_nodes` sizes the eval job, which runs the inference pool (one vLLM engine per DP rank behind a single router, `gpus_per_node / inference.vllm.tensor_parallel_size` engines per node), one env server per eval source, and the evaluator:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2  # trainer job
num_infer_nodes = 1  # eval job (inference pool + evaluator)

[inference.vllm]
tensor_parallel_size = 8

[slurm]
job_name = "my-run"
```

The only coupling is weight checkpoints on the shared filesystem, so the jobs' lifetimes are independent: when training finishes, the trainer job exits and releases its nodes even while evals are still running; the eval job keeps draining pending checkpoints and exits after evaluating the final one (`max_steps` — without it the eval job never sees a final checkpoint and holds its allocation until walltime). Trainer and evaluator log to a single shared W&B run across both jobs — the trainer creates it, the evaluator finalizes it. Any train × inference layout works: `num_nodes` and `num_infer_nodes` are fully independent.

### SFT-Specific Knobs

| Knob | What it controls |
|---|---|
| `data.name` | HF dataset name or local path |
| `data.batch_size` | Tokens per trainer step (packed) |
| `data.seq_len` | Per-sample sequence length |
| `loss_mask.*` | Which roles contribute to loss (system / user / assistant / tool). |
| `val.interval` | Run validation every N steps; `val.data` mirrors `data` |
| `eval.interval` | Run online evals every N steps; see [Online Evals](#online-evals) |

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

`uv run evals` runs multi-env evals against a live inference server — the same evaluator that powers SFT online evals. Without an `[online]` block it runs one epoch of every eval source against the weights the server currently serves, then exits. Start inference separately and point the eval client at it:

```toml
model = "Qwen/Qwen3-4B"

[eval.client]
base_url = "http://localhost:8000/v1"

[eval.concurrency]
min_inflight = 8
max_inflight = 128

[eval.sampling]
temperature = 0.6
top_p = 0.95

[[eval.source]]
name = "aime2025"
group_size = 4

[eval.source.env.taskset]
id = "aime25"

[eval.source.env.agent.harness]
id = "null"

[eval.source.env.agent.runtime]
type = "subprocess"
```

- **Env servers**: the evaluator spawns one env server per source at its derived loopback address (`eval.env_server_base_port` + source index). Sources with an explicit `serve.address` are externally managed — the evaluator only connects.
- **Concurrency**: in-flight episodes are sized by the same adaptive concurrency controller as the orchestrator, driven by engine KV pressure from the inference `/metrics` polls. Bound it with `[eval.concurrency]` (`min_inflight` / `max_inflight` / `initial_inflight`). Eval episodes are measurements: an overload cut only blocks new admissions, in-flight episodes always run to completion.
- **External inference APIs** (e.g. Prime Inference) expose no vLLM `/metrics`, so adaptive concurrency has no load signal. The evaluator probes `/metrics` at startup and fails fast unless the band is pinned (`min_inflight = max_inflight`), which runs fixed concurrency. See [`examples/evals/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/evals/swe.toml) for a full example (SWE-bench Verified + Terminal-Bench 2 against Prime Inference, 1h agent cap).
- **Results** land in the configured monitors (`eval/{env}/...` metrics, episode traces via the file monitor) plus a console success line per env.

With an `[online]` block (`weights_dir`, `max_steps`, `resume_step`) the evaluator instead watches the weights directory for stable `step_{n}` checkpoints and evaluates each eligible one — the mode the `sft` launcher configures for [Online Evals](#online-evals).

## Checkpointing

Checkpointing is split across processes because the orchestrator and trainer can be on different machines and on different steps at any given time. Inference is stateless.

| Process | What's saved | Where |
|---|---|---|
| Trainer | FSDP-sharded model (DCP), optimizer, scheduler, progress | `<run_dir>/checkpoints/step_N/trainer/` |
| Orchestrator | Progress, per-env data state | `<run_dir>/checkpoints/step_N/orchestrator/` |
| Inference | _nothing_ — re-pushed from the latest checkpoint on restart | n/a |
| Trainer (HF weights) | HF-compatible weight snapshot for serving | `<run_dir>/weights/step_N/` |

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

### Serving Checkpoints

HF-compatible weight snapshots are written under `<run_dir>/weights/step_N/` whenever a full checkpoint runs (or you can write weights-only via `--ckpt.weights-only` for cheaper snapshots). Upload directly:

```bash
uv run hf upload <user>/<model>-RL outputs/<run_name>/weights/step_100
```

For LoRA runs, set `ckpt.weights.save_adapter_separately = true` to also write the raw adapter alongside the merged weights — useful when serving the adapter through a separate `/load_lora_adapter` call.

## Observability

### Log Files

The launcher tees every process's stdout/stderr into `<run_dir>/logs/attempt_<n>/` — every launch (fresh or resumed) gets its own numbered attempt directory, and `logs/latest` symlinks to the current one. The full layout (single-node runs skip the `node_*.log` and `router.log` files — there the router logs into `inference.log`):

```
<run_dir>/logs/latest/     # symlink -> attempt_<n>, one per launch
├── trainer.log                  # rank 0 only; symlink → trainer/node_0.log on multi-node
├── orchestrator.log             # single instance, single file
├── inference.log                # symlink → inference/node_0.log on multi-node
├── trainer/
│   ├── node_*.log               # per-node trainer stdout (multi-node only)
│   └── torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log   # per-rank
├── inference/
│   ├── node_*.log               # per-node inference stdout (multi-node only)
│   └── router.log               # the single global router (multi-node only)
└── envs/{train,eval}/<env_name>.log # one env server process per source (broker + its workers)
```

Env logs are the first place to look for env-side errors (most user code lives there). Verbosity is controlled by `orchestrator.log.vf_level`. For multi-rank trainer debugging, drop into `logs/trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log` — verbose and per-rank.

Live tailing from a single point (works on the head node for multi-node runs over a shared filesystem):

```bash
tail -F <run_dir>/logs/latest/{trainer,orchestrator,inference}.log
tail -F <run_dir>/logs/latest/trainer/node_*.log   # multi-node only
tail -F <run_dir>/logs/latest/inference/router.log # multi-node only
```

### Console Output

`scripts/tmux.sh` opens a 4-pane tmux session that follows `trainer.log`, `orchestrator.log`, `inference.log`, and the union of env worker logs. Start it before launching:

```bash
bash scripts/tmux.sh -o outputs/my-run
# then in the Launcher window:
uv run rl @ ... --run.name my-run
```

Pass `-s <session>` and `-o <run_dir>` (the run directory, `<output_dir>/<run_name>`) to run multiple parallel experiments side-by-side in different sessions. The helper also works on a SLURM head node — `bash scripts/tmux.sh my-rl-job /shared/outputs/my-rl-job`.

### Weights & Biases

W&B is off by default (the file monitor, which writes `metrics.jsonl` and the per-step trace files to the run directory, is on by default):

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
