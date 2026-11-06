# Training

This page covers everything you need to launch, observe, checkpoint, and recover a `prime-rl` training run — the RL trainer (and the distillation algorithms that run through it) and the SFT trainer. For multi-node and cluster layouts, see [Scaling](scaling.md). For the loss math and algorithm knobs, see [Algorithms](algorithms.md). For standalone evals, see [Eval](eval.md).

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
  - [Custom Loss](#custom-loss)
  - [Important Metrics](#important-metrics-1)
- [Pretraining](#pretraining)
- [Checkpointing](#checkpointing)
  - [Enabling Checkpoints](#enabling-checkpoints)
  - [Resuming a Run](#resuming-a-run)
  - [Pausing or Checkpointing a Live Run](#pausing-or-checkpointing-a-live-run)
  - [Updating the Config Live](#updating-the-config-live)
  - [Serving Checkpoints](#serving-checkpoints)
- [Observability](#observability)
  - [Config Files](#config-files)
  - [Log Files](#log-files)
  - [Dashboard](#dashboard)
- [Monitors](#monitors)
  - [File](#file)
  - [Weights & Biases](#weights--biases)
  - [Prime](#prime)
- [Rules of Thumb](#rules-of-thumb)

## Entrypoints

| Command | Purpose | Notes |
|---|---|---|
| `uv run launch` | Runs the entrypoint named by the config's `mode`. | See [`uv run launch`](#uv-run-launch). |
| `uv run rl` | Wraps the trainer, orchestrator, and inference server in one launch from a merged TOML. | The default for any RL run. Runs locally for single-node experiments; submits to SLURM for single- or multi-node when `[slurm]` is set (see [Scaling § SLURM](scaling.md#slurm)). |
| `uv run sft` | Supervised fine-tuning on a HF dataset. | Launches torchrun internally; never call torchrun directly. |
| `uv run pretrain` | Pretraining from scratch on tokenized shards. | Launches torchrun internally; see [Pretraining](#pretraining). |
| `uv run inference` | vLLM server. | Always use this entrypoint over `vllm serve` — it adds `/update_weights`, `/load_lora_adapter`, and `/init_broadcaster`. |
| `uv run orchestrator` | Standalone orchestrator process. | Pair with a separately-launched inference server and one `env-worker-pool` per source. |
| `uv run eval` | Multi-env evals against a live inference server. | One epoch per source, pinned (or adaptive) concurrency, cursor checkpoints + `--resume`, dashboard + optional platform upload; see [Eval](eval.md). |
| `uv run env-worker-pool` | One source's worker pool replica: a broker and its env workers. | Launchers start these automatically. Use this command for standalone pools, including sources whose `serve.address` points to an externally managed pool. |

### `uv run launch`

A config can name its run kind with a top-level `mode`: `rl`, `sft`, `pretrain` or `eval`. `uv run launch` reads it from the config files and runs that entrypoint with the same arguments:

```toml
mode = "pretrain"
```

```bash
uv run launch @ pretrain.toml          # same as: uv run pretrain @ pretrain.toml
uv run launch @ pretrain.toml --help   # pretrain's help
```

`mode` is optional for `uv run rl`, `sft`, `pretrain` and `eval`. When several config files set `mode`, they must set the same one.

### Env worker pools

Each train/eval source has its own env configuration. Two sources can use the same taskset
with different agents or settings. Each source can have multiple worker pool replicas:
single-node runs start one, and multi-node RL runs start one per inference node. An explicit
`serve.address` selects one externally managed pool instead.

```text
Orchestrator
└── Source (env configuration)
    ├── Pool replica 1: PRL::EnvBroker
    │   ├── PRL::EnvWorker
    │   └── PRL::EnvWorker
    └── Pool replica 2: PRL::EnvBroker
        └── PRL::EnvWorker
```

The orchestrator selects the least-loaded replica; its broker dispatches to the least-busy
worker. Each worker loads its own Verifiers env and can execute multiple episodes concurrently.
The orchestrator loads the dataset and sends task data with each request. Even a one-worker
pool has a broker and a separate worker process.

`[serve]` configures the pool's address, static/elastic worker count and per-worker episode
concurrency. Each broker publishes its bound address under
`configs/attempt_N/resolved/envs/<split>/<name>.address` (or `<name>.<replica>.address` for
multiple replicas). Broker and worker logs share the source's log file.

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
| `orchestrator.train.group_size` | Rollouts generated per task. |
| `orchestrator.max_staleness` | Maximum token staleness (default 8): the version a batch trains on minus the policy version that sampled the token, queue time included. Staler tokens are masked out of the loss; the rest of the sample still trains. Tokens without a policy label (Dynamo engines) take their rollout's dispatch version. A sample is dropped only when nothing is left to train, and rollouts are never cancelled for staleness. The main staleness dial on long agentic rollouts — bump for throughput, lower for tighter on-policyness. Watch `staleness/token/masked`, `staleness/rollout/*` and `mismatch_kl/all/mean` when tuning. |
| `[orchestrator.train.algo]` | Training algorithm — its `type` names it (`grpo` default, `max_rl`, `rae`, `hierarchical_grpo`, `opd`, `opsd`, `sft`, `echo`). See [Algorithms](#algorithms). |
| `trainer.model.freeze_moe_router` | Freeze MoE router gates (default `true` for RL, `false` for SFT). Router updates in RL shift the trainer's expert choices away from the ones inference sampled, amplifying the train-inference mismatch, so RL keeps the pretrained router. Set `false` to train it. |
| `[[orchestrator.train.source]]` | Training sources. List multiple tables for multi-env training; weight them via `ratio`. See [Configuration § Training sources](configuration.md#training-sources-orchestratortrainsource). |
| `[[orchestrator.eval.source]]` + `orchestrator.eval.interval` | Eval environments and cadence (default every 100 steps). |

**Monitoring:**

| Knob | What it does |
|---|---|
| `log.level` | Process log level for trainer + orchestrator (`info` default; falls back to `$PRIME_LOG_LEVEL`). Set per-process via `trainer.log.level` / `orchestrator.log.level`, or globally on the `rl` entrypoint to propagate to both. The orchestrator's env servers and its [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) logs use the same level. |
| `--monitors.wandb` (+ `--monitors.wandb.project`, `--monitors.wandb.name`) | Enable Weights & Biases logging. See [Weights & Biases](#weights--biases). |
| `--no-monitors.prime` | Stop streaming metrics and episodes to the Prime Intellect platform (Prime Lab), on by default. See [Prime](#prime). |

**Run management:**

| Knob | What it does |
|---|---|
| `--output-dir outputs` | Directory that groups related runs. Each run writes its artifacts to its own run directory `<output_dir>/<run_name>` (`<run_dir>` below). Defaults to `$PRL_OUTPUT_DIR` if set, else `outputs`. |
| `--run.name <name>` | Run name, also the run directory name under `<output_dir>` (override the directory separately via `--run.dir`). Auto-generated as `<envs>--<model>--<short-id>` when unset, so every launch gets a fresh, readable run directory. Set an explicit name for a predictable path — required to resume the run later. |
| `--clean` | Wipe the run directory before starting. Useful when re-running a named run during iteration. |
| `--max-steps N` | Stop after `N` trainer steps. Overrides the config value. |
| `--dry-run` | Resolve + validate the full config, write per-process configs to `<run_dir>/configs/latest/resolved/`, and exit without launching. The fastest way to debug a misbehaving config. |

### Algorithms

The RL entrypoint supports several training algorithms, switched via `[orchestrator.train.algo]`'s `type` (see [Algorithms](algorithms.md#the-algorithm-abstraction) for the full reference, model references, and per-algorithm customization):

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

Frozen models are declared inline on the algorithm, named where the model is used — `[orchestrator.train.algo.teacher]` for `opd` (the frozen model scored against) and `sft` (the model it samples from) — each with `name` + `base_url`. `opsd` declares no frozen model: it self-distills against the live policy. The `rl` entrypoint only manages policy inference — start frozen-model servers yourself and point `base_url` at them:

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
- `is_timeout/{all,env}/mean` — share of rollouts stopped by a stage deadline (`setup`/`agent`/`finalize`/`scoring` `_timeout`). Training counts a timed-out rollout as an error, so it stays out of the batch; eval scores it.
- `num_turns/{all,env}/mean` — for multi-turn envs.
- `empty_rollouts/{all,env}`, `errored_rollouts/{all,env}` — non-zero is fine in small numbers; sustained > 5% is a smell.
- `eval/{env}/{avg@k,pass@k}` — eval scores when `[orchestrator.eval]` is set.

**Stability** (trainer):

- `mismatch_kl/{all,env}/{mean,std,max}` — KL between trainer's current policy and the (older) inference policy that generated the rollouts. A sustained, growing mean is the early-warning sign for off-policy collapse.
- `mismatch_kl_by_staleness/{bin}/{mean,std,max}` — the same KL per log2 bin of token staleness (`0`, `1`, `2-3`, `4-7`, …, `64+`). Bin `0` is the numerical trainer–inference mismatch alone; the rise across bins is the policy drift that staleness adds.
- `entropy/{all,env}/mean` — too low means mode-collapse; too high means the model isn't committing.
- `is_masked/mean` — fraction of tokens masked by the IPO trust region.
- `optim/grad_norm` — spikes precede divergence; check the loss config or lower the LR.

**Performance** (trainer + orchestrator step independently):

| Source | Metric | Reading |
|---|---|---|
| trainer | `time/wait_for_batch` | **high → orchestrator bottleneck** |
| orchestrator | `time/wait_for_policy` | **high → trainer bottleneck** |
| orchestrator | `staleness/token/mean`, `staleness/token/masked` | mean token staleness of the shipped batch, and the share of its tokens masked past `max_staleness` |
| orchestrator | `staleness/rollout/{mean,max}` | rollout staleness (max token staleness) of the shipped batch |
| orchestrator | `staleness/rollout/{in_flight,in_queue}/{mean,max}` | in-flight and in-queue split of rollout staleness |
| orchestrator | `staleness/rollout/dropped` | queued rollouts dropped because every token passed `max_staleness` |

See [Async Training & Staleness](algorithms.md#async-training--staleness) for the token and rollout staleness definitions.

The trainer warns when batch wait time exceeds active trainer time. Add inference nodes when this warning persists. The orchestrator warns when policy wait time exceeds active orchestrator time. Add trainer nodes when this warning persists. The orchestrator also warns when it discards more than half of an episode window. This warning reports stale, errored, and no-signal counts.

## SFT Trainer

`uv run sft` runs supervised fine-tuning from a HF dataset. It shares model loaders, FSDP setup, checkpointing, and the chat-template plumbing with the RL trainer, so a typical workflow is _SFT → RL → SFT → …_ without any reformatting.

### Dataset Format

Two accepted layouts:

- **Prompt-completion**: a HF dataset with `prompt` and `completion` columns ([TRL format](https://huggingface.co/docs/trl/en/dataset_formats#prompt-completion)). The trainer masks out the prompt and computes loss only over the completion.
- **Messages**: a HF dataset with a single `messages` column containing a list of chat turns. The trainer interprets the whole conversation as one sample, applies role-based loss masking, and trains over all assistant turns.

If both columns are present, `messages` takes precedence. A dataset that stores these under other names maps them in `[data.columns]`; the trainer renames the columns at load time:

```toml
[data.columns]
messages = "conversation"
tools = "schemas"
```

**Per-message loss selection.** An optional `message_loss_mask` column contains one boolean or integer 0/1 per message. For example, `[0, 0, 0, 1]` on a user/assistant/user/assistant conversation trains only the final assistant turn while retaining the full context. For prompt-completion rows, the mask covers the concatenated prompt and completion messages. Zero excludes a message; one preserves the normal role and renderer loss settings. A missing or null mask leaves those settings unchanged. The renderer applies selection without changing token IDs, including ownership of assistant closing tokens. Invalid mask lengths or entries raise an error.

To read a differently named column, set `data.columns.message_loss_mask = "selection"`.

**Tool definitions and renderer controls.** For tool-use SFT, add a `tools` column in OpenAI function-calling format. Each row's value can be either a list of dicts or a JSON-encoded string of a list.

Renderer-backed SFT reads template controls from the typed `[renderer]` config in the SFT TOML. For example:

```toml
[renderer]
name = "qwen3"
enable_thinking = false
```

Dataset columns can set renderer template controls per row, on top of the `[renderer]` config. `[data.columns.renderer]` maps renderer fields to columns and defaults to `reasoning_effort = "reasoning_effort"`, so a `reasoning_effort` column sets that field on renderers that have it (for example `gpt-oss`, `qwen3.8`, `deepseek-v4`). Rows with a null value use the configured renderer unchanged. This lets one run mix reasoning efforts, with the `[renderer]` value as the default for rows that do not set one:

```toml
# The renderer field is reasoning_effort; this dataset stores it in an effort column.
[data.columns.renderer]
reasoning_effort = "effort"
```

If a model needs another template control, add it to that model's renderer config in `renderers` (for example a new field on the relevant `*RendererConfig`) and consume it in the renderer implementation.

**Custom renderers.** A renderer can also live outside the `renderers` package. Set `name = "custom"` and point `import_path` at the class, as `my_module.Class` or `path/to/file.py:Class`; relative file paths resolve against the launch directory. The class sets `config_class` to its own `BaseRendererConfig` subclass, which validates the remaining `[renderer]` keys and declares its template controls:

```toml
[renderer]
name = "custom"
import_path = "my_renderers.glm53_no_thinking.NoThinkingGLM53Renderer"
clear_thinking = true
```

The same config selects the renderer for RL under `[orchestrator.renderer]`. The [`renderers` README](https://github.com/PrimeIntellect-ai/renderers#custom-renderers) shows the class behind this example.

**Renderer-backed tokenization.** SFT tokenization is renderer-only. The [`renderers`](algorithms.md#renderers) package owns message-to-token conversion and loss attribution end-to-end, so position-dependent chat templates (for example templates that strip past `<think>` blocks across user turns) do not corrupt the loss mask. `[renderer]` defaults to `name = "auto"`; set a typed renderer config only when you need model-specific template controls. Hand-coded renderers ship for Qwen3, Qwen3.5, Qwen3.8, GLM-5, GLM-4.5, Kimi K2/K2.5, MiniMax M2, DeepSeek V3, Nemotron 3, GPT-OSS, and VLM families such as Qwen3-VL/Qwen3.5/Qwen3.8.

**VLM training requires a custom PrimeRL implementation.** Training a model with `[model.vlm]` set (SFT or RL) only works for models with a registered PrimeRL VLM class: Qwen3.5 dense and MoE, and Qwen3.8 Flash Next (`qwen4_exp`, text and images). See the [multimodal examples](../examples/extra/vlm/README.md) for SFT and multi-node image RL configurations.

See [Algorithms § Multi-Turn Trajectories](algorithms.md#multi-turn-trajectories) for the full picture.

### Launch

The minimal SFT run trains `Qwen3-0.6B` on the `reverse-text` SFT dataset:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --monitors.wandb
```

Multi-GPU and multi-node use torchrun under the hood (the `sft` entrypoint manages this for you — see [Scaling § SFT and Torchrun](scaling.md#sft-and-torchrun) for non-default layouts; multi-node SFT goes through [SLURM](scaling.md#slurm)).

The run streams to the Prime Intellect platform by default — see [Prime](#prime).

### Online Evals

`uv run sft` can evaluate the model on rollout-based envs as it trains, reusing the RL orchestrator's eval machinery. Configure an `[eval]` block — the same shape as `[orchestrator.eval]`: multiple `[[eval.source]]` envs with per-source `interval` / `select` / `group_size` / sampling overrides — plus an `[inference]` block for the vLLM server:

```toml
[eval]
interval = 25

[eval.select]
limit = 32

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

The launcher starts the inference server, one env worker pool per eval source, and an online-eval process next to the trainer (it logs to `logs/attempt_<n>/eval.log`). NCCL is the default weight transport. The trainer broadcasts weights at startup (fail-fast) and at every step an eval env is due, Every broadcast runs the same four-stage handshake in `broadcasts/step_{n}`: the trainer offers the version (`.sender_ready`) and blocks, the online-eval process acknowledges (`.receiver_ready`), then the trainer transfers (`.started`) and commits (`.finished`). It runs the due envs sequentially per broadcast, so every epoch measures exactly one policy version. Set `[weight_broadcast] type = "filesystem"` to reload weights from disk instead. LoRA and externally managed inference use filesystem broadcast automatically. The base model is evaluated before the first step (disable with `eval.skip_first_step`), and the final broadcast always fires every env. In-flight eval episodes are cancelled by default when the next checkpoint is ready, so stale evals do not delay a weight update. Set `eval.cancel_on_new_checkpoint = false` to drain every triggered epoch instead. The trainer can idle while it waits for slow evals. They are sized by the same adaptive concurrency controller as the orchestrator; bound it with `[eval.concurrency]` (`min_inflight` / `max_inflight`; set them equal for fixed concurrency).

#### Multi-Node Trainer and Inference Pool

On a `multi_node` deployment, one SLURM job reserves `deployment.num_train_nodes + deployment.num_infer_nodes` nodes. The first `num_infer_nodes` run the inference pool, router, env worker pools, and online-eval process. The remaining nodes run the trainer. The inference pool runs one vLLM engine per DP rank behind one router, with `gpus_per_node / inference.vllm.tensor_parallel_size` engines per node:

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
| `eval.interval` | Run online evals every N steps; see [Online Evals](#online-evals) |

### Custom Loss

`[loss]` selects the SFT trainer's loss: token cross-entropy (`type = "ce"`, the default) or, with `type = "custom"`, a loss function of yours in its place. The samples are stock SFT's: the chat rendered per `data.columns` and `[loss_mask]`, shifted for next-token prediction. Shuffling, rank sharding, epochs, resume, packing, normalization, clipping, validation and checkpointing stay the trainer's. The loss is computed **per micro-batch** on the backbone's final hidden states: the function returns a summed loss and the count it averages over, and the trainer divides by the global count across ranks and micro-batches, as it does with the token count for cross-entropy. This is the SFT counterpart of the RL [custom loss](algorithms.md#custom-loss), a function called as `my_loss(inputs, **kwargs)` with the config's `kwargs`, except that it returns a summed loss with its count, and its metrics are summed and divided by the global count rather than averaged per sequence.

```python
# my_module.py
from prime_rl.trainer.models.layers.lm_head import token_logprobs
from prime_rl.trainer.sft.loss import LossInputs, LossOutputs

def focal_loss(inputs: LossInputs, gamma: float = 2.0) -> LossOutputs:
    """Token cross-entropy, down-weighting tokens the model already predicts well: -(1 - p)^gamma * log p."""
    batch = inputs.batch
    # one LM-head call for the whole micro-batch: each token's log-probability of its target
    logprobs = token_logprobs(inputs.model.lm_head, inputs.hidden_states, batch["target_ids"])
    mask = batch["loss_mask"]
    focal = -((1 - logprobs.exp()) ** gamma) * logprobs
    return LossOutputs(
        loss=(focal * mask).sum(), count=mask.sum(), metrics={"ce": -(logprobs * mask).sum().detach()}
    )
```

Wire it up:

```toml
[loss]
type = "custom"
import_path = "my_module.focal_loss"
kwargs = { gamma = 2.0 }
```

The dataclasses:

```python
@dataclass
class LossInputs:
    model: nn.Module
    hidden_states: Float[Tensor, "batch seq hidden"]  # post-norm, no vocabulary projection
    batch: Batch                                       # on the device: input_ids, position_ids, target_ids, loss_mask, seq_lens

@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]                   # summed; divided by the global `count`
    count: int | Tensor                       # e.g. samples or tokens in this micro-batch
    metrics: dict[str, Tensor] = {}           # summed; divided like the loss, same names on every call
```

The loss runs inside the model's forward pass, where stock SFT computes its cross-entropy, so it can call any module of the model, such as the LM head through `token_logprobs` above (one call per micro-batch: with a trainable LM head, each call of the fused head's summed cross-entropy keeps a weight-sized gradient buffer until backward, and on an untied model each further call re-gathers the head in backward; keep further calls independent of each other's outputs). Call modules rather than reading their weights: FSDP gathers a unit's parameters while its modules run, so a weight read directly can be a shard (a decoder layer's or, on an untied model, the embedding's). A trainable module belongs to the model, since the optimizer, rank sync and checkpoints see only the model's parameters. Call the same trainable modules on every micro-batch of a step: with `model.dp_replicate > 1` (HSDP) the gradients are all-reduced over the replicas once, after the step's last backward, so a module called only on an earlier micro-batch gets no gradient or fails that backward. `count` is an integer; metrics are scalar tensors with the same names on every call; with a global count of 0 the step's gradients are zeroed (the optimizer still applies weight decay and momentum). Not supported with context parallelism, VLM training or `model.offload = "full"`.

### Important Metrics

Pulled from the console log and mirrored to W&B.

**Progress and loss:**

- `loss/mean`, `loss/perplexity` — main signal. Should decrease through the run.
- `val/loss`, `val/perplexity` — validation metrics when `[val]` is set, logged every `val.interval` steps.
- With a custom loss: `loss/<name>` and `val/<name>` for each metric the loss function returns, averaged like the loss; the perplexities are not logged, since the loss need not be a cross-entropy.
- `eval/{env}/...` — online eval metrics when `[eval]` is set, logged at each evaluated checkpoint step.
- `progress/epoch`, `progress/num_samples`, `progress/num_tokens` — dataset progress.
- `progress/<subset>/ratio_{samples,tokens}` — when training on multiple HF subsets/splits, the realized mixing ratio.

**Stability and optimization:**

- `optim/grad_norm` — spikes precede divergence.
- `optim/lr` — LR schedule.
- For MoE: `max_vio/mean`, `max_vio/max` — mean and max over the step's microsteps of the largest load-balancing violation across layers and EP groups, computed from expert token counts summed across each EP group. `routing_confidence/mean` — mean routing confidence.
- For MoE: `expert_load/{cv,max_mean,cold_frac}/{mean,max}` — expert load per layer over the whole step, from token counts summed across all ranks, as the mean and max over MoE layers. `cv` is std/mean of the per-expert token counts, `max_mean` the busiest expert's count over the mean, and `cold_frac` the fraction of experts receiving under 0.1x the mean. A trainable router that drifts shows up as a rising `cv`/`cold_frac` (MiMo-V2.6 §5.4).
- For MoE with EP: `ep_imbalance/rank/{mean,max}` — the busiest EP rank's routes over the mean rank's (padding included) in each layer and microstep, which sets the layer's expert time. It takes the max over DP groups at each layer and microstep, then the mean and max over the step's layers and microsteps; `ep_imbalance/rank_dp_mean/{mean,max}` takes the DP mean instead.

**Performance:**

| Metric | Reading |
|---|---|
| `perf/throughput`, `perf/throughput_per_gpu` | tokens/s overall and per GPU |
| `perf/mfu` | MFU, including attention FLOPs |
| `perf/mfu_6dn` | MFU from the plain 6 x active params x tokens estimate (no attention FLOPs) |
| `perf/peak_memory` | peak GPU memory from all sources, max over ranks (GiB) |
| `perf/non_torch_memory` | GPU memory outside PyTorch's caching allocator (symmetric memory such as Mega MoE's buffers, NCCL, CUDA context), max over ranks (GiB) |
| `time/step`, `time/forward_backward`, `time/save_ckpt` | step breakdown |

## Pretraining

`uv run pretrain` trains a model from scratch on tokenized text. It shares the trainer, launcher, checkpointing and monitoring with `uv run sft`. What differs is the data, and that the weights start from the model's own init: `model.name` only supplies the architecture, and a resumed run loads its weights from the checkpoint. The from-scratch init is tested on dense models. The defaults are SFT's, except `optim.lr = 1e-3` and a 10-step linear learning-rate warmup.

**Tokenize first.** prime-rl does not tokenize; it reads the output of [datatrove](https://github.com/huggingface/datatrove)'s `DocumentTokenizer`: `*.ds` token files, each with its `.ds.index` (document ends) and `.ds.metadata`. datatrove is a separate install (`pip install datatrove`), not a prime-rl dependency. Set `eos_token` to the tokenizer's end-of-text token, which is not always its EOS: for Qwen it is `<|endoftext|>`, not `<|im_end|>`. datatrove adds no separator by default and prime-rl adds none either, so without one the documents run together and the model never learns to stop; the trainer checks at startup that the documents end with one. If a document's text contains the separator string, the tokenizer may encode it as the separator token, and the loss at that position is then masked too.

```python
# Run with datatrove installed. Tokenize held-out documents into another folder for validation.
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer

LocalPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"),
        DocumentTokenizer(
            output_folder="data/fineweb-edu",
            tokenizer_name_or_path="Qwen/Qwen3-0.6B",
            eos_token="<|endoftext|>",
        ),
    ],
    tasks=16,
).run()
```

**Configure.** Only `data.paths` is required:

```toml
mode = "pretrain"          # optional; lets `uv run launch` run this config
max_steps = 1000

[model]
name = "Qwen/Qwen3-0.6B"   # only the architecture is read

[data]
paths = ["data/fineweb-edu"]
seq_len = 2048
batch_size = 256

[val]                      # optional
interval = 100
num_samples = 512          # documents per validation; leave out to score every held-out document

[val.data]
paths = ["data/fineweb-edu-val"]
seq_len = 2048
```

All files in `data.paths` (in folder order, then file name order) form one token pool. Each epoch reads every `seq_len` window of the pool once, in a new random order and from a new random offset drawn from `data.seed`. Attention and positions restart at every document; `data.document_masking = false` makes a window one sequence. In training, the model is not trained to predict the next document's first token: there is no loss where the input is the document separator (`data.eod_mask_loss = false` turns this off). The separator is the special token the training shards' documents end with, detected at startup; shards whose documents start with BOS have nothing to mask. A resumed run reads exactly the windows an uninterrupted run would, also on another number of GPUs, as long as `data.num_workers` stays the same.

Validation scores every document in `val.data.paths` on its own, from its first token, and logs the mean loss over all scored tokens as `val/loss`. A document is cut into non-overlapping windows of `val.data.seq_len` scored tokens; a longer document continues in the next window, which does not see the tokens before it. Every token after a document's first is scored, and no position predicts a token of another document. The windows are packed into rows of `val.data.seq_len * val.data.micro_batch_size` tokens and shared out over the data-parallel ranks. `val.num_samples` limits every validation to the first documents of the held-out shards, the same ones every time; without it, every validation scores all of them.

**Launch.**

```bash
uv run pretrain @ pretrain.toml
```

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

### Asynchronous Checkpoints

By default a trainer step waits until its checkpoint is written to storage. With `--ckpt.async-save` (`--trainer.ckpt.async-save` also works for `rl`), a save only copies each rank's model and optimizer state into pinned host memory. A background thread then writes the copy to storage while training continues:

```bash
uv run rl @ rl.toml --ckpt.interval 2 --ckpt.async-save
uv run sft @ sft.toml --ckpt.interval 2 --ckpt.async-save
```

The checkpoint holds the same state that a synchronous save at that step would write. It works with every `model.offload` mode:

- With `model.offload = "none"`, the copy from GPU memory runs alongside the next step's forward and backward passes. The trainer waits for it to finish before the optimizer step changes the state. Until then, any temporary GPU tensors the checkpoint is built from, such as gathered entries of fused parameters, stay allocated.
- With `optimizer` or `full` offload, the save copies the CPU optimizer state and returns once the copy is done, because offloaded state changes outside the optimizer step. The copy reads the state where it lives, without moving it through the GPU as a synchronous save does.

Some things to know:

- **Host memory.** Each rank keeps pinned host buffers about the size of its checkpoint shard (model plus optimizer state), and reuses them for every save. On a node, this adds up to the node's share of the full checkpoint. With optimizer offload, this comes on top of the offloaded state itself.
- **One save at a time.** A save first waits for the previous write to finish. If the checkpoint interval is shorter than a write takes, the trainer waits for the write.
- **Completeness.** DCP writes `trainer/.metadata` after every rank has written its shards. A resume skips any step whose trainer checkpoint has no `.metadata`, for example one that was still being written when the job died. `keep_last` and `keep_interval` only count checkpoints that finished writing, so cleanup never deletes the newest finished checkpoint to make room for one that is still in flight.
- **Failures.** If a background write fails, the trainer raises its error at the next optimizer step, the next save or the end of training, and stops. The final checkpoint, and any checkpoint requested through the control API, is written before the trainer exits or reports it as saved.

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

A fork continues the other run, so both runs must be of the same kind. To start a new stage from another run's weights, such as RL from an SFT checkpoint, set `trainer.model.init_from` instead. It loads only the weights, and the run starts at step 0 with a fresh optimizer, scheduler and orchestrator. `model.name` still sets the architecture and tokenizer.

```bash
uv run rl @ rl.toml --max-steps 50 --ckpt --run.name my-rl \
  --trainer.model.init-from outputs/my-sft/checkpoints/step_500
```

#### What a resume restores

- The curriculum of each train source, and the live config (see [Updating the Config Live](#updating-the-config-live)).
- The open train groups: groups that were drawn but not shipped at the checkpoint. The checkpoint saves no partial groups, so a resume re-runs each one whole, before it draws new tasks. A re-run group goes through the curriculum again, with its new rollouts. A group that finished without training (for example, rejected by the curriculum) does not re-run.
- A resume drops the open groups of a removed source or task. It also drops those of a streaming taskset, which cannot find a task by key.

#### Changing sources on resume

A resume can add, remove or change train and eval sources.

- Sources match by name (`name`, else the env id). Renaming a source counts as removing it and adding a new one. A source that you add on resume uses the launch config.
- Tasks match by `Task.key`. The default key hashes the task data without the generated row index (`idx`), so adding, removing or moving rows keeps the state of the other tasks. A change to a task's other data, including a config-level `system_prompt`, makes it a new task. Identical rows need distinct `TaskData.id` values or a custom `Task.key` to keep their state separate.
- A new train source starts with a fresh curriculum. The checkpointed curriculum of a removed train source is dropped.
- When a source's taskset gains or loses tasks, `StandardSampler` keeps the sampling counts of the remaining tasks. New tasks start at the lowest remaining count, so they join the current pass. `DifficultyPoolSampler` drops the rewards of removed tasks, and new tasks start unseen.
- A streaming taskset resumes at the same offset into the stream, so a changed stream does not resume at the same task.
- A changed sampler type starts the sampler fresh. Admission gates match by name: a new gate starts fresh, and the state of a removed gate is dropped.
- Eval sources hold no checkpointed state. A new eval source first runs at the next step that is a multiple of its interval.

### Pausing or Checkpointing a Live Run

With `--trainer.control` and `--orchestrator.control` (SFT: `--control`), the trainer serves a small control API on port 8040 and the orchestrator on 8041. Each saves only its own state, so a pause asks the trainer first: it accepts the pause at its next step N, trains N, saves it and stops. The orchestrator is then asked to pause at N and saves its state as of shipping N, which it can always do because a paused trainer never requests N + 1. A pause is `committed` on each once that component saved N, and the run continues with `--resume`:

```bash
uv run rl @ rl.toml --ckpt --trainer.control --orchestrator.control --run.name my-run

curl -X POST localhost:8040/v1/actions -H 'content-type: application/json' -d '{"type": "pause"}'
# {"id": "3f2a...", "type": "pause", "state": "pending", "step": null, "changes": null}
curl localhost:8040/v1/actions/3f2a...
# {"id": "3f2a...", "type": "pause", "state": "accepted", "step": 12, "changes": null}
curl -X POST localhost:8041/v1/actions -H 'content-type: application/json' -d '{"type": "pause", "step": 12}'
# {"id": "9c41...", "type": "pause", "state": "pending", "step": 12, "changes": null}
```

Once paused, both keep serving the API until the run is stopped, so its owner can read the committed step first.

A `checkpoint` action takes the same two calls and saves step N without stopping the run. Since the trainer keeps going, the orchestrator has to receive its request before the trainer finishes N and requests N + 1; a later request gets a 409 because the orchestrator no longer has the state of N:

```bash
curl -X POST localhost:8040/v1/actions -H 'content-type: application/json' -d '{"type": "checkpoint"}'
curl -X POST localhost:8041/v1/actions -H 'content-type: application/json' -d '{"type": "checkpoint", "step": 12}'
```

The trainer also lists the checkpoints in its checkpoint dir, ascending by step. A step is `complete` once the trainer finished saving it; an incomplete one is still being written (or was interrupted):

```bash
curl localhost:8040/v1/checkpoints
# {"checkpoints": [{"step": 10, "complete": true}, {"step": 12, "complete": false}]}
```

### Updating the Config Live

With `--orchestrator.control`, you can also resubmit the full orchestrator config while the run is going. Read the current config, edit it, and submit it back as an `update_config` action:

```bash
curl localhost:8041/v1/config > current.json
# {"revision": 0, "step": 41, "config": {...}}
jq '{type: "update_config", config: (.config | .max_staleness = 4 | .train.source[0].sampling.temperature = 0.8)}' current.json > update.json
curl -X POST localhost:8041/v1/actions -H 'content-type: application/json' -d @update.json
# {"id": "7b0e...", "type": "update_config", "state": "pending", "step": null, "changes": [{"path": "train.source[0].sampling.temperature", "old": 1.0, "new": 0.8}, {"path": "max_staleness", "old": 8, "new": 4}]}
curl localhost:8041/v1/actions/7b0e...
# {"id": "7b0e...", "type": "update_config", "state": "applied", "step": 43, "changes": [...]}
```

The orchestrator validates the config and compares it with the running one when you submit it. A rejected update returns 422 with the changed fields that are frozen in `frozen`, and any validation `errors`. It returns 409 if another update is queued or the run is pausing. The update applies after the orchestrator ships its next step: rollouts dispatched from then on use the new config, while running rollouts, and groups that already started, keep their old settings. Nothing is cancelled. Each applied update bumps the config revision, is logged, and is written to `configs/attempt_<n>/resolved/updates/rev_<k>.json`. Checkpoints record the live config, so `--resume` continues with the updates that were in effect at the checkpoint. The rule is per field: if the launch config still has the value the checkpointed run was launched with, the live value is restored; if you changed the field in the launch config, your value wins. The orchestrator logs the restored fields and writes them to `updates/rev_<k>.json` of the new attempt.

Only these fields are live; every other field is frozen:

| Field | Effect |
|---|---|
| `batch_size`, `max_staleness` | The next batch uses the new size, and queued samples are checked against the new staleness bound. |
| `concurrency.min_inflight`, `concurrency.max_inflight` | Moves the adaptive concurrency band. A lower limit cancels nothing; new rollouts wait until enough finish. A target set through the concurrency API stays in place. `concurrency.initial_inflight` only seeds the startup cap, so it may change (e.g. to stay below a lower `max_inflight`) without effect. |
| `train.source[*].ratio` | Changes the env mix weights. |
| `train.source[*].group_size` | Changes how many rollouts each new group of the source gets. Groups that already started keep the size they started with. `batch_size` must stay divisible by every source's group size, and `concurrency.max_inflight` must stay at least the largest. Set it per source: the group-level `train.group_size` is only a default for the sources and is frozen. |
| `train.source[*].sampling` | Changes a source's `temperature`, `max_completion_tokens` and `extra_body`. `top_p`, `top_k` and `logprobs` are frozen: they decide sampling replay and top-logprob streams at launch. Set sampling per source: the group-level `train.sampling` is only a default for the sources and is frozen. |
| `eval.source[*].interval` | Changes eval cadence from the next policy update. |
| `train.source[*].algo.length_penalty` | Changes a GRPO source's length penalty: its weights, its type, or turning it on (a config) or off (`null`). It applies to every group scored from the next step on, including groups already running, since a group is scored when its last rollout finishes. Set it per source: the group-level `train.algo` is only a default for the sources and is frozen. |

Frozen fields include the model, tokenizer, renderer, transports, monitors, logging, checkpointing, output directory, padding and the trainer topology, which are fixed at launch. `seq_len` and `max_steps` are frozen because the trainer depends on them. Eval group sizes, algorithms (except the length penalty), curricula, eval sampling and task selection are frozen, as are adding, removing, renaming or reordering sources and turning eval on or off. In the config, a live field is annotated with `Annotated[T, Live()]` (from `prime_rl.configs.diff`); every other field is frozen.

### Exporting Checkpoints

Trainer checkpoints are DCP-sharded; export them to HF-format safetensors with `tools/convert_dcp_to_bf16.py`. The script reads the model config from the run's resolved config and writes sharded safetensors plus config/tokenizer assets to `<ckpt_dir>/weights` (or a second positional arg). It exports full fine-tunes only — LoRA checkpoints are rejected.

```bash
# stream one decoder layer at a time on CPU
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_10

# load and write up to four decoder layers concurrently on CPU
uv run python tools/convert_dcp_to_bf16.py outputs/my-run/checkpoints/step_10 --workers 4
```

N-gram tables are read one stored DCP chunk at a time (one per training rank) and written as their HF table shards. `--workers` bounds how many decoder layers or tables are in flight; peak host memory is each in-flight decoder layer with its conversion buffers, plus, for each in-flight table, about twice its largest DCP storage chunk in the stored dtype (PyTorch holds the raw bytes while decoding a chunk).

The export is staged in a hidden directory next to the output directory and renamed into place when complete, so the output's parent must be writable, a mount point cannot itself be the output directory, and replacing an existing export needs disk for both until the old one is removed. A failed or interrupted publish restores the previous export. An existing non-empty directory other than the default export dir is only replaced with `--overwrite`.

The exported directory loads directly into `uv run inference --vllm.model <dir>` or any HF consumer. Quantize it to blockwise FP8 (DeepSeek/GLM format, loads natively in vLLM) with `tools/convert_bf16_to_fp8.py <dir>`, or go straight from the checkpoint with `tools/convert_dcp_to_fp8.py <ckpt_dir>` (streams decoder layers on CPU and writes only `<ckpt_dir>/weights-FP8`, with no intermediate BF16 export; `--workers` controls concurrency, and `--device cuda` enables GPU quantization instead of the default CPU); dequantize an fp8-only release (e.g. GLM-5-FP8) for training with `tools/convert_fp8_to_bf16.py <dir>`.

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
├── eval.log                     # `uv run eval` process, or the SFT online-eval process
├── inference.log                # symlink → inference/node_0.log on multi-node
├── trainer/
│   ├── node_*.log               # per-node trainer stdout (multi-node only)
│   └── torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log   # per-rank
├── inference/
│   ├── node_*.log               # per-node inference stdout (multi-node only)
│   └── router.log               # the single global router (multi-node only)
└── envs/{train,eval}/<env_name>.log # source's worker pool (broker + worker processes)
```

Env logs are the first place to look for env-side errors (most user code lives there). Verbosity is controlled by `orchestrator.log.level`. For multi-rank trainer debugging, drop into `logs/latest/trainer/torchrun/<rdzv>/attempt_0/<rank>/{stdout,stderr}.log` — verbose and per-rank.

Live tailing from a single point (works on the head node for multi-node runs over a shared filesystem):

```bash
tail -F <run_dir>/logs/latest/{trainer,orchestrator,eval,inference}.log
tail -F <run_dir>/logs/latest/trainer/node_*.log   # multi-node only
tail -F <run_dir>/logs/latest/inference/router.log # multi-node only
```

### Dashboard

`uv run dashboard [output_dir ...]` (default `outputs/`) serves a local web dashboard at `http://localhost:7788` with six views per run: an overview (the W&B overview sections, read from the file monitor's `metrics.jsonl`), metrics (every key in that file, one pane each, with a regex filter), the resolved configs, a rollout trace viewer with per-token overlays (advantage, entropy, trainer/sampling mismatch, IPO stable mask, loss and content masks), merged component logs, and markdown reports from `<run>/reports/`. The trace viewer separates the transcript, the wall-clock timeline of physical prefix branches, a wall-clock terminal replay of model and tool activity, and a top-to-bottom semantic graph of labeled model-call relationships. Replay uses recorded model-call and message timestamps; because providers do not persist per-token timestamps, it reveals response text evenly across the measured call span and labels that cadence as inferred. Agent and context labels in the semantic graph show the latest and peak prompt lengths first, with cumulative token processing available below on hover and click. It only reads the run dirs, so it is safe to point at a live run; pass several output directories to track parallel experiments. A taken port automatically bumps to the next free one, so several dashboards coexist on one node.

A coding agent on the same machine can drive the open dashboard: `POST /api/view` with an on-disk address (`{"run", "tab", "step", "kind", "subset", "episode", "highlight": [...]}`) navigates every connected tab there and paints quote-anchored highlights in the trace viewer. Reports cite traces with `[^id]` markers whose JSON definitions carry the same address plus a verbatim quote; the dashboard re-checks each quote against the trace files and marks the citation verified or broken, so answers stay grounded in what is actually on disk. The `dashboard` skill documents the full contract.

## Monitors

Monitors receive the run's metrics and episodes. The file and Prime monitors are on by default; W&B is opt-in.

### File

The file monitor writes the run's metrics (`metrics.jsonl`) and trace stream under `<run_dir>/monitors/file/`, and feeds the local [dashboard](#dashboard). Turn it off with `--no-monitors.file`.

### Weights & Biases

W&B is off by default:

```bash
uv run rl @ rl.toml --monitors.wandb                      # default project, random name
uv run rl @ rl.toml --monitors.wandb.project my-proj --monitors.wandb.name run-42
```

The trainer and orchestrator log into a **single shared W&B run**, so all metrics from both processes land in one place. Shared mode requires the W&B SDK ≥ 0.19.9 and is incompatible with `monitors.wandb.offline = true`.

prime-rl deliberately logs a **large number of metrics** for maximum observability: every rollout metric is emitted per subset (`all`/`effective`), per statistic (`mean`/`max`/`min`/`p10`/`p90`), and per environment alongside a cross-env aggregate, so a multi-env run can emit thousands of series. To keep that navigable, every training run (RL and SFT) gets an **auto-created `overview` saved view** curating the handful of metrics that matter into `train`, `eval`, `stability`, and `performance` sections (with per-env breakdowns). The view is created once per project and adapts to the run's environments; if a later run uses a different set of environments, a new versioned view (`overview-v2`, …) is created instead of overwriting the first.

### Prime

Every `rl` and `sft` run registers on the Prime Intellect platform (Prime Lab) and streams training metrics and episodes to the platform dashboard. The launcher prints the Prime run's link next to the local dashboard link (and the W&B run's, with `--monitors.wandb`) once the run registers. Turn it off with `--no-monitors.prime`:

```bash
uv run rl @ rl.toml --no-monitors.prime
uv run sft @ sft.toml --no-monitors.prime   # same flag on the SFT trainer
```

Name the Prime run in TOML:

```toml
[monitors.prime]
name = "my-experiment"
```

The process that supervises the launch registers the run before the components start, and every component logs to it. That is the `rl` or `sft` launcher, also inside a single-node SLURM job, or the run owner that wraps a multi-node SLURM job's step. The run's id is also the run identity (`PRL_RUN_ID`) and the W&B run id. On `rl` the orchestrator uploads the episodes and its metrics, and the trainer uploads its per-step metrics. A trainer started without the `rl` launcher needs `--no-monitors.prime`. On `sft` the run carries the dataset-batched training fields (`max_steps`, `batch_size`, `seq_len`) and the trainer's per-step metrics and validation losses. Online evals log to the same run, like an `rl` run's: the online-eval process uploads its eval metrics and every eval episode. The components only flush their uploads; the process that registered the run finishes it once every component exited. A launch that fails marks the run failed, and so does a SLURM job that SLURM cancels or stops at its time limit.

The platform run needs `PRIME_API_KEY` (`prime login` or the env var) and a team (`PRIME_TEAM_ID`, or the team selected with `prime login`) enabled for external runs. Like the W&B monitor, a run without a key, or one whose registration fails (no team, a team outside the allowlist), fails at launch: authenticate, select a team with `prime switch`, or pass `--no-monitors.prime`. `PRIME_RUNS_MODE=disabled` keeps the monitor configured but opens no platform run; `RUN_ID=<id>` attaches to an external run a launcher already created instead of registering a new one. Currently internal-only.

## Rules of Thumb

- **Start small.** Run `examples/basic/reverse-text/rl.toml` end-to-end on 2 GPUs before scaling. If the smoke run finishes cleanly, your install is good.
- **Batch size ≥ 64.** Smaller batches give noisy gradient estimates and the trainer's overhead-per-step dominates throughput. 64 is the practical floor; 128–512 is the range for quick ablations; production RL often runs at 1024+.
- **Group size ≥ 8.** Bigger groups (`orchestrator.train.group_size`) make it more likely that a task produces a mix of high- and low-reward rollouts, which is what gives the trainer a usable signal — if all rollouts in a group succeed or all fail, the within-group advantage collapses to zero and the trainer learns nothing from that task. Bigger groups also tighten advantage normalization. 8 is the floor; 16–32 is common.
- **Runs never share a directory.** Every launch writes to its own run directory `<output_dir>/<run_name>`, auto-named `<envs>--<model>--<short-id>` by default. Name runs you want to find again or resume with `--run.name <name>`; re-using a name blocks unless you resume or pass `--clean`.
- **Use `--dry-run` before SLURM.** Validators (e.g. CP needs flash-attention) fail fast in dry-run and slow in queue.
