# FAQs

Frequently-asked questions, grouped by topic. For full background see the linked pages.

## Table of Contents

- [Getting started](#getting-started)
- [Configs](#configs)
- [RL training](#rl-training)
- [SFT training](#sft-training)
- [Checkpoints and resume](#checkpoints-and-resume)
- [Scaling](#scaling)
- [Memory and OOM](#memory-and-oom)
- [Observability](#observability)
- [Models and environments](#models-and-environments)

## Getting started

### What's the fastest way to verify my install works?

The SFT debug config runs end-to-end on CPU or any single GPU with fake data:

```bash
uv run sft @ configs/debug/sft/train.toml
```

For the full RL stack on 2 GPUs, the GSM8K example is the smallest realistic run:

```bash
prime env install primeintellect/math-env
bash scripts/tmux.sh
uv run rl @ configs/gsm8k/rl.toml --wandb.name smoke-test --ckpt
```

See [Overview § Quick run](overview.md#quick-run).

### What hardware do I need?

Any NVIDIA GPU with compute capability ≥ 8.0 (RTX 3090/4090/5090, A100, H100, H200, B200). MoE training with FP8 needs SM ≥ 9.0 (Hopper or newer). RL needs at least 2 GPUs in practice (1 inference + 1 trainer), but you can co-locate both on a single GPU for the smallest debug runs.

### Why `uv run` and not `python`?

`uv` manages our virtualenv lock and pins dependencies precisely. Running `python` directly will pick up a different interpreter or miss extras (e.g. DeepGEMM). Always use `uv run <command>`.

## Configs

### My TOML file isn't being loaded — what's wrong?

Most common cause: missing space between `@` and the path. Use `@ rl.toml`, not `@rl.toml`. Otherwise the loader treats `@rl.toml` as a CLI flag.

If the file is loading but Pydantic complains, run `--dry-run --output-dir /tmp/x` to see exactly which field doesn't match the schema.

### How do I disable a feature that's enabled by default?

For an optional sub-config typed `SomeConfig | None`, pass the `--no-<name>` form on the CLI:

```bash
uv run rl @ rl.toml --no-trainer.gc        # disable garbage collection config
```

In TOML, comment out or remove the section.

### How do I override an env var in TOML?

You can't directly — env vars are a separate source. To force a fixed value, set it in TOML; the precedence order (CLI > TOML > env > defaults) means the TOML wins.

### How do I add a new environment to my training mix?

Add another `[[orchestrator.train.env]]` table. Lists are replaced wholesale on overlay, so include the full list every time:

```toml
[[orchestrator.train.env]]
id = "math-env"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }

[[orchestrator.train.env]]
id = "reverse-text"
```

See [Configuration § Environments](configuration.md#environments-orchestratortrainenv).

## RL training

### What does `max_async_level` actually do?

It caps how many steps inference can run ahead of training. `1` (default) is pipelined — inference for step n+1 runs concurrently with trainer step n; off-policy drift is minimal. `2` absorbs slower weight broadcasts (e.g. cross-WAN). Higher values give more throughput at the cost of more drift; watch `mismatch_kl/all/mean`. See [Algorithms § Tuning `max_async_level`](algorithms.md#tuning-max_async_level).

### Why are there two W&B runs per RL job?

The trainer and orchestrator log as separate runs so their step indices and timings stay independent. The names are `<your-name>-trainer` and `<your-name>-orchestrator`. Group them in W&B if you want a unified view.

### My reward isn't improving. What should I check first?

In order:

1. `reward/all/mean` and `reward/all/std`. If std is ~0, the env is too easy or rewards are degenerate — increase difficulty or check the rubric.
2. `is_truncated/all/mean`. If high, your model is hitting `max_completion_tokens` — either raise it or the model isn't learning to stop.
3. Eval scores vs train rewards. If train reward rises but eval is flat, you may be hitting a chat-template prefix violation; see [Algorithms § Multi-turn trajectories](algorithms.md#multi-turn-trajectories).
4. `mismatch_kl/all/mean`. If growing, drop `max_async_level` or LR.
5. `optim/grad_norm`. Sustained spikes mean you're about to diverge — drop LR.

### How do I evaluate without training?

Use `vf-eval`:

```bash
uv run vf-eval math-env \
  -a '{"dataset_name": "openai/gsm8k", "dataset_subset": "main"}' \
  -m PrimeIntellect/Qwen3-0.6B \
  -b http://localhost:8000/v1 -n 50 -t 2048
```

This talks to any OpenAI-compatible endpoint, so it works against `uv run inference`, hosted endpoints, or a stale checkpoint mid-run.

### What's the difference between `training_mode = "sft"` and the standalone `uv run sft`?

`uv run sft` is the traditional path: load a HF dataset, train the model. No orchestrator, no teacher.

`orchestrator.training_mode = "sft"` uses the RL pipeline to hard-distill from a teacher: the teacher (any OpenAI-compatible endpoint) generates the completions, and the student trains on them as they're produced. Use this when you want on-the-fly teacher supervision against a moving student. See [Training § Training modes](training.md#training-modes-rl--opd--sft-via-orchestrator).

## SFT training

### Why does Qwen3 fail multi-turn SFT silently?

Qwen3's default chat template strips past `<think>` blocks when re-tokenizing, which violates the prefix property the SFT trainer depends on. Use a model with a patched chat template — we ship one at `PrimeIntellect/Qwen3-0.6B`. See [Algorithms § Multi-turn trajectories](algorithms.md#multi-turn-trajectories).

### Can I train on `prompt`/`completion` and `messages` mixed in one dataset?

Yes — if both columns are present in a row, `messages` takes precedence. The trainer will use `messages` for that row and ignore `prompt`/`completion`.

### How do I do tool-calling SFT?

Tool-calling SFT works out of the box if your dataset uses the `messages` format with tool messages embedded. The renderer handles tool turns the same as assistant turns. Make sure your model's chat template supports tool tokens.

## Checkpoints and resume

### How often should I checkpoint?

For long runs: every 50–100 steps with `--ckpt.keep-last 3` for rolling rollback and `--ckpt.keep-interval 500` for permanent milestones. For short experiments: end-of-training only (`--ckpt` with no interval). See [Training § Checkpointing](training.md#checkpointing).

### How do I resume from the latest checkpoint?

```bash
uv run rl @ rl.toml --max-steps 100 --ckpt.resume-step -1   # -1 means latest
```

`--max-steps` is the absolute target, not the remainder.

### Can I serve a mid-training weight checkpoint?

Yes. HF-compatible weights are written to `<output_dir>/weights/step_<N>/` on every checkpoint. Use them directly:

```bash
uv run inference --model.name outputs/weights/step_500
```

or upload to HF:

```bash
uv run hf upload <user>/<model>-RL-500 outputs/weights/step_500
```

### Can the inference server stay running across trainer restarts?

Yes. The orchestrator pushes the resumed checkpoint into inference automatically. No need to restart the inference server.

## Scaling

### Single GPU is too tight. What's the minimum useful layout?

2 GPUs — one for inference, one for the trainer. The default placement in `rl` does exactly this. See [Scaling § Single-node multi-GPU](scaling.md#single-node-multi-gpu).

### Multi-node without SLURM or K8s?

Yes, see [Scaling § Multi-node (manual)](scaling.md#multi-node-manual). You need a shared filesystem and a reachable inference IP. Set the three `OUTPUT_DIR` / `INFERENCE_SERVER_IP` / `INFERENCE_SERVER_API_KEY` env vars on every node and launch each process by hand.

### How big a difference does NCCL weight broadcast make?

NCCL broadcast is much faster than filesystem for local-cluster setups, at the cost of being synchronous: it requires `max_async_level = 1` and doesn't support LoRA today. Use it for multi-node single-cluster RL where you want maximum throughput; stick with filesystem for cross-WAN, LoRA, or async-heavy setups.

### Should I use TP, DP, EP, or CP?

- **TP (inference)**: scale within a node, up to `num_attention_heads / 2`. Past that, returns diminish.
- **DP (inference and trainer)**: scale throughput linearly across replicas. Default scaling lever.
- **EP (trainer, MoE only)**: shards expert weights; the right knob for MoE memory and throughput together.
- **CP (trainer)**: shards a sequence across GPUs along the token axis. Needed for sequences past ~32K tokens. Stick to CP ≤ 8.

See [Scaling § Parallelism knobs](scaling.md#parallelism-knobs).

## Memory and OOM

### Trainer CUDA OOM — what should I try first?

In order:

1. `--model.ac` (full activation checkpointing).
2. Lower `seq_len` or `data.micro_batch_size`.
3. `--model.optim-cpu-offload` (offloads only optimizer state).
4. `--model.cp 2` (context parallelism; requires flash-attention and the custom impl).
5. `--model.fsdp-cpu-offload` as a last resort (significant throughput hit).

The kitchen-sink config in [Scaling § Memory-tight recipe](scaling.md#memory-tight-recipe) combines all of the above.

### Inference CUDA OOM?

Tighten `inference.gpu_memory_utilization` (try 0.85), lower `inference.model.max_model_len`, or split inference across more GPUs with `inference.parallel.dp`.

### Why is `optim_cpu_offload` not slowing me down much?

In RL you typically take many gradient-accumulation micro-steps per optimizer step, so the H2D/D2H transfer is amortized. In pretraining the cost is more visible.

## Observability

### Where's the log file for a specific environment worker?

`<output_dir>/logs/envs/{train,eval}/<env_name>/env_worker_<id>.log`. Most silent training kills come from OOM in env workers — start there.

### How do I get more verbose env logging?

```bash
uv run rl @ rl.toml --orchestrator.log.vf-level debug
```

Or set `PRIME_VF_LOG_LEVEL=debug` in the environment.

### vLLM is logging too much. Can I quiet it?

Set `inference.log.level = "warning"` (or pass `--inference.log.level warning`). Note that `inference.log` only controls the prime-rl logger; vLLM's own logging is controlled by `VLLM_LOGGING_LEVEL` env var.

### What's the fastest way to see KV cache pressure?

```bash
curl -s http://localhost:8000/metrics | grep gpu_cache_usage_perc
```

Approaching 1.0 means KV cache is saturated and request latency will spike. Reduce `max_model_len` or split across more inference GPUs.

## Models and environments

### Which models have a custom optimized implementation?

GLM-5, Qwen3 MoE, Qwen3.5 MoE, Qwen3 / Qwen3.5 VLMs, Poolside Laguna, MiniMax M2, Nemotron H, Trinity (AFMoE), GLM-4 / GLM-4.5 / INTELLECT-3, GPT-OSS (HF-MoE only). See the table in [Advanced § MoE models](advanced.md#moe-models).

Other HF causal LMs work via the HF path (`impl = "hf"` or `"auto"`) but without EP, FP8, or the custom kernels.

### Can I train a VLM?

Yes — Qwen3-VL, Qwen3.5, Qwen3.5-MoE out of the box. Add `[model.vlm]` and use bfloat16 dtypes. See [Advanced § Vision-language models](advanced.md#vision-language-models).

### How do I install an environment from the Environments Hub?

```bash
prime env install primeintellect/math-env
uv run python -c "import math_env"   # verify
```

Then reference by ID in your config. See [Advanced § Environments](advanced.md#environments).

### Can I install an environment from outside the Hub?

Yes — install with `uv pip install -e path/to/my-env` and reference it by its `id` (the env's package name). The orchestrator will discover it.

### My environment hangs occasionally. What's happening?

Most likely it's running user code that blocks on a network call or an external service (e.g. a math verifier, a sandbox). Check the env worker logs and the event-loop lag metrics on the env server. The orchestrator's `max_retries` and `errored_rollouts` metric should tell you how often rollouts fail vs hang.
