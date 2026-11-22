# Qwen3.5-4B training recipes

This directory contains the supported Prime-RL v0.9 recipes for GlyphBench:

- `single_task.toml`: one task, one GRPO group per optimizer step.
- `multitask.toml`: balanced task-homogeneous groups drawn from a task manifest.
- `multitask_craftax100_tasks.txt`: a 100-task mixture weighted toward Craftax,
  used by the default multitask recipe.
- `hero100_tasks.txt`: the paper's balanced 100-task training mixture.
- `multitask_pilot_tasks.txt`: a small eight-task integration panel.
- `multitask_fixed_eval_tasks.txt`: fixed tasks used for periodic evaluation.

The TOML files are starting configurations: the single-task recipe uses a
512-rollout group, while the multitask recipe uses 16 groups of 64. Both default
to learning rate `1e-6`. The paper's final multitask experiment uses different
settings, given below. Adjust the Slurm partition and node counts for your
cluster, or render a local one-node configuration.

## Install

The release lock uses Transformers 5.10.4, overriding Prime-RL's older pin to
include the upstream [tokenizer path-traversal fix](https://github.com/huggingface/transformers/commit/eaaaf8494dd5386634ae37d1d122212fdc315be5).
The remaining GPU runtime versions stay pinned to the Prime-RL recipe.
The [dependency notes](../../../docs/DEPENDENCIES.md) describe remaining
advisories, the Qwen3.5-4B serving mitigation, and its limits. Review those
settings before changing the model or overriding vLLM configuration.

The RL stack is intentionally separate from a normal GlyphBench install because
it includes CUDA-specific PyTorch, vLLM, and FlashAttention packages.

```bash
uv sync --extra rl
```

Prime-RL writes under `runs/rl/`, which is ignored by Git. To use online W&B
logging, export `WANDB_API_KEY` in your shell or put it in the ignored `.env`
file. W&B entity selection follows your authenticated account; the public
configs do not name an organization.

## Validate a recipe

Prime-RL can resolve and validate a config without launching workers:

```bash
uv run rl @ configs/rl/qwen35-4b-glyphbench/single_task.toml --dry-run true
uv run rl @ configs/rl/qwen35-4b-glyphbench/multitask.toml --dry-run true
```

## Render a run-specific config

Use `scripts/rl/render_prime_rl_config.py` instead of editing a tracked recipe
for routine task, topology, or run-name changes.

```bash
# Single-task run on Slurm.
uv run python scripts/rl/render_prime_rl_config.py single \
  --output runs/rl/_rendered/snake.toml \
  --run-name snake-easy \
  --task glyphbench/classics-snake-easy-v0

# Multitask run with a custom repository-local task manifest.
uv run python scripts/rl/render_prime_rl_config.py multitask \
  --output runs/rl/_rendered/multitask.toml \
  --run-name multitask \
  --task-file path/to/tasks.txt \
  --group-size 64 --batch-size 1024
```

Then validate and launch through stock Prime-RL:

```bash
uv run rl @ runs/rl/_rendered/snake.toml --dry-run true
uv run rl @ runs/rl/_rendered/snake.toml
```

For Slurm, Prime-RL generates its standard launcher under the run directory.
Configure cluster-specific resources through that launcher and your generated
config.

## Paper's 100-task experiment

The paper trains one Qwen3.5-4B policy on `hero100_tasks.txt`, with 18 tasks each
from Classics, Craftax, Miniatari, and MiniGrid, 19 from MiniHack, and nine from
Procgen. Each update uses 1,024 rollouts in 32 groups of 32, with learning rate
`5e-6`. Render those settings explicitly:

```bash
uv run python scripts/rl/render_prime_rl_config.py multitask \
  --output runs/rl/_rendered/paper-multitask.toml \
  --run-name paper-multitask \
  --task-file configs/rl/qwen35-4b-glyphbench/hero100_tasks.txt \
  --group-size 32 --batch-size 1024 --learning-rate 5e-6 \
  --max-steps 250 --eval-interval 50 --eval-examples 4

uv run rl @ runs/rl/_rendered/paper-multitask.toml --dry-run true
```

A custom `--task-file` selects the same tasks for training and evaluation.
The command above evaluates 400 episodes: four held-out seeds for each of the
100 tasks. The training pool starts at seed 42 and contains 512 seeds; the
evaluation pool starts at 1,000,000. The default recipe instead evaluates a
50-task panel and should not be confused with this experiment.

The [README figure](../../../README.md#train-and-evaluate) reports the paper's
training and evaluation curves through update 250. It describes one training
run, not an average over training seeds.

## One-node smoke run

`--single-node` removes the Slurm block and divides one host between trainer
and inference workers. The following renders a small one-step integration run
for an eight-GPU node and disables W&B and checkpoint writes:

```bash
uv run python scripts/rl/render_prime_rl_config.py single \
  --output runs/rl/_rendered/single-node-smoke.toml \
  --run-name single-node-smoke \
  --single-node --train-gpus 4 --infer-gpus 4 \
  --group-size 8 --max-inflight 8 --max-steps 1 \
  --no-eval --no-checkpoint --no-wandb

uv run rl @ runs/rl/_rendered/single-node-smoke.toml --dry-run true
uv run rl @ runs/rl/_rendered/single-node-smoke.toml
```

For a multitask smoke run, use `multitask --pilot --group-size 8
--batch-size 8` with the same single-node flags. GPU splits are resource
controls, not paper-protocol settings; choose values that fit your hardware.

## Tracking and diagnostics

The reference configs log scalar training, rollout, evaluation, and system
metrics to W&B. Use `--wandb-project`, `--wandb-group`, and repeatable
`--wandb-tag` overrides when rendering a config. Use `--no-wandb` for an
entirely local run.

Full token-level trace files and the raw per-replica vLLM metric surface are
disabled by default because they are high volume. Enable them only for bounded
debugging with `--capture-traces` or `--raw-inference-metrics`. Inspect a saved
trace with:

```bash
uv run python scripts/rl/audit_prime_rl_trace.py path/to/traces.jsonl
```

## Reproducibility

Both recipes use the Verifiers v1 GlyphBench taskset, mixed training seeds,
fixed periodic evaluation seeds, Qwen3.5 thinking retention, IPO loss, prefix
caching, and a 65,536-token context capacity. The task manifests and exact
dependency revisions in `pyproject.toml`/`uv.lock` are part of the experiment
definition. Save the generated config with each run to record all overrides;
keep run artifacts untracked under `runs/`.
