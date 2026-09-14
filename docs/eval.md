# Eval

This page covers `uv run eval` — evaluating one or more environments against a live inference server — and where the online evals of a training run come from. For launching and observing training runs, see [Training](training.md).

> **AI agents working in this repo:** the equivalent runbook is at [`skills/eval/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/eval/SKILL.md).

## Table of Contents

- [Standalone Evals](#standalone-evals)
  - [Launch](#launch)
  - [Configuration](#configuration)
  - [Run Directory and Resume](#run-directory-and-resume)
  - [Monitors and Platform Upload](#monitors-and-platform-upload)
- [Online Evals](#online-evals)
- [Metrics](#metrics)

## Standalone Evals

`uv run eval` evaluates one or more environments against a live inference server and exits after one epoch per source. It reuses the orchestrator's eval pipeline: env servers are spawned per source, episodes are admitted under the adaptive concurrency controller, and every episode streams into the run's trace stream and metrics.

### Launch

The default client is Prime Inference with `deepseek/deepseek-v4.1-flash`. Authenticate with `PRIME_API_KEY` or `prime login`:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8                                   # Prime Inference, the default client
uv run eval @ configs/debug/eval/gsm8k.toml                         # the same run as a TOML
```

To evaluate a model you serve yourself, start a `uv run inference` vLLM server (or any OpenAI-compatible API) and point the client at it:

```bash
uv run inference --vllm.model Qwen/Qwen3-4B
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1
```

Single-source shorthands: `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block (`--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`), `-n`/`-r` set `num_examples`/`group_size`, `-m` the model, and `-c N` pins the concurrency band (`concurrency.min_inflight = max_inflight = N`). The shorthands cannot be combined with a TOML that defines `[[source]]` blocks. `uv run eval -h` lists them.

Against vLLM the concurrency band adapts to KV usage like the orchestrator's. External APIs expose no vLLM `/metrics`, so there the band must be pinned (`-c N`, or `min_inflight = max_inflight` in `[concurrency]`) or the startup `/metrics` probe fails fast.

### Configuration

Multi-source runs use a TOML (`EvalConfig` in `packages/prime-rl-configs/src/prime_rl/configs/eval.py`). The eval block is flattened to the top level — `[[source]]`, `[client]`, `[concurrency]`, `[sampling]`, `num_examples`, `group_size` — and each source takes the same `env` block as `[[orchestrator.eval.source]]`:

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]
max_inflight = 128

[sampling]
max_completion_tokens = 2048

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

Per-source `num_examples`, `group_size` and `sampling` override the top-level defaults. Every source's env server is spawned by the eval process at `tcp://127.0.0.1:<env_server_base_port + index>` unless the source sets `serve.address`, in which case the server is externally managed.

The basic examples ship an `eval.toml` next to their `sft.toml` and `rl.toml` (e.g. [`examples/basic/reverse-text/eval.toml`](../examples/basic/reverse-text/eval.toml)) for the baseline and final evals of the walkthrough; override the model with `-m` to evaluate a trained checkpoint. Smoke configs against Prime Inference live in [`configs/debug/eval/`](../configs/debug/eval).

### Run Directory and Resume

The run writes to `output_dir / run.name` (auto-generated as `<envs>--<model>--<short-id>`) with the same layout as training runs: `configs/attempt_<n>/` (the launch command, TOML, and resolved `eval.json`), `logs/attempt_<n>/eval.log` plus one `envs/eval/<name>.log` per source, and `monitors/file/` with `metrics.jsonl` and the trace stream. `--clean` wipes a used run directory, `--dry-run` writes the config and exits, `--no-dashboard` skips the dashboard daemon.

The console shows the start line, log paths and dashboard URL, then only per-env results and warnings; `eval.log` keeps everything.

The task cursor is checkpointed after every completed group (`[ckpt]`: `interval` counts completed groups, `keep_last` prunes older cursors; disable with `--no-ckpt`). Relaunch with the same `--run.name` and `--resume` (or `--resume.step N`) to skip the completed prefix; partially completed groups are retried:

```bash
uv run eval @ eval.toml --run.name my-eval
uv run eval @ eval.toml --run.name my-eval --resume
```

### Monitors and Platform Upload

The file monitor is on by default and feeds the [dashboard](training.md#dashboard). `--monitors.wandb` logs the run to Weights & Biases. `--monitors.prime` uploads each source's finished epoch as an evaluation on the Prime Intellect platform (one evaluation per source, named after the run) and logs its URL:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8 --monitors.prime
```

## Online Evals

Training runs evaluate the live policy on a step interval with the same eval sources:

- **RL** — the `[orchestrator.eval]` block (`RLOnlineEvalConfig`): `interval`, `[[orchestrator.eval.source]]` with per-source overrides, and `skip_first_step` / `retrigger_on_resume`. The orchestrator runs the epochs against its inference pool; see [Training § Useful Knobs](training.md#useful-knobs).
- **SFT** — the `[eval]` block (`SFTOnlineEvalConfig`) plus an `[inference]` block. The `sft` launcher starts the inference server, one env server per source, and an online-eval process that evaluates every weight broadcast; see [Training § Online Evals](training.md#online-evals).

Both write their eval episodes and metrics into the training run's monitors, so results show up next to the training curves in the dashboard and W&B; the SFT online-eval process logs to `logs/attempt_<n>/eval.log`.

## Metrics

Eval metrics mirror the training rollout hierarchy under the `eval/<env>` scope:

| Metric | Reading |
|---|---|
| `eval/<env>/all/<agent>/reward/mean` | mean reward over the epoch |
| `eval/<env>/all/<agent>/is_truncated/mean` | share of rollouts cut by the length limit |
| `eval/<env>/all/<agent>/has_error/mean` | share of rollouts that raised |
| `eval/<env>/all/seq_len/mean` | mean episode length in tokens |
| `eval/<env>/policy_version` | the policy the epoch measured (online evals) |

Task-specific env metrics (e.g. `correct_answer`, `format`) appear under the same prefix. The console summary line per source (`Evaluated <env> ... Reward 0.8125`) reports the epoch mean.
