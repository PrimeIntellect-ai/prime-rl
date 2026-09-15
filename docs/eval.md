# Eval

This page covers `uv run eval` — evaluating one or more environments against a live inference server — and where the online evals of a training run come from. For launching and observing training runs, see [Training](training.md).

> **AI agents working in this repo:** the equivalent runbook is at [`skills/eval/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/eval/SKILL.md).

## Table of Contents

- [Standalone Evals](#standalone-evals)
  - [Launch](#launch)
  - [Configuration](#configuration)
  - [Run Directory and Resume](#run-directory-and-resume)
  - [Live View](#live-view)
  - [Monitors and Platform Upload](#monitors-and-platform-upload)
- [Online Evals](#online-evals)
- [Metrics](#metrics)

## Standalone Evals

`uv run eval` evaluates one or more environments against a live inference server and exits after one epoch per source. It reuses the orchestrator's eval pipeline: env servers are spawned per source, episodes are admitted under the concurrency controller (pinned at 128 by default), and every episode streams into the run's trace stream and metrics.

### Launch

The default client is Prime Inference with `deepseek/deepseek-v4.1-flash`. Authenticate with `PRIME_API_KEY` or `prime login`:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8                                   # Prime Inference, the default client
uv run eval @ configs/debug/eval/single-turn.toml                   # the same shape as a TOML
```

To evaluate a model you serve yourself, start a `uv run inference` vLLM server (or any OpenAI-compatible API) and point the client at it:

```bash
uv run inference --vllm.model Qwen/Qwen3-4B
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1
```

Single-source shorthands: `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block (`--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`), `-n`/`-r` set `num_examples`/`group_size`, `-m` the model, and `-c N` pins the concurrency band (`concurrency.min_inflight = max_inflight = N`). The shorthands cannot be combined with a TOML that defines `[[source]]` blocks. `uv run eval -h` lists them.

Concurrency is pinned at 128 in-flight episodes by default; `-c N` repins it. External APIs expose no vLLM `/metrics` to adapt to, so the pin is what they run with. Against a vLLM server, set `min_inflight < max_inflight` in `[concurrency]` to let the band adapt to KV usage like the orchestrator's.

### Configuration

Multi-source runs use a TOML (`EvalConfig` in `packages/prime-rl-configs/src/prime_rl/configs/eval.py`). The eval block is flattened to the top level — `[[source]]`, `[client]`, `[concurrency]`, `[sampling]`, `num_examples`, `group_size` — and each source takes the same `env` block as `[[orchestrator.eval.source]]`:

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]        # adaptive against vLLM (the default pins 128)
min_inflight = 8
max_inflight = 256

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

Per-source `num_examples`, `group_size` and `sampling` override the top-level defaults. Every source's env server is spawned by the eval process unless the source sets `serve.address`, in which case the server is externally managed. A spawned server binds an OS-assigned loopback port and publishes it to `configs/attempt_N/resolved/envs/eval/<name>.address`, which the eval process reads, so concurrent runs on one host never collide on a port.

The basic examples ship an `eval.toml` next to their `sft.toml` and `rl.toml` (e.g. [`examples/basic/reverse-text/eval.toml`](../examples/basic/reverse-text/eval.toml)) for the baseline and final evals of the walkthrough; override the model with `-m` to evaluate a trained checkpoint. Smoke configs against Prime Inference live in [`configs/debug/eval/`](../configs/debug/eval), one per shape: single turn, multi turn (a sandboxed terminal task), resume, and multi env, plus AIME 2026 at avg@16 and Terminal-Bench 2 at avg@4.

### Run Directory and Resume

The run writes to `output_dir / run.name` (auto-generated as `<envs>--<model>--<short-id>`) with the same layout as training runs: `configs/attempt_<n>/` (the launch command, TOML, and resolved `eval.json`), `logs/attempt_<n>/eval.log` plus one `envs/eval/<name>.log` per source, and `monitors/file/` with `metrics.jsonl` and the trace stream. `--clean` wipes a used run directory, `--dry-run` writes the config and exits, `--no-dashboard` skips the dashboard daemon.

Like the training launchers, the console shows the start line, log paths and dashboard URL, stays quiet while the eval runs (only errors surface), and ends with the success line and the dashboard URL again. Results and progress are in the dashboard; `eval.log` keeps everything.

The task cursor is checkpointed after every completed group (`[ckpt]`: `interval` counts completed groups, `keep_last` prunes older cursors; disable with `--no-ckpt`). Relaunch with the same `--run.name` and `--resume` (or `--resume.step N`) to skip the completed prefix; partially completed groups are retried:

```bash
uv run eval @ eval.toml --run.name my-eval
uv run eval @ eval.toml --run.name my-eval --resume
```

### Live View

Env servers stream every rollout to the eval process as it happens: the trace's header when it is minted, then one delta per committed turn (the new messages and the model call behind them) and per phase change (boot, setup, agent, finalize, scoring). The file monitor appends those deltas to `monitors/file/traces/live/<trace_id>.jsonl` and deletes the file when the episode lands in the finished trace stream, so that directory is exactly what is in flight and finished traces stay where they always were. The dashboard's traces tab shows in-flight rollouts in the episode table itself: a pulsing dot in the `#` cell marks the row, a phase badge says where the rollout is (boot, setup, running, finalize, scoring), the `dispatched → arrived` column shows the dispatch time with the elapsed time in brackets, and the turn, token and branch counts grow as deltas land. The status filter toggles in-flight and completed rows, and a live row opens in the trace viewer, which follows the rollout turn by turn (the transcript stays pinned to the newest turn until you scroll up; folded entries stay folded). The table and an open live trace poll once a second and transfer nothing while the rollout is unchanged. The progress line in `eval.log` counts the in-flight rollouts by phase (`2 inflight episodes ... - boot 1 · running 1`). Read a live trace from a shell with `uv run python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]`. The same stream exists for RL runs, covering train and online-eval rollouts alike.

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

The dashboard's metrics tab reads the trace stream directly. It shows one env at a time; the filter menu picks the env and whether errored episodes join the distributions. A block bar has one cell per expected episode: landed episodes (errored ones in red; click one to open it), in-flight rollouts as pulsing outlines (click one to follow it), and the rest empty, with the percentage beside it. The expected count is what the eval runner wrote to `monitors/file/eval_plan.json` when it counted the epoch's tasks. Below the bar, each distribution the episodes carry gets a beeswarm pane, resizable like the training charts: avg@k and pass@k over tasks, the env's own rewards and metrics, rollout time, input and output tokens, turns, branches and cost. Timing is one pane for the whole hierarchy: an icicle bar shows how the zoomed node's mean time splits into its children (boot, setup, agent, finalize, scoring under the episode; model and harness under agent), one strip per episode below shows the same split per rollout, and clicking a segment, a breadcrumb step or a node in the tree legend zooms. A metric every episode shares appears as a chip instead of a pane, and a metric with only a few distinct values, such as a binary reward, as a counted dot plot with the count and share of each value. Every dot is an episode (or a task): hover it for its value and click it to open the trace; hover the pane's background for the summary (min, p10, p25, median, mean, p75, p90, max), which a faint boxplot behind the dots also sketches. Errored episodes stay out of the distributions until the filter includes them, where they read red.
