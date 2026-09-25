# Eval

This page covers `uv run eval` — evaluating a model in one or more environments. For the online evals of a training run, see [Training](training.md#online-evals).

> **AI agents working in this repo:** the equivalent runbook is at [`skills/eval/SKILL.md`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/skills/eval/SKILL.md).

## Table of Contents

- [Launch](#launch)
- [Configuration](#configuration)
- [Resume](#resume)
- [Monitors](#monitors)
  - [File](#file)
  - [Prime](#prime)
- [Metrics](#metrics)

`uv run eval` runs one epoch of every configured source and exits. It reuses the orchestrator's eval pipeline: env servers are spawned per source, episodes are admitted under the concurrency controller, and every episode streams into the run's trace stream and metrics.

## Launch

By default the model is served by Prime Inference; authenticate with `PRIME_API_KEY` or `prime login`:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8                                   # Prime Inference
uv run eval @ configs/debug/eval/single-turn.toml                   # the same shape as a TOML
```

To evaluate a model you serve yourself, start a `uv run inference` vLLM server (or any OpenAI-compatible API) and point the client at it:

```bash
uv run inference --vllm.model Qwen/Qwen3-4B
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1
```

Single-source shorthands: `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block (`--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`), `-n`/`-r` set `num_examples`/`group_size`, `-m` the model, and `-c N` pins the concurrency band (`concurrency.min_inflight = max_inflight = N`). The shorthands cannot be combined with a TOML that defines `[[source]]` blocks. `uv run eval -h` lists them.

Against a local vLLM deployment, set `min_inflight < max_inflight` in `[concurrency]` to dynamically adjust the number of concurrent episodes for maximum throughput. An external API exposes no vLLM `/metrics` to adapt to, so pin the concurrency there (`-c N`, i.e. `min_inflight = max_inflight`).

## Configuration

Multi-source runs use a TOML (`EvalConfig` in `packages/prime-rl-configs/src/prime_rl/configs/eval.py`). The eval block is flattened to the top level — `[[source]]`, `[client]`, `[concurrency]`, `[sampling]`, `num_examples`, `group_size`, `min_rollouts` — and each source takes the same `env` block as `[[orchestrator.eval.source]]`:

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]        # adaptive against a local vLLM deployment
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

Per-source `num_examples`, `group_size`, `min_rollouts`, and `sampling` override the top-level defaults. Every source's env server is spawned by the eval process unless the source sets `serve.address`, in which case the server is externally managed. A spawned server binds an OS-assigned loopback port and publishes it to `configs/attempt_N/resolved/envs/eval/<name>.address`, which the eval process reads, so concurrent runs on one host never collide on a port.

By default, eval runs every task once. Set `group_size` to run each task a fixed number of times. Set `min_rollouts` to target a total rollout count. PRL uses the resolved task count to choose the smallest `group_size` that reaches the target. For example, 500 tasks get `avg@2` for a target of 1,000, while 200 tasks get `avg@5`.

Set `min_rollouts` globally or on one `[[source]]`. It is mutually exclusive with `group_size` at the same level. A source can override the global rollout setting. The same setting works under `[orchestrator.eval]` in RL and `[eval]` in SFT. Config resolution counts tasks before dry runs and live runs. An infinite taskset needs a positive `num_examples` bound. Resolved configs contain the computed `group_size`.

## Resume

An interrupted run resumes from its trace stream. Relaunch with the same `--run.name` and `--resume`: the episodes that landed rejoin the epoch as if they had just arrived (stream, metrics and platform upload cover the whole epoch) and only the rollouts still owed run. Errored episodes and the ones the interruption cut off run again.

```bash
uv run eval @ eval.toml --run.name my-eval
# <interrupted>
uv run eval @ eval.toml --run.name my-eval --resume
```

The previous attempt's `monitors/file` is kept as `monitors/file.attempt_N`; the resumed attempt writes a fresh one. Nothing is deleted, and a resume reads every attempt's stream.

A landed episode counts toward the task with its `task.key`, so `num_examples` and `group_size` may change between the two launches: kept episodes are matched to the new selection and the rest is owed. The model, the sampling and each source's env must match the config the landed episodes were measured with (kept beside them in `monitors/file/eval.json`); a resume that changes them stops with the differing config paths. Rollouts that complete a task's landed group join that group, so pass@k and the dashboard see one group per task. Use `--clean` to start over instead.

## Monitors

Monitors receive the run's metrics and episodes through one abstraction. The file monitor is on by default; `--monitors.wandb` logs the run to Weights & Biases; `--monitors.prime` streams it to the Prime platform.

### File

The file monitor writes the run's metrics and trace stream under `output_dir / run.name / monitors/file/` and feeds the local [dashboard](training.md#dashboard). Env servers stream every rollout to the eval process as it happens: the trace's header when it is minted, then one delta per committed turn (the new messages and the model call behind them), per phase change (boot, setup, agent, finalize, scoring), and a preview of each model request's uncommitted messages, so a tool result shows the moment the model is asked about it rather than with the reply. The file monitor appends those deltas to `monitors/file/traces/live/<trace_id>.jsonl` and deletes the file when the episode lands in the finished trace stream, so that directory is exactly the live set and finished traces stay where they always were. The dashboard's traces tab shows live rollouts in the episode table itself: a pulsing dot in the `#` cell marks the row, a phase badge says where the rollout is (boot, setup, running, finalize, scoring), the `dispatched → arrived` column shows the dispatch time with the elapsed time in brackets, and the turn, token and branch counts grow as deltas land. The status filter toggles live and completed rows, and a live row opens in the trace viewer, which follows the rollout turn by turn (the transcript stays pinned to the newest turn until you scroll up; folded entries stay folded; the pending messages of the request in flight render dimmed with an `awaiting model` chip). The table and an open live trace poll once a second and transfer nothing while the rollout is unchanged. The progress line in `eval.log` counts the live rollouts by phase (`2 inflight episodes ... - boot 1 · running 1`). Read a live trace from a shell with `uv run python -m prime_rl.monitors.file.traces <run_dir> [<trace_id>]`. The same stream exists for RL runs, covering train and online-eval rollouts alike.

The dashboard's overview tab reads the trace stream directly. It shows one env at a time; the filter menu picks the env and whether errored episodes join the distributions. A block bar has one cell per expected episode: landed episodes (errored ones in red; click one to open it), live rollouts as pulsing outlines (click one to follow it), and the rest empty, with the percentage beside it. The expected count is what the eval runner wrote to `monitors/file/plan.json` when it counted the epoch's tasks. Below the bar, a summary row gives the headline numbers: avg@k and pass@k, the error and truncation rates (yellow past 10%, red past 50%), mean turns and branches in one tile, mean episode time and tokens, and the total cost when the run has costs; hover a tile for its distribution. The env's own rewards and metrics follow, a beeswarm pane each, resizable like the training charts. The usage section is one pane splitting each episode's tokens into input and output, and the timing section is the same pane over the phases of a rollout (boot, setup, model, harness, finalize, scoring, and `other` for wall time no phase accounts for): an icicle bar shows the mean split, one strip per episode below shows the split per rollout, and a legend sits above. Hovering a phase highlights it across the pane; the strips show 40 episodes by default and all of them on `show all`. A metric every episode shares appears as a chip instead of a pane, and a metric with only a few distinct values, such as a binary reward, as a counted dot plot with the count and share of each value. Every dot is an episode (or a task): hover it for its value and click it to open the trace; hover the pane's background for the summary (min, p10, median, p90, max), with a faint boxplot behind the dots. Errored episodes stay out of the distributions until the filter includes them, where they read red. The metrics tab charts every key of the run's `metrics.jsonl`: the epoch's `eval/<env>/…` summaries and the time-keyed `inference/` and `concurrency/` rows the runner samples while the epoch is in flight.

### Prime

`--monitors.prime` streams each source's epoch as an evaluation on the Prime platform:

```bash
uv run eval gsm8k -n 32 -r 4 -c 8 --monitors.prime
```

The monitor opens one platform evaluation per env when its epoch starts, streams every episode to it as it lands, and closes it with the epoch's aggregates. The run leaves its platform identity in `monitors/prime/run.json` (the training run's id and URL, or one entry per eval epoch), and the dashboard's top bar shows a `view on platform` link from the moment the evaluation opens: a direct link for a training run or a single evaluation, a menu when a run has several.

## Metrics

Eval metrics mirror the training rollout hierarchy under the `eval/<env>` scope:

| Metric | Reading |
|---|---|
| `eval/<env>/all/<agent>/reward/mean` | mean reward over the epoch |
| `eval/<env>/all/<agent>/is_truncated/mean` | share of rollouts cut by the length limit |
| `eval/<env>/all/<agent>/has_error/mean` | share of rollouts that raised |
| `eval/<env>/all/seq_len/mean` | mean episode length in tokens |

Task-specific env metrics (e.g. `correct_answer`, `format`) appear under the same prefix. The console summary line per source (`Evaluated <env> ... Reward 0.8125`) reports the epoch mean.
