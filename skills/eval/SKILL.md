---
name: eval
description: Launch and monitor prime-rl evals — the `uv run eval` entrypoint, its config and CLI shorthands, run directory, resume, logs and metrics. Use when asked to evaluate a model or checkpoint on an environment, smoke-test an environment, or check on an eval run.
---

# Eval

`uv run eval` evaluates a model in one or more environments and exits after one epoch per source. It reuses the orchestrator's eval pipeline: one env server per source, a concurrency band, every episode through the monitors, and a trace stream that `--resume` continues from. Online evals of training runs are the `training` skill.

## Start an eval

The user launches runs; hand over the command unless told otherwise. Always cap smokes (`-n`, `-r`).

```bash
uv run eval gsm8k -n 32 -r 4                                       # Prime Inference
uv run eval gsm8k -n 32 -r 4 -c 8 --env.agent.harness.id bash      # pin concurrency; set a field of the env block
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1   # a local `uv run inference` server
uv run eval @ eval.toml --run.name my-eval                          # multi-source TOML
uv run eval @ eval.toml --run.name my-eval --resume                 # continue an interrupted run
uv run eval @ eval.toml --dry-run                                   # resolve and write the config, exit
uv run eval @ eval.toml --monitors.prime                            # stream each source's epoch to the platform
```

- Config: `EvalConfig` (`packages/prime-rl-configs/src/prime_rl/configs/eval.py`); `uv run eval -h` lists the fields.
- Entrypoint: `src/prime_rl/entrypoints/eval.py`; implementation `src/prime_rl/eval/eval.py`, shared engine `src/prime_rl/eval/runner.py`.
- Shorthands (single-source runs): `<taskset-id>`, `--env.<field> <value>`, `-n` `num_examples`, `-r` `group_size`, `-m` `model`, `-c N` pins `concurrency.min_inflight = max_inflight = N`. They cannot be combined with a TOML that defines `[[source]]` blocks.
- Concurrency: an API exposes no vLLM `/metrics`, so run pinned there; against `uv run inference` set `min_inflight < max_inflight` in `[concurrency]` to adapt to KV usage.
- Env servers: spawned per source on OS-assigned ports (published to `configs/attempt_N/resolved/envs/eval/<name>.address`) unless the source sets `serve.address`.
- Ready-made configs: `configs/debug/eval/*.toml`, one per shape (single turn, multi turn, resume, multi env, aime2026, tb2).

Minimal multi-source TOML (the eval block is flattened to the top level; per-source `num_examples`, `group_size`, `sampling` override the top level):

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[[source]]
env.taskset.id = "gsm8k"
env.agent.harness.id = "bash"

[[source]]
env.taskset.id = "aime25"
env.agent.harness.id = "null"
env.agent.runtime.type = "subprocess"
```

## Monitor an eval

Run dir: `output_dir / run.name` (auto `<envs>--<model>--<short-id>`; `ls -t outputs | head -1` finds the latest). The console stays quiet while the eval runs; everything is in the files and the dashboard (`dashboard` skill).

```
{run_dir}/
├── configs/latest/            # command.txt, the launch TOML, resolved/eval.json
├── logs/latest/
│   ├── eval.log               # the eval process
│   └── envs/eval/{name}.log   # one log per env server
└── monitors/file/             # metrics.jsonl, the trace stream, traces/live/ (one file per live trace), plan.json
```

```bash
tail -F {run_dir}/logs/latest/eval.log
grep -E "WARNING|ERROR" {run_dir}/logs/latest/eval.log {run_dir}/logs/latest/envs/eval/*.log
grep SUCCESS {run_dir}/logs/latest/eval.log                            # one "Evaluated <env> ... Reward 0.xxxx" line per source
uv run python -m prime_rl.monitors.file.traces {run_dir} [<trace_id>]   # live rollouts by phase, or one assembled live trace
```

The progress line in `eval.log` counts live rollouts by phase (`- boot 1 · running 3`). A rollout stuck in `boot` for minutes is waiting on its sandbox; one in `running` with a frozen turn count is waiting on a model call or tool.

Metrics live under `eval/<env>/all/<agent>/…` (`reward/mean`, `is_truncated/mean` — raise `sampling.max_completion_tokens` when high, `has_error/mean`, the taskset's own metrics). Validate a result by reading a few traces in the dashboard rather than trusting the mean alone.

Stop a run with SIGINT/SIGTERM to the eval PID (`ps aux | grep PRL::Eval`); `--resume` restores the landed episodes from the trace stream and runs only the rollouts still owed (`num_examples`/`group_size` may change, the model, sampling and env config may not). Env servers are children of the eval process and exit with it.
