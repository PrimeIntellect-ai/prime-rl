---
name: eval
description: Launch and monitor prime-rl evals — the `uv run eval` entrypoint (standalone evals of any model against Prime Inference or a local vLLM server), its config and CLI shorthands, run directory, resume, logs, metrics, and platform upload. Use when asked to evaluate a model or checkpoint on an environment, smoke-test an environment, or check on an eval run.
---

# Eval

`uv run eval` evaluates a model in one or more environments and exits after one epoch per source. It reuses the orchestrator's eval pipeline: one env server per source, concurrency pinned at 128 by default (adaptive on request), every episode streamed through the monitors (file monitor + trace stream by default; W&B and the Prime platform on request), and a task cursor checkpointed after every completed group so an interrupted run resumes with `--resume`. Online evals of training runs (`[orchestrator.eval]` for RL, `[eval]` for SFT) share the same source shape — see the `training` skill for those.

Two phases — start the eval, then read its results.

## Start an eval

The user launches runs; hand over the command unless told otherwise. Always cap the size of smokes (`-n`, `-r`).

```bash
uv run eval gsm8k -n 32 -r 4                                     # Prime Inference (default client + model), 128 in flight
uv run eval gsm8k -n 32 -r 4 -c 8                                # repin the band
uv run eval gsm8k -n 32 -r 4 -c 8 --env.agent.harness.id bash    # a field of the source's env block
uv run inference --vllm.model Qwen/Qwen3-4B                      # or a local vLLM server ...
uv run eval gsm8k -n 32 -r 4 -m Qwen/Qwen3-4B --client.base_url http://localhost:8000/v1   # ... still pinned; adapt via [concurrency]
uv run eval @ eval.toml --run.name my-eval                        # multi-source TOML
uv run eval @ eval.toml --run.name my-eval --resume               # resume the interrupted run
uv run eval @ eval.toml --dry-run                                 # resolve + write the config, exit
```

Shorthands (single-source runs): `<taskset-id>` names the run's only source, `--env.<field> <value>` sets a field of that source's env block (`--env.agent.harness.id bash`, `--env.taskset.tasks '["fix-git"]'`), `-n` `num_examples`, `-r` `group_size`, `-m` `model`, `-c N` pins the concurrency band (`concurrency.min_inflight = max_inflight = N`). The shorthands cannot be combined with a TOML that defines `[[source]]` blocks. `uv run eval -h` lists them.

Defaults: model `deepseek/deepseek-v4.1-flash` on Prime Inference (`PRIME_API_KEY`, else the `prime login` config). Concurrency is pinned at 128 (`-c N` repins). External APIs expose no vLLM `/metrics`, so the pin is what they run with; against `uv run inference` set `min_inflight < max_inflight` in `[concurrency]` to adapt to KV usage like the orchestrator's (the startup `/metrics` probe fails fast if the band is adaptive and the endpoint has no metrics).

Minimal multi-source `eval.toml` (`EvalConfig`, `packages/prime-rl-configs/src/prime_rl/configs/eval.py`; the eval block is flattened to the top level):

```toml
model = "Qwen/Qwen3-4B"
num_examples = 32   # always cap eval size for smokes
group_size = 4

[client]
base_url = "http://localhost:8000/v1"

[concurrency]       # adaptive against vLLM; the default pins 128
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

[monitors.prime]    # optional: upload each source's epoch as a platform evaluation
```

Per-source `num_examples`, `group_size` and `sampling` override the top-level defaults. Ready-made configs: `examples/basic/<env>/eval.toml` (the walkthroughs' baseline/final evals against a local server; `-m` swaps in a trained checkpoint) and `configs/debug/eval/{single-turn,multi-turn,resume,multi-env,aime2026,tb2}.toml`, all on Prime Inference (gsm8k; 16 terminal-bench-2 tasks in sandboxes; an interruptible gsm8k run; 16 terminal-bench-2 tasks under the bash and rlm harnesses side by side; AIME 2026 at avg@16; the full Terminal-Bench 2 at avg@4).

- Entrypoint: `src/prime_rl/entrypoints/eval.py` (shorthand expansion), implementation `src/prime_rl/eval/eval.py`, shared engine `src/prime_rl/eval/runner.py`.
- Env servers: spawned by the eval process unless the source sets `serve.address` (externally managed); each binds an OS-assigned loopback port and publishes it to `configs/attempt_N/resolved/envs/eval/<name>.address`, so concurrent runs on one host never collide.
- Platform: `--monitors.prime` (needs `PRIME_API_KEY` or `prime login`) creates one evaluation per source on app.primeintellect.ai once its epoch finishes and logs the URL.
- Resume: cursor checkpoints are on by default (`[ckpt]`: `interval` counts completed task groups, `keep_last` prunes older cursors; `--no-ckpt` disables). Relaunch with the same `--run.name` and `--resume` (or `--resume.step N`); partially completed groups are retried.

## Monitor an eval

Run dir: `output_dir / run.name` (auto `<envs>--<model>--<short-id>`; `ls -t outputs | head -1` finds the latest). `--clean` wipes a used run dir.

```
{run_dir}/
├── configs/latest/            # command.txt, the launch TOML, resolved/eval.json
├── logs/latest/
│   ├── eval.log               # the eval process (everything; the console only shows errors + the final success line)
│   └── envs/eval/{name}.log   # one log per env server
├── monitors/file/             # metrics.jsonl + the trace stream (dashboard reads these)
│   └── traces/live/{trace_id}.jsonl   # the env server's deltas of one in-flight trace; deleted when its episode lands in the stream
└── checkpoints/step_{cursor}/eval/progress.pt   # task cursor, newest kept
```

Check-in:

```bash
tail -F {run_dir}/logs/latest/eval.log
grep -E "WARNING|ERROR" {run_dir}/logs/latest/eval.log {run_dir}/logs/latest/envs/eval/*.log
grep SUCCESS {run_dir}/logs/latest/eval.log            # one "Evaluated <env> ... Reward 0.xxxx" line per source (file only; the console is quiet while it runs)
uv run python -m prime_rl.monitors.file.traces {run_dir}              # every live rollout: phase, turns, tokens, elapsed, last message
uv run python -m prime_rl.monitors.file.traces {run_dir} <trace_id>   # one in-flight trace, assembled from its deltas
```

Env servers stream each rollout turn by turn as deltas; the file monitor keeps one file per in-flight trace under `monitors/file/traces/live/` and drops it when the episode finishes, so `ls` there is the in-flight set. The progress line in `eval.log` counts the in-flight rollouts by phase (`- boot 1 · running 3`) and the dashboard's traces tab shows them in the episode table (tinted rows, phase badge, growing turn/token/branch counts, status filter `in flight` / `finished`) and opens them in the viewer mid-flight. A rollout stuck in `boot` for minutes is waiting on its sandbox; one stuck in `running` with a growing turn count is working, one with a frozen turn count is waiting on a model call or tool.

Metrics (`monitors/file/metrics.jsonl`, W&B, dashboard) live under `eval/<env>/all/<agent>/…`: `reward/mean`, `is_truncated/mean` (rollouts cut by the length limit — raise `sampling.max_completion_tokens`), `has_error/mean`, plus the taskset's own metrics; `eval/<env>/all/seq_len/mean` is the episode length. Validate a result by reading a few traces in the dashboard (`dashboard` skill) rather than trusting the mean alone.

Stop a run with SIGINT/SIGTERM to the eval PID (`ps aux | grep PRL::Eval`); the cursor checkpoint lets `--resume` pick it up. Env servers are children of the eval process and exit with it.
