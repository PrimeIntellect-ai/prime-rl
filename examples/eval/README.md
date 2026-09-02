# Eval examples

Each file is a complete `uv run eval @ examples/eval/<name>.toml` run against Prime Inference, the
default client (`PRIME_API_KEY` or `prime login`; pinned concurrency since external APIs expose no
vLLM metrics). Set `client.base_url` to a local `uv run inference` server and drop the
`[concurrency]` pin to let the band adapt. `uv run eval -h` lists the single-source shorthands (`<taskset-id>`, `--env.<field>`,
`-n`, `-r`, `-c`, `-m`).

| File | Shows |
|---|---|
| `gsm8k.toml` | Single-turn math with the bash harness, several rollouts per task |
| `wordle.toml` | The textarena Wordle game, the engine playing the user (`env.player` seat) |
| `wiki-search.toml` | A shared tool server (search) behind the agent, with a turn cap |
| `terminal-bench-2.toml` | Named tasks (`env.taskset.tasks`) in prime sandboxes, whole-rollout retries |
| `best-of-n.toml` | A reusable env paired with a taskset (`env.id`), pass@k over 8 attempts |
| `agentic-judge.toml` | A judge agent verifying each attempt in its own sandbox (`env.solver` / `env.judge` seats) |
| `rlm-docker.toml` | The rlm harness in a sized docker sandbox, explicit sampling, a taskset split |
| `swe.toml` | Two sources in one run: SWE-bench Verified + Terminal-Bench 2 |

The shorthand equivalents, for single-source runs:

```bash
uv run eval gsm8k -n 5 -r 3 -c 8 --env.agent.harness.id bash
uv run eval gsm8k --env.id best-of-n --env.n 8 ...
uv run eval terminal-bench-2 --env.taskset.tasks '["fix-git"]' --env.agent.harness.id bash ...
uv run eval gsm8k --env.agent.harness.id rlm --env.agent.runtime.type docker --env.agent.runtime.cpu 4 ...
uv run eval gsm8k --env.agent.harness.disabled_tools '["shell_tool"]' ...
```

Add `--monitors.prime` to upload each source's finished epoch as an evaluation on the Prime
Intellect platform, and `--run.name <name>` to get a run directory you can `--resume`.
