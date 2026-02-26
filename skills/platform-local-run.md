# Platform Local Run

Stream live training metrics/samples to the Prime Intellect platform dashboard from any machine, without hosted infrastructure.

## Prerequisites

Your `PRIME_API_KEY` must have `rft:write` scope. Log in via the `prime` CLI:
```bash
prime login
```

Or set the environment variable directly:
```bash
export PRIME_API_KEY=pit_...
```

## Usage

Add a `[platform]` section to your TOML config:

```toml
[platform]
run_name = "my-experiment"        # optional
base_url = "https://api.primeintellect.ai"  # default, change for local dev
```

Then run as normal:
```bash
uv run rl @ config.toml
```

prime-rl will:
1. Resolve `PRIME_API_KEY` from env var or `~/.prime/config.json`
2. Call `POST /api/v1/rft/external-runs` → get `run_id`
3. Set `RUN_ID` for all subprocesses (`PRIME_API_KEY` is passed through unchanged)
4. Auto-configure `PrimeMonitor` to stream metrics/samples to the platform
5. On exit, call `PUT /api/v1/rft/external-runs/{run_id}/status`

## Minimal example config

```toml
max_steps = 100

[model]
name = "Qwen/Qwen3-4B"

[platform]
run_name = "my-experiment"

[orchestrator]
rollouts_per_example = 8

[[orchestrator.env]]
id = "reverse-text"

[trainer.optim]
lr = 1e-5

[ckpt]

[inference]
```

## Optional platform fields

```toml
[platform]
run_name = "qwen3-reverse-text"
wandb_project = "prime-rl"
wandb_entity = "my-team"
team_id = "clxxx..."          # show run under a team
```

## Auth

`PRIME_API_KEY` is used for both creating the run and streaming monitoring data.
The key must have `rft:write` scope — this is granted by default when you `prime login`.

For local platform dev, create a token with `rft:write` scope in the local DB and set it as `PRIME_API_KEY`.
