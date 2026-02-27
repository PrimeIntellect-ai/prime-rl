# Prime Platform Local Run

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

Add `[orchestrator.prime_monitor]` to your TOML config:

```toml
[orchestrator.prime_monitor]
run_name = "my-experiment"   # optional; defaults to W&B run name if set
```

Or as a CLI override:
```bash
uv run rl @ config.toml --orchestrator.prime_monitor.run_name "my-experiment"
```

If `RUN_ID` is not already set in the environment, prime-rl will automatically:
1. Resolve API key from `PRIME_API_KEY` env var or `~/.prime/config.json` (via `prime login`)
2. Call `POST /api/v1/rft/external-runs` → extract `run.id`, print dashboard URL
3. Set `RUN_ID` in the orchestrator process
4. Stream metrics/samples/distributions to `api/internal/rft` during training
5. On completion (or crash), call `PUT /api/v1/rft/external-runs/{run_id}/status`

If `RUN_ID` is already set (hosted K8s runs inject it directly), registration is skipped and monitoring proceeds normally.

## Optional fields

```toml
[orchestrator.prime_monitor]
run_name = "qwen3-reverse-text"
team_id = "clxxx..."          # show run under a team
base_url = "https://api.primeintellect.ai/api/v1/rft"  # default
```

## Auth

`PRIME_API_KEY` (from env or `~/.prime/config.json`) is used for both creating the run and streaming data.
The key must have `rft:write` scope — granted by default when you `prime login`.

`team_id` is resolved from `prime_monitor.team_id` in config, then from `~/.prime/config.json`.
