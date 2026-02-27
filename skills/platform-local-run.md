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

Enable platform streaming for a single run with a CLI override — no TOML editing needed:

```bash
uv run rl @ config.toml --prime_platform.base_url "https://api.primeintellect.ai"
```

For a persistent configuration, add `[prime_platform]` to your TOML config instead:

```toml
[prime_platform]
run_name = "my-experiment"        # optional; defaults to W&B run name if set
base_url = "https://api.primeintellect.ai"  # default, change for local dev
```

prime-rl will:
1. Resolve `PRIME_API_KEY` from env var or `~/.prime/config.json`
2. Call `POST /api/v1/rft/external-runs` → get `run` object, extract `run.id`, print dashboard URL
3. Set `RUN_ID` in the orchestrator process environment
4. Auto-configure `PrimeMonitor` to stream metrics/samples to the platform
5. On successful completion, call `PUT /api/v1/rft/external-runs/{run_id}/status`

## Optional platform fields

```toml
[prime_platform]
run_name = "qwen3-reverse-text"
team_id = "clxxx..."          # show run under a team
```

## Auth

`PRIME_API_KEY` is used for both creating the run and streaming monitoring data.
The key must have `rft:write` scope — this is granted by default when you `prime login`.

For local platform dev, create a token with `rft:write` scope in the local DB and set it as `PRIME_API_KEY`.

## Team ID resolution

`team_id` is resolved in this order:
1. `prime_platform.team_id` in config
2. `PRIME_TEAM_ID` environment variable
3. `team_id` in `~/.prime/config.json` (written by `prime login`)

## SLURM / multi-node

`[prime_platform]` in `rl.toml` works for all deployment modes. For SLURM multi-node, `write_subconfigs()` serializes the resolved `prime_platform` config into `orchestrator.toml` automatically — no manual changes needed.
