# Platform Monitoring

Use `orchestrator.prime_monitor` to register a run on the Prime Intellect platform and stream training metrics, samples, and distributions.

> **Internal-only for now:** external run registration is currently only enabled for internal / allowlisted teams.

## Prerequisites

You need a Prime API key with `rft:write` scope.

Use the CLI:

```bash
prime login
```

Or set an environment variable directly:

```bash
export PRIME_API_KEY=pit_...
```

## Minimal config

```toml
[orchestrator.prime_monitor]
run_name = "my-experiment"
```

You can also override from the CLI:

```bash
uv run rl @ config.toml --orchestrator.prime_monitor.run_name "my-experiment"
```

## Troubleshooting

### `API key not found`

Set the env var from `api_key_var` or run:

```bash
prime login
```

### `External training runs are not enabled for this team`

Your team is not allowlisted yet. This feature is currently internal-only.
