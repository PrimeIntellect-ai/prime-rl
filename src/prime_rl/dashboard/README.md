# Dashboard

`uv run dashboard [output_dir ...]` — a local web dashboard for run output
directories (metrics, resolved configs, rollout traces, cited reports,
component logs) at
http://localhost:7788. Needs the `dashboard` extra. Every instance also serves
the dirs registered by launchers in `~/.cache/prime-rl/dashboard/dirs.json`
(one dashboard per host per user); `--isolated` serves only the given dirs and
skips the registry. See `skills/dashboard/SKILL.md` for discovery,
kill/restart commands, and the local view-command/report contract.

GPU dependencies live behind the `gpu` extra, so this works anywhere — a
cluster head node, a laptop against a mounted outputs dir:

```bash
uv sync --extra dashboard && uv run dashboard [output_dir ...]
```

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.
The integration suite covers it end to end (`tests/integration/dashboard_smoke.py`
runs after every integration test).
