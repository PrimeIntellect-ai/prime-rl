# Dashboard

`uv run dashboard [output_dir ...]` — a local web dashboard for run output
directories (metrics, resolved configs, rollout traces, component logs) at
http://localhost:7788. Needs the `dashboard` extra.

To run it without resolving the project's GPU dependencies (a cluster head
node, a laptop against a mounted outputs dir), use the standalone script form —
it resolves only a handful of small wheels into an isolated script environment:

```bash
uv run --script src/prime_rl/dashboard/server.py [output_dir ...]
```

The script needs its `static/` sibling directory, so copy this whole folder
when moving it to another machine.

This package is fully AI-generated and maintained by agents - it is not meant to be read or edited by humans. Change it by asking an agent, and verify through the browser smoke tests.
The integration suite covers it end to end (`tests/integration/dashboard_smoke.py`
runs after every integration test).
