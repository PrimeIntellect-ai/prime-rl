---
name: dashboard
description: Find, start, use, and stop the local run dashboard — the web UI for run output dirs (metrics, configs, traces, logs). Use when asked for the dashboard URL, when a run needs watching in the browser, or when a dashboard must be restarted or killed.
---

# Run dashboard

`uv run dashboard [output_dir ...]` (default `outputs/`; needs the `dashboard`
extra) serves a web UI at `http://localhost:7788`. It only reads run dirs —
safe against live runs.

## One dashboard per host per user

Every dashboard instance serves the dirs it was started with **plus** every dir
in the per-user registry (`~/.cache/prime-rl/dashboard/dirs.json`, re-read
live). Launchers (`rl`, `sft`) register their output dir on every
start and, in interactive sessions, auto-start a dashboard only when none is
live — an already-running one absorbs the new dir automatically, whatever port
it is on. `--no-dashboard` opts a run out; non-interactive launches (CI, nohup)
register their dir but never spawn.

## Isolated mode

`--isolated` serves only the given dirs: no registry read or write, no
discovery claim, and launchers ignore the instance. Use it for focused views
(demos, debugging one run dir) or to keep a scratch dir out of the registry.

## Finding the live dashboard

Ports spill over (a taken 7788 bumps to 7789, ...), so **never probe port
7788**. Discovery:

```bash
cat ~/.cache/prime-rl/dashboard/daemon.json   # {"pid": ..., "url": "http://localhost:<actual port>"}
curl -sf $(jq -r .url ~/.cache/prime-rl/dashboard/daemon.json)/api/runs > /dev/null && echo live
ps aux | grep PRL::Dashboard             # process title
```

Hand the researcher the `url` from `daemon.json`. Launcher logs also print it:
startup ends with a `Dashboard · <url>` banner. The auto-started instance logs
to `~/.cache/prime-rl/dashboard/daemon.log`.

## Stopping / restarting

```bash
kill $(jq -r .pid ~/.cache/prime-rl/dashboard/daemon.json)   # the discovered instance
pkill -f PRL::Dashboard                                 # every dashboard on the host
```

A clean exit releases `daemon.json`; a stale file from a dead process is taken
over by the next start. Killing a dashboard never affects runs (it only reads),
and killing a run never takes the dashboard down (it runs in its own session).
Restart by launching any run, or directly: `uv run dashboard`.

## Dependency-free dashboard

GPU dependencies live behind the `gpu` extra, so the dashboard installs
anywhere (cluster head node, laptop against a mounted outputs dir) without
resolving the training stack:

```bash
uv sync --extra dashboard && uv run dashboard [output_dir ...]
```
