---
name: training
description: Launch and monitor prime-rl training runs. Use when starting, supervising, or debugging an RL/SFT run. Routes to `start-run` (entrypoints + how to launch) and `monitor-run` (logs, metrics, check-ins).
---

# Training

Two phases — start the run, then watch it.

- **Start a run** — see `start-run/SKILL.md` for the `rl`, `sft`, and `inference` entrypoints and how to launch them (single-node, SLURM, dry-run).
- **Monitor a run** — see `monitor-run/SKILL.md` for the runbook: how to find the output dir, what to tail, which metrics to watch, and how to restart safely.

Both subskills assume the `configs` skill for config-loading mechanics.
