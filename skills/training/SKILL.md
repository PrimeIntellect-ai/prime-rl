---
name: training
description: Launch and monitor prime-rl training runs. Use when starting, supervising, or debugging an RL/SFT run. Routes Model Factory to model-factory, and local or SLURM runs to start-run and monitor-run.
---

# Training

Choose the deployment workflow first.

- **Model Factory** — see [model-factory](../model-factory/SKILL.md) for `prime train` launches, custom source revisions, credentials, private datasets, and remote monitoring. Requests such as "run this on model factory" use this path unless the user explicitly requests SLURM.
- **Start a local / SLURM run** — see `start-run/SKILL.md` for the `rl`, `sft`, and `inference` entrypoints and how to launch them (single-node, SLURM, dry-run).
- **Monitor a local / SLURM run** — see `monitor-run/SKILL.md` for the runbook: how to find the output dir, what to tail, which metrics to watch, and how to restart safely.

The local start/monitor subskills assume the `configs` skill for config-loading mechanics. Model Factory dispatch takes a TOML filename directly and has its own CLI flags. Standalone evals (`uv run eval`) are the `eval` skill; online evals follow the selected training workflow.
