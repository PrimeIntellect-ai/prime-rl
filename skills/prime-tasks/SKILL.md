---
name: prime-tasks
description: Where task data and taskset code live across prime-tasks and prime-envs, and how to fix a broken task. Use when a Harbor task has a bad image, test, instruction, or asset, or before editing a taskset in prime-envs to work around one task.
---

# prime-tasks vs. prime-envs

| Repo | Holds |
| --- | --- |
| [`prime-tasks`](https://github.com/PrimeIntellect-ai/prime-tasks) | Raw Harbor task data: `task.toml` (with the image ref), `instruction.md`, `environment/` (Dockerfile, assets), `tests/`, `solution/`. |
| [`prime-envs`](https://github.com/PrimeIntellect-ai/prime-envs) | Runtime logic: taskset classes, configs, prompts, judges, `registry.json`, and the pinned `prime-tasks` commit. |

## Rules

- Do not put task data in `prime-envs`. No task dirs, Dockerfiles, test files, or data files for single tasks.
- Do not special-case single tasks in taskset code. No `if task.name == ...` that swaps an image, test, instruction, or answer.
- Pin `prime-tasks` by commit SHA, never by `main`.

## Fix a broken task

Open two companion PRs and link them to each other:

1. **`prime-tasks`**: fix the task dir. For an image fix, change the Dockerfile under `environment/`, build and push the image, and set the new ref in `task.toml`. For an upstream dataset, export the full task dir (see `datasets/terminal-bench-2/`).
2. **`prime-envs`**: after the `prime-tasks` PR merges, point the taskset at its merge commit. Update `git_commit_id` in `registry.json`, or the pinned repo URL in the taskset. Bump the env version and add a changelog line to its README.

Taskset code may select which tasks come from `prime-tasks` (e.g. `REPAIRED_TASKS` in `pi_terminal_bench_2`). The task data itself stays in `prime-tasks`.
