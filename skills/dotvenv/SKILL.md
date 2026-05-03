---
name: dotvenv
description: Rules for the local `.venv/` directory. Use when tempted to edit, patch, or "fix" installed package code, or when reading files under `.venv/` to understand third-party behavior.
---

# .venv

`.venv/` is the local virtual environment created by `uv`. It contains installed third-party packages, not project source. Treat it as read-only.

## Never edit anything under `.venv/`

Do not modify files in `.venv/` for any reason — not to patch a bug in a dependency, not to add a print statement for debugging, not to "try a fix" before pushing it upstream. Edits to `.venv/`:

- are silently overwritten by the next `uv sync` / `uv lock` / dependency change
- do not exist for anyone else (teammates, CI, training jobs on other nodes)
- create the illusion of a working fix that disappears as soon as the env is rebuilt

If a dependency is broken, fix it the right way:

- **Wrong version pinned** → update `pyproject.toml` and run `uv sync --all-extras`.
- **Upstream bug** → pin a fork or a specific commit in `pyproject.toml` (use a 7-char commit hash for git deps, per `AGENTS.md`).
- **Need to inspect behavior** → read the file, do not edit it. Add prints/logging in *our* code that calls into the dependency, not inside the dependency.
- **Need a local patch** → vendor the relevant code into `src/` under our own module, or open an issue/PR upstream. Do not hand-edit installed files.

## Reading is fine

Reading files under `.venv/` to understand how a library works is encouraged. Just do not write to them.

## If you find yourself editing `.venv/`

Stop, revert the change, and pick one of the options above. If you have already edited files there, run `uv sync --all-extras` to restore the env to a clean state before continuing.
