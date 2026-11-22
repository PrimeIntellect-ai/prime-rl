# scripts/

Utilities for playing games, inspecting trajectories, and producing assets.
Run a script with `--help` for its full options.

## Demo + replay

- **`demo_all_envs.py`** — Watch a uniform-random agent play any env in the
  same rich TUI layout `gb replay` uses (header bar, system-prompt panel,
  grid + side panels). Useful for smoke-testing a freshly installed env or
  visually scanning an entire suite.
  ```bash
  uv run python scripts/demo_all_envs.py --suite minigrid --delay 0.1
  uv run python scripts/demo_all_envs.py --env glyphbench/craftaxfull-v0 --pause
  uv run python scripts/demo_all_envs.py --list
  ```
  Pause-mode hotkeys: `→` advance, `←` rewind, `s` system prompt, `l` legend,
  `a` action list, `q`/`n` next env.

- **`replay_trajectory.py`** — Replay a single saved trajectory `.jsonl` with
  canonical observation glyphs plus terminal color, or export it as a GIF.
  ```bash
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl
  uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl --delay 0.2
  uv run --extra assets python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif out.gif
  uv run python scripts/replay_trajectory.py path/to/trajectories/   # replay all in dir
  ```
  Note: this is the legacy single-file viewer. For browsing whole runs
  directories with multi-panel TUI and pause-mode hotkeys, use `gb replay`
  (see `docs/REPLAY.md`).

## Asset generation

- **`record_random_gifs.py`** — Render a random-agent GIF for every env (or a
  subset) via the `replay_trajectory.export_gif` renderer.
  ```bash
  uv run --extra assets python scripts/record_random_gifs.py --output ./out/gifs/
  uv run --extra assets python scripts/record_random_gifs.py --suite minigrid --steps 30
  uv run --extra assets python scripts/record_random_gifs.py --env glyphbench/atari-pong-v0
  ```

- **`upload_assets.py`** — Upload a directory of files (e.g. the GIFs produced
  by `record_random_gifs.py`) to a Hugging Face dataset repo. Reads `HF_TOKEN`
  from the shell or from local `.env`; the destination is always explicit.
  ```bash
  uv run --extra assets python scripts/upload_assets.py --repo owner/name
  GLYPHBENCH_ASSET_REPO=owner/name uv run --extra assets python scripts/upload_assets.py --replace
  uv run --extra assets python scripts/upload_assets.py --repo owner/name --src gifs --dst gifs --private
  ```

## Catalog

- **`generate_env_catalog.py`** — Refresh `docs/ENVIRONMENTS.md` from the
  current env registry.
  ```bash
  uv run python scripts/generate_env_catalog.py
  ```

## Manual play / debugging

- **`play_random.py`** — Play any single env with random actions, printing
  each step to stdout. The simplest possible sanity check for a new env.
  ```bash
  uv run python scripts/play_random.py glyphbench/craftaxfull-v0
  uv run python scripts/play_random.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42 --steps 50
  uv run python scripts/play_random.py glyphbench/minihack-eat-v0 --delay 0.3
  ```

- **`play_interactive.py`** — Interactive curses player: shows numbered
  actions; press the corresponding number key to act. Good for manually
  exploring episode dynamics.
  ```bash
  uv run python scripts/play_interactive.py glyphbench/craftaxfull-v0
  uv run python scripts/play_interactive.py glyphbench/minigrid-doorkey-5x5-v0 --seed 42
  ```

## Shared rendering support

- **`terminal_colors.py`** — Curses color mapping library (not a standalone
  script). Provides `char_attr(ch)` and `init_colors()` used by
  `play_interactive.py`. Import directly; has no `__main__` entry point.
