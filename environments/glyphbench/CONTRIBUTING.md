# Contributing to GlyphBench

Contributions can add environments, improve agent integrations, or make
experiments easier to run and inspect. For setup and common workflows, start
with the [development guide](docs/GETTING_STARTED.md).

## Adding an environment

1. Add an implementation under `src/glyphbench/envs/<suite>/`, extending the
   suite's base class or `BaseGlyphEnv`.
2. Define `action_spec` and implement `env_id()`, `_reset(seed)`,
   `_step(action_index)`, `_render_current_observation()`, and `system_prompt()`.
3. Register the class in the suite's `__init__.py` with `register_env(...)`.
4. Add focused tests under `tests/envs/<suite>/` and run the shared conformance
   tests. Reuse existing coverage where it already exercises the behavior.
5. Refresh the catalog and the affected suite's random baseline:

```bash
uv run python scripts/generate_env_catalog.py
uv run python eval/random_baseline.py --episodes 25 --include-suite <suite>
```

The game prompt should explain the objective, rules, rewards, and termination
conditions. The harness appends the action menu separately; avoid repeating
it in `system_prompt()`.

## Environment contract

- Make success measurable, with enough variation to study learning and agent
  behavior. A task should be understandable from its rules and observations.
- Use one Unicode codepoint per grid cell and a clear legend. Keep spatial
  information in the grid and complementary state in the HUD. See the
  [observation format](docs/OBSERVATION_FORMAT.md).
- Reproduce the same trajectory from the same seed and action sequence.
- Keep standard task horizons below 512 turns and cumulative returns in
  `[-1, 1]`. Extended Atari, CraftaxFull, and NetHack tasks are evaluated
  separately; the full-game adapters preserve native rewards.
- Distinguish a valid but ineffective game action from malformed model output.
  Game actions follow the simulator's rules. Invalid integer indices raise an
  error; the harness handles unparseable replies through its forfeit policy.
- Put diagnostic information in `info`, rather than exposing privileged state
  in the model's observation.

Tests should cover seeded dynamics, the observation and action contracts,
termination, and task-specific edge cases. Shared rollout tests check standard
task horizons and return bounds. Add a regression test when a bug can recur;
avoid duplicating the same assertion across several test layers.

## Evaluation and provider integrations

The standard evaluation path uses Verifiers and an OpenAI-compatible endpoint.
Reusable public endpoint definitions belong in `configs/endpoints.toml`, with
credentials supplied through environment variables. Provider clients for the
long-horizon Pro harness live in `src/glyphbench/pro_harness/clients.py`.
Document any new setup in the [evaluation guide](eval/README.md).

## Code style and checks

Use Python 3.12, type annotations, and Ruff's 100-character line-length setting.
Mypy checks the core, protocol, and termination modules in strict mode, as
configured in `pyproject.toml`.

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python scripts/check_release_hygiene.py
uv build
```

Keep pull requests focused and describe the behavior that changes, why it
changes, and how it was checked. Do not commit populated environment files,
provider keys, local machine paths, generated run artifacts, or notebook outputs.

For a bug report, include the task ID, seed, action sequence or reproduction
command, and relevant dependency versions. Remove credentials and private
content from any attached logs or trajectories.
