# Getting started

This guide covers a development checkout, model evaluation, training, and
trajectory replay. For a first environment interaction, use the
[README quickstart](../README.md#quickstart).

## Set up a development checkout

Use Python 3.12 and uv:

```bash
git clone --recurse-submodules https://github.com/VmaxAI/glyphbench.git
cd glyphbench
uv sync --frozen --extra dev
uv run glyphbench list-suites
uv run glyphbench list-envs --suite minigrid
uv run python examples/quickstart.py
```

The standard benchmark needs no optional game packages. The CraftaxFull,
NetHack, and AgenticK adapters use the pinned submodules in `third_party/`;
install their dependencies only when using those adapters. CraftaxFull and
NetHack have `craftax` and `nethack` extras. AgenticK setup is documented in
the [evaluation guide](../eval/README.md).

## Evaluate a model

Install the evaluation client, then point it at an existing OpenAI-compatible
server. The scripts default to `http://localhost:8000/v1`; the
[evaluation guide](../eval/README.md) describes endpoint configuration.

```bash
uv sync --extra eval-client
bash eval/run_debug.sh
bash eval/run_full.sh
```

The debug run checks one environment. The full run selects the six standard
suites with the settings in `src/glyphbench/eval_defaults.py`. The evaluation
guide covers endpoint variables, model serving, suite filters, archival
runs, scoring, and output locations. To run a local vLLM server, install the
GPU serving dependencies with `uv sync --extra eval`.

## Train

The training extra installs the pinned Prime-RL GPU stack. Validate a config
before starting workers:

```bash
uv sync --extra rl
uv run rl @ configs/rl/qwen35-4b-glyphbench/single_task.toml --dry-run true
uv run rl @ configs/rl/qwen35-4b-glyphbench/multitask.toml --dry-run true
```

The [training guide](../configs/rl/qwen35-4b-glyphbench/README.md) covers single-task
and multitask runs, the paper configuration, and local smoke runs. Use
`scripts/rl/render_prime_rl_config.py` to select tasks, resources, and logging;
keep generated configs and artifacts under the ignored `runs/` directory.

## Run the held-out reasoning panel

Reasoning Gym has a pinned dataset and evaluator environment. Prepare and
verify the panel before evaluating a model:

```bash
uv run python eval/reasoning_gym/reasoning_gym_eval.py prepare
uv run python eval/reasoning_gym/reasoning_gym_eval.py verify
```

Follow the [Reasoning Gym guide](../eval/reasoning_gym/README.md) for serving,
evaluation, resume, and paired comparisons. W&B publishing is opt-in via
`--publish-wandb`.

## Replay trajectories

```bash
uv run glyphbench replay path/to/run --list
uv run glyphbench replay path/to/run --pause
```

The replay command accepts a directory containing a single Verifiers
evaluation run or a tree of `results.jsonl` files. See the
[replay guide](REPLAY.md) for filters and keyboard controls.

## Local settings

Use environment variables for provider credentials and optional W&B logging.
If a workflow supports `.env` files, copy the tracked `.env.example` to the
ignored `.env` and fill in only the settings you need. Keep populated
credentials and run artifacts out of commits.

## Check your changes

Run the same checks as CI before opening a pull request:

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python scripts/check_release_hygiene.py
uv build
```
