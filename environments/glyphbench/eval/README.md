# Model evaluation

GlyphBench evaluates agents through Verifiers and OpenAI-compatible model
endpoints. The default sweep covers the 303 standard tasks. Extended Atari,
CraftaxFull, and NetHack tasks are evaluated separately.

## Setup and quickstart

For an existing local or remote model server, install the evaluation client:

```bash
uv sync --extra eval-client
source .venv/bin/activate

# One task, two seeds.
bash eval/run_debug.sh

# All standard tasks, three seeds per task.
bash eval/run_full.sh
```

The default endpoint is `http://localhost:8000/v1`. To host a model locally,
install the GPU serving extra and start vLLM in another terminal:

```bash
uv sync --extra eval
uv run --no-sync vllm serve Qwen/Qwen3.5-4B \
  --port 8000 --max-model-len 65536 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --compilation-config '{"custom_ops":["-silu_and_mul"]}'
```

These serving settings include a mitigation for the pinned vLLM runtime and
dense Qwen3.5-4B model. Review the [dependency notes](../docs/DEPENDENCIES.md)
before changing the model or overriding the batch and compilation settings.

For a remote endpoint, set `VLLM_BASE_URL`, `MODEL`, and `API_KEY_VAR`. The last
is the name of an environment variable holding the key, rather than the key
itself. Local vLLM servers can use a nonempty placeholder.

```bash
VLLM_BASE_URL=http://model-server:8000/v1 bash eval/run_full.sh
```

The debug, full, and archival wrappers save results locally and skip Prime
uploads by default. `SKIP_UPLOAD=0` enables Prime uploads using your configured
account. `SAVE_RESULTS=0` disables local result saving; `EVAL_OUTPUT_DIR`
selects an output base directory. Inspect the path printed by Prime, then use
`glyphbench replay path/to/results --pause` or `prime eval tui`.

## Evaluation settings

[`glyphbench.eval_defaults`](../src/glyphbench/eval_defaults.py) defines the
shared protocol. Inspect the current settings with:

```bash
uv run python src/glyphbench/eval_defaults.py json
uv run python src/glyphbench/eval_defaults.py sampling-args
```

The standard settings use 4,096 output tokens per action, a 65,536-token model
context, and the game's native turn horizon. The memory-update output budget
is also 4,096 when a task uses memory. Default sampling uses temperature 1,
`top_p=1`, `top_k=0`, `min_p=0`, repetition penalty 1, zero presence/frequency
penalties, and Qwen thinking enabled for action generations.

### Shared wrapper variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | Model ID served by the endpoint. |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL. |
| `API_KEY_VAR` | `OPENAI_API_KEY_LOCAL` | Name of the variable containing the API key. |
| `SEED` | `42` | First episode seed; three episodes use seeds 42–44. |
| `N_FRAMES` | `0` | Number of previous observations, actions, and rewards included in the next prompt. |
| `ROLLOUTS_PER_EXAMPLE` | `1` | Repeated model rollouts for each task/seed row. |
| `MAX_TOKENS` | `4096` | Action output limit, also advertised as `max_output_tokens` in the prompt. |
| `MEMORY_UPDATE_MAX_TOKENS` | `4096` | Memory-update output limit. |
| `TEMPERATURE` | `1.0` | Model sampling temperature. |
| `SAMPLING_ARGS` | From `eval_defaults` | Additional sampling settings as JSON. |
| `SKIP_UPLOAD` | `1` | Keep results local instead of uploading to Prime. |
| `SAVE_RESULTS` | `1` | Save the evaluation results for replay. |
| `EVAL_OUTPUT_DIR` | Prime default | Optional output base directory. |

### Debug wrapper

| Variable | Default | Meaning |
|---|---|---|
| `NUM_EPISODES` | `2` | Episode seeds per task. |
| `TASK_IDS` | `["glyphbench/minigrid-empty-5x5-v0"]` | JSON array of task IDs. |
| `MAX_TURNS` | Unset | Optional shortened game horizon for a smoke test. |
| `AUTO_START_VLLM` | `0` | Start a local vLLM server if none is reachable. |
| `VLLM_STARTUP_TIMEOUT` | `1200` | Seconds to wait for an automatically started server. |
| `VLLM_LOG` | `/tmp/glyphbench-vllm.log` | Log for the automatically started server. |
| `VLLM_MAX_MODEL_LEN` | `65536` | Context capacity of the automatically started server. |

```bash
MODEL=Qwen/Qwen3-1.7B NUM_EPISODES=4 \
  TASK_IDS='["glyphbench/miniatari-pong-v0"]' bash eval/run_debug.sh

AUTO_START_VLLM=1 bash eval/run_debug.sh
```

### Full and archival wrappers

`EPISODES` defaults to `3` seeds per task. `run_full.sh` excludes
`atari`, `craftaxfull`, `nethack`, and `agentick`; these exclusions select the
six standard suites. Set `EXCLUDE_SUITES='[]'` to include every registered task.
`run_archival.sh` selects only Atari, CraftaxFull, and NetHack.

The standard full sweep contains **909 task/seed examples** (303 tasks × three
seeds), before any `ROLLOUTS_PER_EXAMPLE` repeats. The wrappers pass `-n -1`
to Prime to evaluate every row produced by the loader. Prime's `-n` counts
total examples; it is not the number of seeds per task.

```bash
EPISODES=10 bash eval/run_full.sh
bash eval/run_archival.sh
```

Full-game tasks retain long horizons and their own rewards. Choose inference
budgets appropriate to those tasks; a standard-task context limit may end a
long game early. The standalone [Pro harness](../src/glyphbench/pro_harness/README.md)
is another option for CraftaxFull and NetHack.

## Optional game adapters

The standard suites require no optional upstream game packages. From a checkout
with submodules initialized, install the adapters you need:

```bash
uv sync --extra eval-client --extra craftax --extra nethack
```

`eval/run_craftax.sh` prepares its own environment and runs the debug wrapper
on `glyphbench/craftaxfull-v0` by default. The experimental AgenticK adapter
uses the checked-out package in `third_party/agentick`; install it into your
active environment with `uv pip install -e third_party/agentick` and keep it
when running commands by using the active Python or `uv run --no-sync`.

## Azure evaluation and tracking

`run_azure.sh` connects Verifiers to an Azure Responses deployment through a
local compatibility proxy. Set `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_RESPONSES_ENDPOINT` in your shell or ignored `.env` file, and
choose your deployment with `AZURE_OPENAI_MODEL`:

```bash
AZURE_OPENAI_MODEL=your-deployment \
  TASK_IDS='["glyphbench/minigrid-empty-5x5-v0"]' \
  NUM_EPISODES=2 bash eval/run_azure.sh
```

The script prepares the evaluation client and tracking dependencies in a
separate environment, adding CraftaxFull or NetHack dependencies only when
selected. It saves results under `outputs/eval/` by default and skips Prime
uploads. W&B logging is disabled unless you set `WANDB_LOG=1`:

```bash
WANDB_LOG=1 WANDB_PROJECT=glyphbench-evals \
  AZURE_OPENAI_MODEL=your-deployment bash eval/run_azure.sh
```

Authenticate W&B separately using `WANDB_API_KEY` or its login command. For
the standalone Pro wrapper, tracking is enabled by an explicit
`WANDB_PROJECT`; see its [tracking instructions](../src/glyphbench/pro_harness/README.md#tracking).

## Python API and task selection

```python
import glyphbench

env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=3,
    n_frames=0,
    max_output_tokens=4096,
)
```

`use_memory=None`, the default, follows each task's registered setting.
`True` enables separate action and memory generations; `False` disables the
memory scaffold. Memory updates see the action outcome and next observation,
and the final action need not have a memory update. See the
[observation guide](../docs/OBSERVATION_FORMAT.md#memory-mode).

Match the environment's `max_output_tokens` to the evaluation client's
sampling limit. `max_turns=None` uses the native game horizon. The
[integration guide](../docs/INTEGRATION.md) documents the current loader options;
Prime-RL training uses the separate native Verifiers v1 API.

When `task_id` is omitted, use `include_suites`, `exclude_suites`,
`include_tasks`, and `exclude_tasks`. Task filters accept exact IDs or fnmatch
patterns. Include filters combine with OR, and exclusions always win. An
explicit `task_id` cannot be combined with filters. An unfiltered Python loader
includes extended and experimental tasks; the shell wrapper supplies the
standard benchmark exclusions. Native training recipes select explicit task
manifests rather than these evaluation filters.

## Random-agent reference and scoring

[`random_baseline.json`](random_baseline.json) contains five uniform-random
episodes per standard task, using each game's native horizon. Each entry
includes mean, standard deviation, minimum, maximum, median, episode length,
and individual returns. These are empirical reference values from a small
sample, rather than exact expected returns.

Regenerate the reference or an individual suite with:

```bash
uv run python eval/random_baseline.py --episodes 25
uv run python eval/random_baseline.py --episodes 25 --include-suite minigrid
```

The script also accepts task patterns, exclusions, `--include-all`, and an
output path. `--max-turns` changes the task budget, so leave it unset for a
reference at the native horizon. Run `--help` for all options.

Verifiers records the sum of game rewards as `episodic_return`. Standard
returns are bounded to `[-1, 1]`. The paper averages episodes within each task,
then weights task means equally; it also reports suite and task breakdowns.
The random baseline is available for interpreting results, but the evaluator
does not automatically subtract or normalize by it. Full Craftax and NetHack
preserve native rewards and should be reported separately.

The default scripts use three episode seeds per task for a manageable sweep.
The paper's model comparison uses 25 matched seeds per task, so reproducing
that comparison requires the paper's episode count as well as its model and
sampling settings. The [return-distribution notebook](../notebooks/return_violins.ipynb)
provides one way to inspect saved results.

## Diagnostics and related tools

The [failure-mode guide](../docs/llm-agent-failure-modes.md) distinguishes
forfeits, successful parse retries, output truncation, game timeouts, and
context-capacity stops. A truncated response can still contain a complete
action; a missing memory tag alone does not imply a recorded parse failure.
Use [trajectory replay](../docs/REPLAY.md) to inspect the underlying decisions.

The pinned [Reasoning Gym evaluator](reasoning_gym/README.md) handles the paper's
external reasoning panel, resumable runs, and paired comparisons.
