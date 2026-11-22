# Long-horizon agent harness

`glyphbench.pro_harness` drives CraftaxFull and NetHack directly through
`BaseGlyphEnv`, using an OpenAI-compatible chat endpoint or Azure Responses.
The `normal` variant retains a linear conversation. The default `pro` variant
adds area-specific guidance, structured memory, and remembered landmarks.
This is a separate agent from the Verifiers evaluation and training harnesses.

## Quickstart

Install the full-game adapter and optional media/tracking tools in a checkout
with its submodules initialized:

```bash
uv sync --extra craftax --extra tracking

uv run --no-sync python -m glyphbench.pro_harness \
  --task glyphbench/craftaxfull-v0 \
  --backend openai --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3.5-4B --api-key-env OPENAI_API_KEY_LOCAL \
  --num-episodes 3 --seed 42
```

Set `OPENAI_API_KEY_LOCAL` to the endpoint's key, or a nonempty placeholder
for a local server that does not authenticate requests. For NetHack, install
`--extra nethack` in place of `--extra craftax`.

The wrapper prepares a separate environment with the selected game's
requirements:

```bash
TASK=glyphbench/craftaxfull-v0 MODEL=Qwen/Qwen3.5-4B \
  BASE_URL=http://localhost:8000/v1 bash eval/run_pro.sh

BACKEND=azure TASK=glyphbench/nethack-full-v0 \
  AZURE_OPENAI_MODEL=your-deployment REASONING_EFFORT=high \
  bash eval/run_pro.sh
```

Azure requires `AZURE_OPENAI_RESPONSES_ENDPOINT` and `AZURE_OPENAI_API_KEY` in
the shell or ignored `.env` file. The direct CLI uses `--azure-model` to select
a deployment and `--azure-endpoint` to override the endpoint variable.
Supported reasoning settings depend on that deployment.

## The interaction loop

In Pro mode, the action prompt contains recent actions, the current area's
guide, recalled landmarks, stored memory, and the current observation. The
model responds with a named action, normally `<action>NAME</action>`. The Pro
parser accepts several response forms and game-specific synonyms; its behavior
differs from the strict evaluation parser.

After the game steps, a memory update can see the action's reasoning, parsed
outcome, reward, and next observation. It writes a `<memory>...</memory>` block
containing changes to the scratchpad and spatial notes. The default
`--memory-update-every 1` updates every turn that needs another decision.
Larger values reduce periodic updates; rewards, achievements, area changes,
parsing failures, and truncations can still trigger an immediate refresh.

Persistent state includes:

- A bounded scratchpad with strategy, plans, lessons, tactical notes, and floor
  checklists. Omitting a section from an update retains its previous contents.
- Per-area landmark records, updated through memory directives such as
  `MAP_ADD`, `MAP_REMOVE`, `MAP_GET`, and `MAP_GET_ALL`.
- An observation-based navigation summary of visited tiles, recent routes,
  frontiers, and last-seen features. Disable it with `--no-exploration-aid`
  when studying its effect.

## Budgets and failures

The default `max_output_tokens=None` omits an explicit output cap in model
requests. Provider output and context limits still apply. Normal mode keeps
conversation history; Pro mode reconstructs prompts from its explicit state.
The harness does not impose an additional episode limit based on token count.
Use `--max-output-tokens` to set an output limit or `--max-turns` to shorten
the game's native horizon.

Transport failures and rejected responses have bounded retries, recorded in
run metrics. An action that remains unparseable after retries uses the
configured forfeit policy. Pro defaults to `--forfeit-mode noop`, which applies
the game's fallback action; `freeze` advances only the turn counter. The
evaluation loader's default forfeit mode is `freeze`, so account for this
setting when comparing harnesses.

Run `python -m glyphbench.pro_harness --help` for all settings, including
request timeouts, retry counts, episode concurrency, and sampling options.

## Outputs and scoring

Results are written under `outputs/pro_harness/` unless `--output-dir` selects
a different base directory. Each run directory includes the task, backend,
model, harness variant, and reasoning settings. Use `--run-tag` to distinguish
runs with otherwise identical settings.

| File | Contents |
|---|---|
| `metrics.json` | Per-episode metrics and aggregate summaries. |
| `interactions.json` | Action and memory calls, full outputs, and token usage. |
| `transcript_<ep>.jsonl` | Observations, actions, and outcomes consumed by replay. |
| `episode_<ep>.mp4` | Native Craftax frames, when video recording is available and enabled. |
| `config.json` | Run configuration with credential values removed. |
| `progress_<ep>.json` | Intermediate progress when checkpointing is enabled. |

The direct CLI saves at episode end by default. The shell wrapper sets
`CHECKPOINT_EVERY=50` to flush progress during long games; the corresponding
CLI option is `--checkpoint-every`. Video is enabled by default for Craftax,
and `--no-video` disables it. NetHack has no native video renderer here.

`episode_return` is the native cumulative game reward. A separate `raw_score`
reports Craftax achievement count or NetHack's in-game score. Craftax return
includes weighted achievement rewards and health changes, so it is different
from the number of achievements. The paper's Craftax percentage scoring uses
native cumulative reward, not that achievement count.

## Tracking

W&B logging is opt-in. Set a project explicitly through the CLI or wrapper,
and authenticate using `WANDB_API_KEY` or W&B's login command:

```bash
WANDB_PROJECT=glyphbench-evals \
  TASK=glyphbench/craftaxfull-v0 MODEL=Qwen/Qwen3.5-4B \
  BASE_URL=http://localhost:8000/v1 bash eval/run_pro.sh
```

The direct CLI accepts `--wandb-project`, `--wandb-entity`, `--wandb-group`,
`--wandb-name`, and `--wandb-run-id`. Its project default also follows an
explicit `WANDB_PROJECT` environment variable. Logging may upload full
interaction tables, replay artifacts, and videos along with metrics.

The separate Azure Verifiers wrapper uses `WANDB_LOG=1` to enable its upload;
see the [evaluation guide](../../../eval/README.md#azure-evaluation-and-tracking).

## Replay

```bash
uv run --no-sync glyphbench replay outputs/pro_harness --list
uv run --no-sync glyphbench replay outputs/pro_harness --pause
uv run --no-sync glyphbench replay outputs/pro_harness \
  --env glyphbench/nethack-full-v0 --episode 0
```

Replay displays the saved text observation, generated output, parsed action,
and feedback. See the [replay guide](../../../docs/REPLAY.md) for filters and
pager controls.

## Craftax observation experiments

`--observation-mode text`, the default, sends the GlyphBench grid and HUD.
CraftaxFull Pro also supports `native_text`, using the upstream coordinate-list
renderer, and `pixels`, sending native frames through Azure Responses:

```bash
uv run --no-sync python -m glyphbench.pro_harness \
  --task glyphbench/craftaxfull-v0 \
  --backend azure --azure-model your-deployment \
  --harness-variant pro --observation-mode pixels \
  --image-detail original --reasoning-effort high
```

Pixel mode requires the native renderer and a deployment that accepts images.
The saved text transcript is then a diagnostic view; the model receives PNG
frames on action and memory turns. Run configuration and interaction records
identify the observation mode. Native video recording is configured separately.

## Modules

[Configuration](config.py), [provider clients](clients.py),
[game adapters](adapters.py), [prompts](prompting.py), [action parsing](parser.py),
[memory](memory.py), [navigation](navigation.py), [rollout control](harness.py),
[media](media.py), [tracking](tracking.py), and [CLI](run.py).
