# Use GlyphBench with your own agent

## Direct game loop

An environment is a plain Python object. Pass an integer action index to
`step`; use `action_spec.index_of(name)` if your agent chooses names.

```python
from glyphbench.core import make_env

env = make_env("glyphbench/minigrid-doorkey-6x6-v0")
try:
    observation, info = env.reset(seed=42)
    total = 0.0
    while True:
        # Supply your own agent; it returns one of the provided action names.
        action_name = your_agent(observation, env.action_spec.names)
        action = env.action_spec.index_of(action_name)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    print(f"Episode return: {total}")
finally:
    env.close()
```

The action vocabulary is fixed for an episode. The agent must decide which
of those actions makes sense in the current state; the menu does not hide
unusable actions. Invalid indices raise an error. A valid action that cannot
be performed follows the game's rules, such as attempting to move into a wall.

| Attribute or method | Description |
|---|---|
| `env.reset(seed)` | Start an episode; return `(observation, info)`. |
| `env.step(action_index)` | Advance one turn; return `(observation, reward, terminated, truncated, info)`. |
| `env.action_spec.names` | A `tuple[str, ...]` of canonical action names. |
| `env.action_spec.n` | Number of discrete actions. |
| `env.action_spec.index_of(name)` | Resolve a name or supported alias to an integer index. |
| `env.action_spec.render_for_prompt()` | Render the action menu and descriptions. |
| `env.system_prompt()` | Game rules, objectives, rewards, and termination conditions. |
| `env.close()` | Release resources. |

The observations contain a `[Grid]` and optional `[Legend]`, `[HUD]`, and
`[Message]` sections. The action menu is separate. For a custom model prompt,
combine `env.system_prompt()` with `env.action_spec.render_for_prompt()` and
your own output-format instructions. The bundled harness asks for a complete
`<action>NAME</action>` tag, then resolves the name through `ActionSpec`.
See the [observation guide](OBSERVATION_FORMAT.md).

`ActionSpec.index_of` accepts case-insensitive names, environment-specific
aliases, and unambiguous no-op aliases (`NOOP`, `WAIT`, `DONE`, `PASS`, `SKIP`).
It raises `KeyError` for an unknown or ambiguous name. These conversions happen
before `env.step`, which accepts integer indices only.

## Verifiers v1 for training

The installed package provides a Verifiers v1 taskset, environment, and harness:

```python
from verifiers.v1.utils.loaders import environment_class, harness_class, taskset_class

Taskset = taskset_class("glyphbench")
Environment = environment_class("glyphbench")
Harness = harness_class("glyphbench")
```

`GlyphBenchTasksetConfig` accepts a `tasks` list or a newline-delimited
`task_file`. It streams tasks in balanced round-robin order with configured
seed sampling. The environment opens one interaction for the episode and
supplies a new observation after each action. The in-process harness sends
model requests through Verifiers' authenticated interception endpoint,
preserving its traces and supporting Prime-RL's prefix reuse.

The v1 integration requires `use_memory=false`; training uses conversation
history instead of separate memory-update calls. Games end at their native
horizons. Set the per-turn output budget in both the taskset and the client's
sampling configuration; the model context limit is a separate trajectory
budget. See the [Prime-RL recipes](../configs/rl/qwen35-4b-glyphbench/README.md).

## Verifiers evaluation loader

The evaluation loader preserves the `prime eval run` interface and its
`results.jsonl` format:

```python
import glyphbench

env = glyphbench.load_environment(
    task_id="glyphbench/minigrid-empty-5x5-v0",
    num_episodes=3,
    max_output_tokens=4096,
)
```

The main options are:

| Option | Default | Meaning |
|---|---|---|
| `task_id` | `None` | A task ID or list of IDs; otherwise select the registry using filters. |
| `num_episodes` | `3` | Episodes per task. |
| `seed` | `42` | First episode seed; later episodes increment it. |
| `n_frames` | `0` | Number of previous observations, actions, and rewards retained in the next prompt. |
| `max_turns` | `None` | Override the game's native turn cap. |
| `max_output_tokens` | `4096` | Action budget advertised to the model; match it in client sampling settings. |
| `use_memory` | `None` | Use each task's registered default; `True` or `False` overrides all selected tasks. |
| `memory_update_max_tokens` | `4096` | Output budget for a memory update; `None` reuses the action sampling limit. |
| `parse_retries` | `1` | Same-turn retries after an invalid action reply. |
| `forfeit_mode` | `"freeze"` | After failed parsing, advance only the clock; `"noop"` instead applies a fallback action. |
| `stop_after_xml` | `True` | Stop generation after the closing protocol tag while preserving it for parsing. |

When `task_id` is omitted, `include_suites` and `include_tasks` select tasks by
suite or exact ID/fnmatch pattern. The include filters combine with **OR**;
`exclude_suites` and `exclude_tasks` always take precedence. The internal dummy
environment is excluded unless named explicitly. An unfiltered loader includes
extended and experimental suites. To match the standard benchmark, use:

```python
env = glyphbench.load_environment(
    exclude_suites=["atari", "craftaxfull", "nethack", "agentick"],
    num_episodes=3,
)
```

An explicit `task_id` cannot be combined with include/exclude filters. Optional
adapters may require their dependencies before they appear in the registry.

The [evaluation guide](../eval/README.md) covers model serving, sampling
settings, and complete command examples. [Memory behavior and failure
handling](OBSERVATION_FORMAT.md#memory-mode) are described separately.

## Inspecting trajectories

Use the replay UI to browse Verifiers evaluation results:

```bash
uv run glyphbench replay path/to/runs --env glyphbench/minigrid-empty-5x5-v0 --pause
```

For a standalone trajectory JSONL file or GIF export:

```bash
uv run python scripts/replay_trajectory.py path/to/trajectory.jsonl
uv run --extra assets python scripts/replay_trajectory.py path/to/trajectory.jsonl --gif out.gif
```

These tools consume different saved formats; see [replay](REPLAY.md) and the
[script index](../scripts/README.md). Prime-RL v1 token traces can be inspected
with `scripts/rl/audit_prime_rl_trace.py`.

## Determinism

The same seed and action sequence reproduce an environment's observations and
rewards. Reproducing a model rollout also requires the same policy, sampling,
and harness settings; environment seeding alone does not fix model randomness.
The evaluation loader uses the same task order and episode seeds when its
selection and seed settings are unchanged.
