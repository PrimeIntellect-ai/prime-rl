# `glyphbench.core`

The core defines the environment interface, observations, actions, and registry.
Its modules use Python and NumPy without importing model providers or harnesses.

## Public types

| Type | Purpose |
|---|---|
| `GridObservation` | Frozen dataclass containing `grid`, `legend`, `hud`, and `message` strings. `render()` produces the canonical text observation. |
| `ActionSpec` | Fixed action vocabulary with tuple-valued `names` and `descriptions`, optional aliases, and `index_of(name)`. |
| `BaseGlyphEnv` | Plain abstract class defining seeded reset, integer-indexed steps, turn limits, rewards, and termination. |

`env.reset(seed=42)` requires an explicit integer seed, passed by position or
keyword. `env.step(action_index)` accepts integer indices only; convert names
with `env.action_spec.index_of(name)`. The action vocabulary is not filtered by
which actions are effective in the current state.

Subclasses define `action_spec` and implement `_reset`, `_step`,
`_render_current_observation`, `system_prompt`, and `env_id`. Observation fields
are always strings; optional fields can be empty. The base class enforces the
standard cumulative return bound, while full-game adapters can opt out to
preserve native rewards.

## Registry and seeds

`register_env` associates a task ID with an environment class and its default
memory setting. `make_env` constructs a registered task; `list_task_ids` applies
suite and task filters. Optional dependencies can affect which adapters are
registered.

Standard games use the seeded `env.rng` NumPy generator. Upstream adapters
seed their native simulators. The same seed and action sequence should produce
the same observations and rewards.

See the [integration guide](../../../docs/INTEGRATION.md) for a complete game
loop and the [contributor guide](../../../CONTRIBUTING.md) for environment
contracts and validation.
