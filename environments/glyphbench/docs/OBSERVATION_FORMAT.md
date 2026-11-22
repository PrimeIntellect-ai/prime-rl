# Observation format and harness

## Environment observations

`GridObservation.render()` produces a text string with sections in this order:

| Section | Contents |
|---|---|
| `[Legend]` | Meanings of the glyphs used by the environment. Omitted when empty. |
| `[HUD]` | Health, inventory, velocity, turn budget, and other complementary state. Omitted when empty. |
| `[Grid]` | The two-dimensional Unicode grid. Always present. |
| `[Message]` | Feedback from the last action. Omitted when empty. |

Each grid cell contains one Unicode codepoint. The per-environment legend is
the reference for its meaning; common examples are `█` for walls, `★` for goals,
and `→ ↓ ← ↑` for player direction. The HUD supplies information that the grid
cannot convey clearly. It should not duplicate visible positions or facing
already encoded by the glyphs.

Terminal replay and glyph-based GIF export may add color, but preserve the
observation's glyphs. The optional full-game adapters also expose native text
and pixel observation modes for separate experiments.

## Model prompts and actions

The evaluation harness combines game rules from `env.system_prompt()` with
`env.action_spec.render_for_prompt()` and instructions for returning an action.
The `[Actions]` menu appears in the system prompt, rather than in every raw
observation. A model chooses from that fixed vocabulary and returns a complete
tag such as:

```text
<action>MOVE_FORWARD</action>
```

The parser resolves the name through the environment's `ActionSpec`. Unknown
names and incomplete tags are parse failures. By default, the evaluation
harness retries once within the same turn. If parsing still fails, the turn
is forfeited: the clock advances without changing the game state, with zero
reward. `forfeit_mode="noop"` applies a fallback action instead, allowing the
game's dynamics and reward to advance.

`max_output_tokens` communicates an output budget in the prompt; the caller
must also set the corresponding limit in its model sampling configuration.
It is distinct from the environment horizon and the inference context limit.

## Observation history

In the evaluation loader, `n_frames=N` retains up to N previous observations
with their actions and rewards, followed by the current observation. The
legend is merged and deduplicated across those frames within each prompt.
With `n_frames=0`, the prompt contains only the current observation and any
explicit memory. Some games are partially observed, so this setting does not
make every task Markovian.

Prime-RL uses the native Verifiers v1 conversation and its context budget.
It does not use the legacy evaluation loader's frame-stacking mechanism.

## Memory mode

The evaluation loader's `use_memory=None` follows each task's registered
memory default. Pass `True` or `False` to override that choice for the selected
tasks. The v1 training integration requires `use_memory=false`.

When memory is enabled, a turn normally uses two model generations:

1. The action call receives the current observation and the previous memory,
   then returns reasoning and an action tag.
2. After the action is applied, the memory call receives the action's reasoning
   and parsed outcome, the reward and termination flags, and the next
   observation. It returns `<memory>...</memory>` with thinking disabled via
   `chat_template_kwargs.enable_thinking=False`.

The next action prompt includes the stored text in a `[Memory]` block. Memory
is intended for plans, discoveries, and facts that are no longer visible; the
current observation remains authoritative when they conflict. A final action
that ends the episode does not need a subsequent memory update.

Action and memory generations are stored as separate trajectory steps, each
with its own prompt and completion. Replay displays both the previous and
updated memory when available. If a memory response cannot be recovered, the
previous memory is retained and `memory_parse_failed` is recorded.
`memory_update_max_tokens` caps the memory generation; reaching the output
limit is recorded separately from a parsing failure.

A memory call contains the action prompt, action response, next observation,
and update instructions before its own output. Account for all of these when
choosing a context limit. The standard evaluation scripts use a 65,536-token
context and 4,096-token action and memory output budgets. Larger histories,
observations, or output budgets may require a larger context.

## Reproducibility and diagnostics

Environment trajectories are deterministic given a seed and action sequence.
Observations can hide state by design; the agent receives the visible grid,
HUD, and messages rather than privileged simulator state. Model sampling and
harness settings must also be fixed when comparing agents.

The [integration guide](INTEGRATION.md) documents loader options. The
[failure-mode reference](llm-agent-failure-modes.md) distinguishes game timeouts,
context limits, output truncation, and parsing failures.
