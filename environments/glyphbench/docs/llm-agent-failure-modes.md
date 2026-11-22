# Agent failure diagnostics

A game timeout, exhausted model context, truncated response, and invalid
action are different events. This guide explains the labels used in saved
Verifiers evaluations and replay. The native v1 training path also records
return, episode length, forfeits, and context-limit stops, but does not use
memory-update calls.

## Events

| Event | Meaning |
|---|---|
| Episode terminated | The game reached an end condition, such as success or death. |
| Episode truncated | The game or its configured turn limit ended the episode before another terminal condition. |
| Context-limit hit | The inference or training framework stopped because the next generation would exceed its context capacity. |
| Action completion truncated | An action response reached its output-token limit. A complete action tag may still be present. |
| Memory completion truncated | A memory response reached its output-token limit. This is independent of whether memory could be parsed. |
| Parse recovery | An invalid action response was followed by a successful same-turn retry. |
| Forfeit | The final action response could not be parsed. In `freeze` mode, only the clock advances and reward is zero. In `noop` mode, the game applies its fallback action and returns that action's reward. |
| Memory parse failure | The memory response could not be recovered; the previous stored memory is retained. |

The standard evaluation settings use a 4,096-token action budget and a
65,536-token context. Both are configurable, and extended full-game runs use
larger defaults. Check the saved run settings before interpreting a limit.

## Evaluation trajectory fields

Each model generation is stored as a trajectory step. With memory enabled,
actions and memory updates are separate steps; the final action may have no
memory update. `TrajectoryStep.is_truncated` records output truncation.

| Extra field | Where | Meaning |
|---|---|---|
| `glyphbench_step_role` | Both | `"action"` or `"memory"`. |
| `parse_failed` | Action | The final action response could not be parsed. |
| `parse_failure_reason` | Action | `"no_action_tag"`, `"unknown_name"`, or `None` on success. |
| `action_chosen` | Action | Canonical action name or `"FORFEIT"`. |
| `forfeit` | Action | Whether parsing failed after any retries. |
| `parse_recovered` | Retried action | Whether a retry produced a valid action. |
| `parse_retry_attempts` | Retried action | The retry responses, for inspection. |
| `memory_parse_failed` | Memory | Whether extraction failed. |
| `stored_memory` | Memory | Memory in effect after the update. |

The memory protocol asks for `<memory>...</memory>`. The evaluation parser also
supports recovery of a nonempty response with an accidental action tag; a
missing memory tag alone is therefore not sufficient to infer a recorded
memory parse failure.

## Evaluation metrics

Rates are calculated per rollout; evaluation summaries then aggregate them.
A rate with no applicable action or memory turns is zero.

| Metric | Definition |
|---|---|
| `episodic_return` | Sum of game rewards; the only weighted task reward. |
| `episode_length` | Number of action turns, including forfeits. |
| `episode_terminated_rate` | 1 if the game terminated, otherwise 0. |
| `episode_truncated_max_turns_rate` | 1 if the game's `truncated` flag was set, otherwise 0. |
| `context_limit_hit` | 1 for a recognized framework context-capacity stop. |
| `forfeit_rate` | Forfeited turns divided by action turns. |
| `parse_recovery_rate` | Recovered action turns divided by action turns. |
| `action_completion_truncation_rate` | Truncated action responses divided by action turns. |
| `memory_completion_truncation_rate` | Truncated memory responses divided by memory turns. |
| `memory_parse_failure_rate` | Failed memory updates divided by memory turns. |
| `xml_format_reward` | Verifiers' XML-format compliance metric, when supplied by the parser. |

Standard benchmark returns are bounded to `[-1, 1]`. Full Craftax and NetHack
report native rewards; their values should not be mixed into the standard
aggregate without an explicit scoring protocol.

## Investigating a run

- If forfeits are frequent, inspect both the returned text and output-truncation
  rate. An incomplete tag after a long response calls for a different fix than
  a short response with an unknown action name.
- If game timeouts are frequent and returns are low, replay trajectories to
  distinguish slow progress from repeated ineffective actions or forfeits.
- If memory parsing fails, inspect the memory response and model chat-template
  settings. A retained old memory may explain repeated mistakes on later turns.
- If context limits end episodes, inspect the full prompt and history budget.
  Increasing the environment horizon does not increase model context capacity.

Replay displays `[forfeit]`, `[trunc-action]`, `[trunc-memory]`, and
`[mem-parse-fail]` indicators when the corresponding metadata is available.
Use the [replay controls](REPLAY.md) to inspect full panel contents.

## Implementation

- [Action parsing](../src/glyphbench/protocol.py)
- [Forfeit transitions](../src/glyphbench/core/base_env.py)
- [Memory extraction](../src/glyphbench/verifiers_integration/memory.py)
- [Evaluation state and trajectory fields](../src/glyphbench/verifiers_integration/env.py)
- [Evaluation rubric](../src/glyphbench/verifiers_integration/rubric.py)
- [Native v1 integration](../src/glyphbench/verifiers_v1.py)
- [Replay rendering](../src/glyphbench/cli.py)
