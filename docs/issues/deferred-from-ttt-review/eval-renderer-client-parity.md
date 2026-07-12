# Proposal: allow eval to use the renderer/token client explicitly

## Status

Deferred feature proposal, not an established Prime-RL bug.

## Motivation

Prime-RL normally routes evaluation through the chat-completions relay for continuity with
existing evaluation results. Some experiments need evaluation to follow the renderer/token
path used by training—for example, to compare a treatment that requires exact token IDs with a
baseline under the same transport and chat-template behavior.

## Change considered on the TTT branch

The branch added `EvalEnvConfig.use_renderer_client: bool = False` and taught the dispatcher to
select the renderer client when enabled. TTT eval already requires the renderer path and was
handled automatically; the generic flag was mainly used to make A0/A1 comparisons transport-
matched to TTT arms.

## Why it is deferred

This adds a public configuration option and another evaluation mode to Prime-RL. It is useful
experiment infrastructure but not required to implement TTT itself. Baseline comparability can
instead be documented as a limitation until the option is reviewed on its own.

## Questions

- Which differences exist between renderer and chat-relay evaluation today?
- Should the result metadata record the selected transport?
- Are all eval harnesses compatible with token IDs and renderer-specific messages?
- Should this be an env-level option or inferred from harness capabilities?

## Suggested tests

- Default eval routing remains byte-for-byte unchanged.
- The explicit option selects the renderer pool and model name.
- Mixed eval environments can select different transports safely.
- Result metadata makes cross-mode comparisons auditable.

## Relevant code

- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py`
- `src/prime_rl/orchestrator/dispatcher.py`
- experiment configs that need matched eval transport
