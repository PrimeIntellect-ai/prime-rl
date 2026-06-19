# Eval Client Type Mismatch in Renderer Runs

Date: 2026-06-19

## Summary

`orchestrator.renderer` configures the direct renderer rollout path, but in
`setup_student_inference_pool` the renderer branch hardcodes different client
types for train and eval:

```python
train_client_type = "renderer"
eval_client_type = "openai_chat_completions"
```

This means a config with `[orchestrator.renderer]` uses the renderer client for
training rollouts while SWE-Bench eval still goes through the OpenAI
chat-completions client. The config selects the renderer implementation and
enables renderer-backed training rollout tokenization, but it does not currently
control eval client type.

## Why It Matters

For the Nemotron Nano SWE run, eval rollouts were failing through the
chat-completions eval path with `EmptyModelResponseError`-style symptoms around
tool-call turns. A local experiment changed eval to use the renderer client and
forwarded renderer config/model name into eval client construction. With that
change, step-0 SWE-Bench eval completed:

```text
Evaluated swe-bench-verified-quick (Step 0) | Policy v0 | 46m 29s | Reward 0.3397 | Turns 29.9 | Error 0.0% | Truncation 2.6%
```

Live logs for that run did not contain `EmptyModelResponseError` or
`TruncatedReasoningError` after switching eval to the renderer client.

## Observed Raw Response Shape

The raw-dump instrumentation still captured malformed renderer tool-call turns.
All five live dumps had the same summary shape:

```text
finish_reason=stop
has_content=false
has_raw_tool_calls=true
has_usable_tool_calls=false
status=malformed_structure
```

One dump came from eval rollout `rollout_eb20d2a5` / example `22`; that rollout
still finished `agent_completed`. The other four dumps were train rollouts and
also finished `agent_completed`. This suggests the fatal eval symptom was not
simply "model returned nothing"; it was the eval chat-completions client path
handling renderer/tool-call output differently from the direct renderer client.

## Relevant Code Paths

- `src/prime_rl/orchestrator/utils.py`
  - creates the renderer for `config.renderer is not None`
  - passes `train_client_type="renderer"`
  - hardcodes `eval_client_type="openai_chat_completions"`
- `src/prime_rl/utils/client.py`
  - static inference pool forwards renderer config/model name only to train
    clients in the baseline code
- `src/prime_rl/utils/elastic.py`
  - elastic inference pool has the same train/eval asymmetry

## Fix Direction

Two reasonable fixes are possible:

1. Make eval follow train whenever `orchestrator.renderer` is configured.
2. Add an explicit config field such as `orchestrator.eval_client_type`, validate
   that `"renderer"` requires `orchestrator.renderer`, and forward renderer
   config/model name to eval client construction in both static and elastic
   inference pools.

This repro branch contains a diagnostic workaround that hardcodes eval to the
renderer client and forwards renderer config/model name into eval clients. It is
useful for reproducing the validation result above, but it is not intended as a
polished production fix. A production fix should either make the train/eval
policy explicit in config or deliberately choose "renderer applies to both train
and eval" as the default behavior.

## Local Evidence

Run artifacts are on the training machine:

- `/root/outputs/nemotron-nano-swe/STATUS.md`
- `/root/outputs/nemotron-nano-swe/logs/orchestrator.log`
- `/root/outputs/nemotron-nano-swe/raw-dumps/live/`

The raw-dump instrumentation used for this investigation is in the nested
`verifiers` repro branch.
