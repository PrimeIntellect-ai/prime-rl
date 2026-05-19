# Online Test-Time Training LoRA Notes

This note describes the online test-time training (TTT) LoRA experiment we explored for Prime-RL, the requirements for making it correct, and the main implementation and systems challenges encountered during the first migration attempts.

The design is experimental. It is meant to support research runs, not a merge-ready production feature.

## Goal

The goal is to extend the effective context available to an RL-trained agent without requiring attention over the full history at every generation step.

Instead of keeping all prior tokens in the transformer attention window, the rollout maintains a small per-rollout LoRA. As new exact rollout tokens arrive, the LoRA is trained online at inference time. Later generations use the newest base policy weights plus that rollout-local LoRA, so information from prior context can be represented in weights rather than only in KV cache.

The motivating use cases are:

- Long retrieval or browsing tasks where relevant information may appear earlier than the live attention window.
- Structured environments where repeated tool outputs define a local world model.
- Continual-learning style deployment where the model adapts quickly to an environment without permanently changing the base model during the rollout.
- RL training where rollouts must still be replayable under the policy and adapters that were active when the sampled tokens were generated.

## Final Shape We Converged Toward

The original design had two rollout-local LoRAs:

- `Phi_p`, trained on prompt-side environment/tool response tokens and optionally merged into `Theta`.
- `Phi_c`, trained on model completion tokens and discarded after replay.

That split was removed. The simpler design is:

- `Theta`: the main model weights, updated by normal Prime-RL trainer steps.
- `Phi_ttt`: one rollout-local LoRA, trained online on exact rollout tokens.
- Tool-output world-modeling: a separate trainer-side auxiliary SFT loss on selected tool-output content tokens.

The split is important:

- Online TTT is temporary state for context extension and replay correctness.
- Permanent tool-output training is a normal trainer loss applied to `Theta`.
- TTT adapters are never merged into `Theta`.
- Tool-output SFT does not depend on TTT being enabled.

## Online TTT Procedure

Each rollout owns one TTT session.

For each generation turn:

1. The renderer constructs the exact prompt token ids that will be sent to vLLM.
2. The TTT client identifies the newly added exact prompt tokens since the previous turn.
3. Those new tokens are appended to the session's pending TTT token buffer.
4. For every full `update_every_tokens` chunk in that buffer, the learner trains the rollout LoRA.
5. Generation runs with the current base model plus the current LoRA snapshot, if a LoRA snapshot exists.
6. Completion token ids are appended to the same pending TTT token buffer after generation.
7. New full chunks created by completion tokens are trained.
8. The adapter snapshot used for generation is recorded in the rollout trace for trainer replay.

The default intended TTT cadence is:

```toml
[experimental.ttt]
enabled = false
mode = "online_lora"
window_seq_len = 8192
total_seq_len = 32768
update_every_tokens = 1024
require_exact_token_ids = true

[experimental.ttt.lora]
rank = 8
alpha = 16
dropout = 0.0
target_modules = "auto"
```

`update_every_tokens = 1024` means the LoRA must update after every complete 1024-token chunk. This is not just a performance knob. It affects correctness: tokens after a chunk boundary should be predicted under the adapter trained on the previous chunk.

For a long tool output, the correct semantics are:

```text
before tool output: adapter A0

tokens 0..1023    are predicted under A0
TTT trains on tokens 0..1023 -> adapter A1

tokens 1024..2047 are predicted under A1
TTT trains on tokens 1024..2047 -> adapter A2

tokens 2048..end  are predicted under A2
```

This prevents future-token leakage while still allowing the LoRA to compress earlier parts of a long prompt bridge before later parts are scored or generated.

## Replay Requirement

The trainer must compute RL gradients against the same temporary adapter state that was active when each completion span was sampled.

For sampled model tokens:

- The base `Theta` used for training should be the newest trainer weights.
- The TTT adapter must be the frozen snapshot that was active at sampling time.
- Gradients flow only into `Theta`.
- Adapter tensors are replay context, not trainable parameters in the trainer.

This is why rollouts carry a `ttt_trace`. Each trace entry needs enough information to reconstruct:

- session id
- turn index
- base step observed during generation
- adapter name and path, if one existed
- completion span token offsets
- prompt-side chunk trace information when prompt bridge scoring needs intra-bridge adapter changes

Without this trace, the trainer either ignores TTT during replay or accidentally trains on tokens under the wrong adapter.

## Tool-Output World-Modeling

Tool-output training should be independent from online TTT.

Normal RL loss only applies to model-generated tokens with behavior-policy logprobs and advantages. Tool outputs are not sampled by the policy, so they should not enter the RL importance-ratio loss.

Instead, when enabled, tool-output training adds an auxiliary hard-target NLL/SFT loss in the same forward/backward/optimizer step:

```toml
[experimental.tool_output_training]
enabled = false
weight = 1.0
tool_names = null
content_only = true
```

For the Forth experiments, the intended setting was:

```toml
[experimental.tool_output_training]
enabled = true
weight = 0.5
tool_names = null
content_only = true
```

Later we found that `tool_names` needs to be environment-scoped rather than global. A global setting like:

```toml
tool_names = ["run_code", "submit_code"]
```

is correct for Forth but wrong for DeepDive or other eval environments. Per-environment settings are required so train environments can select their own tools without poisoning unrelated eval rollouts.

The tool-output mask must include only actual tool response content tokens. It must exclude:

- role tags
- separator tags
- renderer scaffolding
- tool-call wrappers
- prompt formatting tokens that were not produced by the tool

Renderer-provided token attribution is therefore required for this path.

## Renderer Requirements

TTT should use the renderer path, not the older token-client rollout path.

The renderer is the right integration point because it has:

- exact prompt token ids immediately before generation
- message-to-token attribution through `message_indices`
- content masks that exclude separators and control tokens
- enough structure to recover prompt bridge boundaries
- tool names and message roles where available

TTT mode should require:

```toml
[orchestrator]
use_renderer = true
use_token_client = false
```

For online TTT, the renderer must:

- strip TTT control keys before forwarding sampling args to vLLM
- reject missing exact token ids when exact mode is required
- apply prompt windowing before generation
- include adapter identity in cache salt
- preserve TTT trace metadata in rollout output
- keep train and eval sessions isolated

For sliding-window-only mode, renderer mode is still required if the implementation relies on renderer-side prompt token windowing. Otherwise TTT control keys can leak into non-renderer OpenAI-compatible client paths and do nothing useful or cause unknown-parameter failures.

## Windowing Requirements

The attention window must reserve room for both prompt tokens and generated completion tokens.

We encountered a serious bug where the TTT Forth config used:

```toml
window_seq_len = 8192
max_completion_tokens = 8192
```

The renderer computed a prompt budget approximately as:

```text
prompt_budget = window_seq_len - max_completion_tokens - headroom
```

That left effectively one prompt token. The learner logs confirmed repeated `prompt_tokens=1` in `/prepare_turn`. The run technically started, but it was not meaningful: the model was generating almost without prompt context.

The config must instead leave real prompt budget, for example:

```toml
window_seq_len = 8192
max_completion_tokens = 2048
max_total_completion_tokens = 32768
```

or:

```toml
window_seq_len = 8192
max_completion_tokens = 4096
max_total_completion_tokens = 32768
```

The code should validate this and fail loudly when:

```text
max_completion_tokens + headroom >= window_seq_len
```

Silently clamping to a one-token prompt makes the run look alive while invalidating the experiment.

## Learner Requirements

The TTT learner is a separate service that owns rollout-local LoRA state.

Required API shape:

- `/start_session`
- `/prepare_turn`
- `/complete_turn`
- `/finish_session`
- `/abort_session`
- `/update_base_weights`
- `/health`

Important behavior:

- One session per rollout trajectory.
- One LoRA and one optimizer per active session.
- Frozen base model weights can be refreshed when Prime-RL broadcasts a new `Theta`.
- Refreshing `Theta` must not reset active LoRA tensors or optimizer state.
- Sessions must be closed or aborted when rollouts complete, fail, or are rescheduled.
- Learner concurrency must be bounded and explicit.
- Admission failure should be fast and clear, for example 429 or 503, not an indefinite wait.

The learner should keep inactive session state in CPU memory by default:

```toml
[experimental.ttt.learner]
session_offload = "cpu_after_request"
adapter_dir = "/large/shared/path"
unload_vllm_adapters = true
delete_consumed_adapters = true
trainer_cache_device_tensors = false
```

Disk is for materialized adapter snapshots needed by vLLM and trainer replay. It should not be the hot-path storage for active LoRA optimizer state unless we deliberately accept much higher latency.

## Adapter Lifecycle Requirements

Adapters should live for as little time as correctness allows.

Correct lifecycle:

1. Active session LoRA lives in learner memory while the request is being processed.
2. Session LoRA and optimizer state are offloaded to CPU after the request.
3. Per-turn adapter snapshots are materialized to shared storage for vLLM generation and trainer replay.
4. vLLM adapter is unloaded after the generation request that used it returns.
5. Trainer loads frozen adapter snapshots only for replay.
6. Trainer evicts CPU/GPU adapter caches after backward.
7. Rank 0 deletes consumed adapter directories only after replay and optimizer step complete successfully.

Never delete an adapter before trainer replay has consumed it. Never unload a vLLM adapter before its generation request has returned.

## Base Weight Update Requirements

`Theta` should always be the newest available policy.

During PipelineRL-style updates:

- vLLM receives the newest policy weights.
- the TTT learner refreshes its frozen base weights.
- active rollout LoRAs remain alive and unchanged.
- rollout sessions should not be cancelled just because `Theta` updated.

This is conceptually different from merging TTT state. The LoRA is temporary rollout state layered on top of the newest base.

We saw vLLM errors around dynamic weight update while TTT adapters were active, including failures near `layers.0.mlp.gate_up_proj.weight`. That path needs careful isolation:

- pause generation before weight update
- make adapter load/unload and base weight update serialization explicit
- avoid closing sessions while the learner is training or materializing adapters
- consider disabling mid-rollout policy refresh for initial smoke tests if dynamic update remains unstable

## Train/Eval Isolation

Evaluation should use TTT if training uses TTT, because otherwise the evaluation path measures a different inference algorithm.

However, there must be no leakage between train and eval:

- Eval rollout sessions must be fresh.
- Train sessions must never be reused by eval.
- Eval adapters must not be merged into the base model.
- Eval adapters must be deleted after eval replay/scoring is no longer needed.
- Tool-output training controls should not automatically leak from train envs into eval envs.

This distinction matters for Forth and DeepDive-style setups, where train and eval tools or data distributions can differ.

## Performance Challenges Observed

The first end-to-end runs exposed several bottlenecks.

### One learner became a bottleneck

With many concurrent rollouts, a single learner serialized too much work:

- LoRA forward/backward updates
- adapter materialization
- vLLM adapter loading
- base weight refresh

Multi-learner routing can help, but only if it preserves correctness:

- sticky route each trajectory to one learner
- keep session state local to that learner
- make adapter paths visible to trainer and vLLM
- avoid multiple copies of full base weights if VRAM does not allow it

An alternative for early experiments is to reduce rollout concurrency and oversampling rather than making the TTT service distributed immediately.

### Adapter materialization and vLLM loading are expensive

Online updates every 1024 tokens imply many adapter versions. If each version is saved, loaded into vLLM, used once, unloaded, then replayed and deleted, the storage and admin-path overhead can dominate.

Potential optimizations:

- keep active LoRA in CPU memory, not disk
- materialize only when generation or replay needs a snapshot
- batch adapter load/unload operations where safe
- reduce unnecessary snapshots before the first trained chunk
- use stricter concurrency limits to avoid overloading vLLM adapter management

### Locking has a correctness/performance tradeoff

The learner has shared base model state. Training, materialization, base refresh, and session close all need serialization boundaries.

Too much locking makes the system slow. Too little locking risks:

- training while base weights are being refreshed
- materializing a half-updated adapter
- closing a session while a forward hook is using it
- unloading an adapter while vLLM still needs it

The safe starting point is conservative locking. Performance can be improved only after profiling confirms which operations can be moved outside the lock without changing semantics.

## Correctness Bugs Found During Bringup

The following issues were found while trying to run DeepDive and Forth experiments.

### Tool-output mask propagation

The renderer client produced `tool_output_train_mask`, but intermediate Verifiers/Prime-RL token structures initially dropped it. The trainer then saw no tool-output tokens and reported zero tool-output loss.

Required fix:

- preserve `tool_output_train_mask` through response token parsing
- preserve it through trajectory interleaving
- initialize false prefixes when later steps add prompt-side tool-output masks
- pack/pad it into trainer microbatches

### Global tool-name filters were wrong

Forth-specific tool names were applied globally. That can disable or distort tool-output training/eval for other environments.

Required fix:

- allow per-environment `tool_output_training`
- inherit global settings only for train envs where intended
- do not inject train tool filters into unrelated eval envs

### TTT session cleanup could miss sessions

Session cleanup depended on rollout metadata that was not always present in output columns. Failed or rescheduled rollouts could leak learner sessions and optimizer state.

Required fix:

- include trajectory id in required rollout state
- fall back to session ids in `ttt_trace`
- abort partial sessions on rollout failure/reschedule
- close sessions at eval/train completion

### TTT replay could double-count

If a single rollout becomes multiple training samples, attaching the full `ttt_trace` to every sample can replay the same completion spans multiple times.

Required fix:

- either force one TTT replay sample per rollout
- or split trace spans exactly across samples

The simpler experimental path is one TTT rollout sample per microbatch.

### Stale tests and old split-LoRA assumptions

Some tests still expected fields from the old prompt/completion LoRA design:

- `token_role`
- `new_prompt_ids`
- `ttt_final_prompt_adapter`

The single-LoRA design should instead test:

- `new_token_ids`
- chunked append-and-train behavior
- trace entries with optional adapter paths
- no TTT control keys leaking into vLLM sampling args

## Experiment Requirements

A meaningful experiment matrix should include:

- RL baseline at 8k context, no TTT.
- RL baseline at 32k context, no TTT.
- RL plus tool-output world-modeling at 8k.
- RL plus tool-output world-modeling at 32k.
- TTT with 8k attention window and 32k total trajectory budget.
- TTT plus tool-output world-modeling for structured environments such as Forth.

For retrieval tasks, the core question is:

- Does online TTT let an 8k-window model behave as if it had access to useful information from a 32k trajectory?

For structured environments, the core question is:

- Does temporary TTT state plus permanent tool-output world-modeling help the model learn environment dynamics and code execution traces?

DeepDive-style environments turned out to have possible reward-hacking paths, so DDBC or a cleaner held-out retrieval benchmark is preferable for online eval. For Forth, evals should include the Forth test set and cheap generalization checks such as general-agent, DeepDive, and Wordle if they are not contaminated by train-specific tool filters.

## Minimal Safe Run Checklist

Before starting another TTT run:

1. Confirm the code and config repos are on feature branches, not `main`.
2. Confirm the cluster checkout has the pushed feature-branch commits.
3. Confirm `orchestrator.use_renderer = true` and `orchestrator.use_token_client = false`.
4. Confirm `window_seq_len > max_completion_tokens + headroom`.
5. Confirm `max_total_completion_tokens` carries the desired long-trajectory budget.
6. Confirm vLLM LoRA is enabled and `max_lora_rank >= experimental.ttt.lora.rank`.
7. Confirm `adapter_dir` is on shared storage visible to learner, vLLM, and trainer.
8. Confirm TTT learner concurrency is high enough to avoid immediate admission failures but low enough not to overload vLLM.
9. Confirm tool-output training filters are environment-scoped.
10. Confirm eval envs do not inherit train-only tool filters.
11. Confirm TTT sessions are closed after train and eval rollouts.
12. Confirm logs show real prompt budgets, not `prompt_tokens=1`.
13. Confirm W&B reports nonzero rollout progress and sane eval completion rates before treating the run as valid.

## Open Questions

The main unresolved questions are:

- Whether online LoRA updates every 1024 tokens are fast enough at useful rollout concurrency.
- Whether vLLM adapter load/unload overhead dominates before model compute does.
- Whether dynamic base weight updates are stable while TTT adapters are active.
- Whether multi-learner routing is worth the extra systems complexity.
- Whether chunk-level prompt replay is sufficient for all long tool-output cases or needs finer-grained trace segments.
- Whether the TTT signal helps beyond simpler baselines such as 32k RL and tool-output world-modeling.

The next implementation work should focus on making invalid configurations impossible, reducing adapter overhead, and producing one clean smoke run before expanding the experiment matrix.
