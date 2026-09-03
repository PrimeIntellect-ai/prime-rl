# Fork-specific changes

This branch is based on Prime-RL `v0.7.1.dev83` (`2ffe374e0`). It contains
only the Prime-RL changes needed by the DataDev search-agent experiments.
DataDev's Beaker launchers, semaphore lifecycle management, auxiliary tool
servers, and experiment TOML files remain in the DataDev repository.

## Trainer controls

Commit: `aa63d80d3` (`feat(trainer): add group-balanced RL reduction`)

### Group-token-mean RL loss

The default remains the upstream token-weighted reduction:

```toml
[trainer.loss]
aggregation = "token_mean"
```

The fork adds a group-balanced reduction:

```toml
[trainer.loss]
aggregation = "group_token_mean"
```

For `group_token_mean`, Prime-RL sums the active RL-token losses for each
rollout group, divides by that group's active RL-token count, and then averages
the resulting group means across the groups represented in the training batch.
Group counts are computed after filtering, masking, and truncation. A batch may
contain an incomplete rollout group; it is normalized using the members and
active tokens actually present in that batch.

The implementation carries a rollout `group_id` through transport, stamps the
post-packing group token counts on trainer microbatches, and accounts for
context-parallel replication during loss normalization. CE and reference-KL
losses retain their upstream token-mean reductions.

### AdamW epsilon

The fork exposes AdamW's numerical-stability epsilon:

```toml
[trainer.optim]
eps = 1e-12
```

The default is `1e-8`, matching the previous implicit optimizer default.

## Legacy rollout budget preservation

Commit: `dc0c61dc7` (`fix(config): preserve legacy rollout token budgets`)

For legacy multi-turn environments, an explicit
`legacy.extra_env_kwargs.max_total_completion_tokens` is no longer overwritten
by the agent's per-turn `max_output_tokens`. This permits a large total episode
budget while retaining a smaller per-generation sampling cap.

## Independent evaluation capacity

Commit: `20a2a730a` (`feat(orchestrator): separate evaluation inflight capacity`)

The fork adds:

```toml
[orchestrator]
max_inflight_episodes = 128
eval_max_inflight_episodes = 512
```

Training and evaluation each obey their own in-flight episode limit. Their
combined occupancy is capped at the larger of the two limits, so enabling a
larger evaluation window does not sum both limits and overrun inference
capacity. When omitted, `eval_max_inflight_episodes` defaults to the resolved
training limit.

## Renderer-backed evaluation

Commit: `eb51f09ad` (`feat(orchestrator): run evaluation through renderers`)

Evaluation uses the configured renderer instead of the plain OpenAI chat
completions client. Static and elastic inference pools both propagate renderer
configuration, renderer model name, and pool size to their evaluation clients.
This aligns training and evaluation tokenization and response parsing.

## Qwen3.5 implicit-thinking parser

Prime-RL commit: `dbbe1d7c4`

Renderer fork commit: `a2f694903ac1e44b0a273df2bea992ea6f68744f`

The renderer submodule points to `goncalorafaria/renderers`. When a
thinking-enabled Qwen3.5 generation prompt opens the thinking block and the
completion never emits the closing token, the entire completion remains
reasoning. Apparent final text or tool-call XML inside that unclosed block is
not exposed as an answer or executable tool call.

## Container overlay

Commit: `9a0464209` (`chore(container): add Qwen3.5 evaluation overlay`)

`Dockerfile.qwen35-renderer-eval-inflight` is the minimal overlay used for the
combined renderer-evaluation and independent-evaluation-capacity image. It is a
deployment artifact, not another runtime behavior change.

## Upstream comparison

The original snapshot is preserved on
`codex/datadev-training-extensions` at `227e38eea`. The review branch
`feat/datadev-training-extensions` has the same source tree, split into the
focused commits above plus this documentation commit.
