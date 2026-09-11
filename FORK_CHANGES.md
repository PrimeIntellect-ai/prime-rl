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

### Research basis: prompt-level loss averaging

This option implements the group/token averaging structure called `prompt_mean`
in [Mercor and SkyRL's training guide, Step 4](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)
(September 1, 2026). They report a 3.9-point improvement over `token_mean` in
their Qwen3.6-35B-A3B ablation. This is evidence from their experiment, not a
measured gain for this fork.

The paper reference is [ScaleRL: The Art of Scaling Reinforcement Learning
Compute for LLMs, Section 3.2 (Loss Aggregation) and Section 4](https://arxiv.org/html/2510.13786v1#S3.SS2).
It distinguishes sample, prompt, and global-token averaging and adopts prompt
averaging. [DAPO, Section 3.3, Equation 12](https://arxiv.org/html/2503.14476v1#S3.SS3)
also normalizes the summed token objective by the total completion length
within a prompt's rollout group, inside the expectation over prompts.

For this fork, let `A_g` be the retained tokens in rollout group `g` with an
active loss mask and nonzero RL weight; `N_g = len(A_g)`, and `K` is the number
of groups with `N_g > 0`. The reduction is:

```text
group_token_mean = (1 / K) * sum_g [sum_{t in A_g} weighted_rl_loss_t / N_g]
token_mean       = sum_g sum_{t in A_g} weighted_rl_loss_t / sum_g N_g
```

Thus groups receive equal outer weight; longer sequences still contribute more
tokens within their group. The correspondence assumes a rollout group denotes
the samples for one prompt. This fork uses the active tokens and group members
actually retained in the training batch, including incomplete groups, rather
than assuming all originally sampled completions survive. These citations
support the aggregation structure, not equivalence to the full DAPO, ScaleRL,
or Mercor recipe: clipping, advantages, filtering, and other loss terms are
separate choices. CE and reference-KL retain their existing reductions.

### AdamW epsilon

The fork exposes AdamW's numerical-stability epsilon:

```toml
[trainer.optim]
eps = 1e-12
```

The default is `1e-8`, matching the previous implicit optimizer default.

### Research basis: configurable AdamW epsilon

[Microsoft AI, *MAI-Thinking-1: Building a Hill-Climbing Machine*,
Section 3.1.5, page 36](https://microsoft.ai/pdf/mai-thinking-1.pdf#page=36)
reports AdamW epsilon `1e-15` for its RL climbs. This motivates exposing `eps`
so published RL optimizer settings can be reproduced. The fork retains its
`1e-8` default; its `1e-12` example is not the MAI setting. To select MAI's
epsilon, set `[trainer.optim] eps = 1e-15`. This changes only epsilon, not the
rest of MAI's optimizer recipe.

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

## Explicit training top-p

Commit: `dc1972200` (`Honor explicit top-p during training rollout inference`)

```toml
[orchestrator.train.sampling]
top_p = 0.97
```

`TrainSamplingConfig.to_sampling_args()` previously always emitted `top_p = 1.0`.
Putting `top_p` in `extra_body` did not solve this: renderer clients prioritize
the explicit sampling fields. The fork adds a validated `(0, 1]` field, forwards
its value, and rejects the shadowed `extra_body.top_p` placement. The default
remains `1.0`. Config tests cover forwarding, defaults, invalid thresholds, and
misplaced overrides; `skills/configs/SKILL.md` documents the supported syntax.

## Consolidated upstream review

The aggregate review branch, `fix/primebeaker-consolidated-fixes`, preserves all
eight commits through `dc1972200` and adds this documentation. It does not
rebase, merge, or fast-forward the working `fix/training-top-p` branch. The
runtime source and dependency pins remain those used by PrimeBeaker's maintained
rebuild source.

The changes span the `prime-rl` runtime, the `prime-rl-configs` package, and the
`renderers` Git submodule. Neither `deps/verifiers` nor
`deps/research-environments` changes relative to `2ffe374e0`. The renderer fix
is an external dependency commit, not source vendored into Prime-RL; reviewers
must also inspect `goncalorafaria/renderers@a2f6949`.

At the upstream inspection on 2026-09-11, `PrimeIntellect-ai/prime-rl/main` was
`43b4e2bd1334267208fb708e830b415a24f94acc`, 176 commits after the fork base.
The aggregate branch has merge conflicts with that revision and is intended
as a draft for reviewing the complete historical patch series.

- Upstream PR #3431 (`84e7312f3`) already adds top-p/top-k training sampling
  with native sampling replay. The fork's top-p commit is preserved to explain
  its deployed behavior, not as a claim that current upstream lacks top-p.
- Upstream PRs #3285 and #3309 introduce adaptive concurrency, including online
  evaluation. The fork's fixed train/eval limits need reconciliation with that
  implementation before a future upstream merge.
- Upstream moved or removed client, optimizer, and transport implementation
  files touched here. Resolving those changes is a separate porting task;
  this review branch intentionally preserves the working version.
- The renderer fork pin must be reconciled with upstream's current renderer
  revision before merging. The container overlay also assumes a prebuilt
  `prime-rl-v071dev83-qwen35-renderer-eval:latest` base; it is not a standalone
  image build recipe.

PrimeBeaker's image catalog distinguishes immutable deployed images from the
maintained rebuild source: those images predate the cleaned-up Git history.
The cataloged commit should not be described as the proven embedded revision
of the historical images.
