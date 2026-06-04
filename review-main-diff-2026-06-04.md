# Review: changes vs main

Branch reviewed: `sebastian/losses-2026-06-04`
Base: `main` via `git diff main...HEAD`
Diff size: 29 files, 1332 insertions, 106 deletions

This was a read-only review of the branch behavior. I did not change implementation files.

## Findings

### P1: Legacy `trainer.loss` configs now fail validation

The branch removes `TrainerConfig.loss` and exposes only `TrainerConfig.losses` at `packages/prime-rl-configs/src/prime_rl/configs/trainer.py:481`. The config base class forbids extra fields (`deps/pydantic-config/src/pydantic_config/cli.py:94`), so existing checked-in configs that still use `[loss]` or `[trainer.loss]` now hard-fail instead of being migrated. Old custom configs also lack the now-required `CustomLossTermConfig.name` (`packages/prime-rl-configs/src/prime_rl/configs/losses.py:119`), and old `type = "default"` no longer maps to the new `type = "rl"` discriminator (`packages/prime-rl-configs/src/prime_rl/configs/losses.py:68`).

Examples still present in the repo:

- `configs/private/int4/ablation/train.toml:27`
- `configs/private/int4/prod/train.toml:22`
- `configs/private/int3/ablation/train.toml:22`
- `configs/private/int4/debug_train/train.toml:24`
- `configs/private/int3/prod/train.toml:22`
- `configs/private/int5/ablations/ipo/dppo.toml:9`
- `configs/private/int5/ablations/ipo/dppo_kl.toml:9`
- `configs/private/int5/ablations/ipo/ipo_no_kl.toml:9`
- `configs/private/int5/ablations/ipo/ipo_plus.toml:9`
- `configs/private/int5/ablations/ipo/ipo_plus_plus.toml:9`
- `configs/private/int5/ablations/ipo/ipo_wide_mask.toml:9`
- `configs/private/int5/ablations/ipo/kimi_k2.toml:9`

Recommendation: either add a targeted before-validator/deprecation migration from legacy `loss` to singleton `losses` entries, or migrate the checked-in configs and add an explicit compatibility-breaking test/error.

### P1: Pre-batch zero-advantage filtering ignores the primary-loss gate

`ZeroAdvantageFilter.primary_active` defaults to always active (`src/prime_rl/orchestrator/filters.py:117`). The orchestrator rewires that gate only for `post_filters` at `src/prime_rl/orchestrator/orchestrator.py:397`, but `pre_filters` run earlier at `src/prime_rl/orchestrator/train_sink.py:280`.

For an env with `enabled_losses = ["echo"]` and an enforcing pre-batch zero-advantage filter, zero-advantage rollouts are still dropped before echo training can happen. In the worst case, echo-only singleton groups can starve training indefinitely.

Recommendation: apply the same `_primary_enabled` gate to zero-advantage filters in both pre- and post-batch filter lists.

### P2: Envs can silently produce zero effective loss

`enabled_losses=[]` passes validation because `check_enabled_losses` only rejects unknown names and multiple echo terms (`packages/prime-rl-configs/src/prime_rl/configs/losses.py:156`), and `OrchestratorConfig.validate_enabled_losses` accepts that resolved list at `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:535`.

At runtime, `TrainSink.process_batch` disables the primary by zeroing `sample.completion_mask` (`src/prime_rl/orchestrator/train_sink.py:354`) but still appends the sample and counts every unfiltered rollout as trainable (`src/prime_rl/orchestrator/train_sink.py:356`, `src/prime_rl/orchestrator/train_sink.py:362`). This can also happen with prompt-role echo when `renderer=None`, or when an echo filter masks everything. The orchestrator then advances and ships a batch that may have no primary tokens and no echo tokens.

Recommendation: require every train env's resolved enabled set to include at least one effective term, and/or compute `n_trainable` from actual loss-bearing primary or echo masks rather than rollout filter status.

### P2: `loss_overrides` payloads are not validated during dry-run

`TrainEnvConfig.loss_overrides` is typed as an arbitrary `dict[str, dict]` (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:222`). Config validation only checks that override keys refer to known echo terms (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:537`; `packages/prime-rl-configs/src/prime_rl/configs/losses.py:166`). Invalid override shapes, unknown echo fields, or non-float alphas are accepted by `RLConfig` and fail later when `_resolve_echo` deep-merges and constructs `EchoLossConfig` at `src/prime_rl/orchestrator/train_sink.py:186`.

Recommendation: validate each override during `OrchestratorConfig.validate_enabled_losses` by deep-merging into the referenced echo config and constructing `EchoLossConfig` there.

### P2: Echo metrics are biased by sequence packing

When echo and non-echo samples share a packed microbatch, non-echo samples get false echo masks backfilled at `src/prime_rl/trainer/batch.py:221`. `compute_loss` still runs the echo core for every split in the packed sample (`src/prime_rl/trainer/rl/loss.py:404`), and empty echo masks report `echo_nll = 0` via `_safe_mean` (`src/prime_rl/trainer/rl/loss.py:281`).

That makes `echo_nll/mean` depend on packing composition: adding non-echo splits lowers the reported mean even though no echo tokens changed.

Recommendation: aggregate echo NLL as numerator/denominator over echo tokens, or skip echo metric entries when `term_mask.any()` is false.

### P2: Echo-only supervised tokens are invisible to trainer entropy/env metrics

The trainer still indexes entropy and env attribution only by the primary `loss_mask` (`src/prime_rl/trainer/rl/train.py:524`, `src/prime_rl/trainer/rl/train.py:529`). For primary-disabled echo envs, training can apply gradients through `echo_mask`, but step logs and env-level entropy metrics can be empty or `nan`.

Recommendation: either add separate echo-token metric paths or define trainable-token metrics over `loss_mask | echo_mask`.

### P3: Token export schema version was not bumped

Token export adds `echo_mask` and `echo_weight` columns at `src/prime_rl/trainer/rl/token_export.py:133`, but `SCHEMA_VERSION` remains `1` at `src/prime_rl/trainer/rl/token_export.py:15`.

Recommendation: bump the schema version or document that v1 consumers must tolerate these additional fields.

## Test Gaps

- The pre-batch zero-advantage gate should have an integration-style test that constructs filters through orchestrator setup or equivalent wiring, not just a direct `ZeroAdvantageFilter(primary_active=lambda ...)` unit test.
- Multi-step echo alignment through `interleave_rollout` is untested. The new append/slice path is at `src/prime_rl/orchestrator/trajectories.py:391`; add a merged two-step rollout test with `EchoAnnotations` and assert `sample.echo_alpha` remains aligned with `prompt_ids + completion_ids`.
- Add a `TrainSink.process_batch` test for `enabled_losses=["echo"]`: shipped samples should have `completion_mask` all false while preserving `echo_alpha`.
- Add config tests for malformed `loss_overrides` and `enabled_losses=[]`.

## Pre-existing/stale docs noticed

These appear to predate this branch, so I am not counting them as regressions from `main`, but the branch touches nearby config docs and may be a good time to fix them:

- `docs/algorithms.md:46` still describes a clipped importance-ratio objective with `min(ratio, delta)`, while the implementation masks by probability difference using `dppo_mask_low/high`.
- `docs/algorithms.md:142` documents `[orchestrator.length_penalty]`, but the current schema places length penalties under `orchestrator.advantage.length_penalty`.
- `docs/algorithms.md:184` documents `[[orchestrator.filters]]`, but the schema uses `pre_batch_filters` and `post_batch_filters`.
- `docs/algorithms.md:205` documents `[orchestrator.buffer]` difficulty pools, which I did not find in the current config schema.
- `skills/configs/SKILL.md:17` uses `--trainer.lr`; current optimizer learning rate is under `--trainer.optim.lr`.
- `skills/configs/SKILL.md:44` and `skills/configs/SKILL.md:48` use `[[orchestrator.env]]` / `--env.0.id`; current train/eval envs live under `orchestrator.train.env` / `orchestrator.eval.env`.

## Verification

- Ran `git diff --check main...HEAD`: clean.
- Did not run `vfcheck` because the request was review-only and `vfcheck` may auto-format.
- Tried a lightweight `uv run python` config import check, but this machine cannot use the repo lockfile because the lockfile is Linux-only: `The current Python platform is not compatible with the lockfile's supported environments`.
- Five subagents reviewed separate slices: config DSL, trainer loss/training integration, orchestrator echo/filtering, packaging/docs/skills, and regression test coverage.
