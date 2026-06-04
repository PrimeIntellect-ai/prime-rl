# Review: `sebastian/losses-2026-06-04`

Scope: read-only review of changes introduced by `main...HEAD`, focused on the new composable losses and echo overlay implementation.

## Findings

### P1: Echo CE uses rollout sampling temperatures

References:
- `src/prime_rl/trainer/rl/train.py:470`
- `src/prime_rl/trainer/rl/train.py:491`

Echo CE reuses `out["logprobs"]`, which are computed with rollout temperatures. Prompt/context tokens were previously masked, but echo now trains them, so `temperature != 1` optimizes `log_softmax(logits / T)` instead of true NLL.

Suggested fix: compute separate temperature-1 logprobs for echo, or explicitly make echo's objective temperature-scaled and document/test it.

### P1: `enabled_losses` does not control the primary loss

References:
- `src/prime_rl/orchestrator/train_sink.py:262`
- `src/prime_rl/trainer/rl/loss.py:308`

Samples always get the global `training_mode`, and if `losses` has no `rl`/`custom`, the trainer silently creates default RL. As a result, `enabled_losses = ["echo"]`, `enabled_losses = []`, or echo-only `losses` still trains RL.

Suggested fix: wire enabled primary terms per sample, or reject configs that omit the active primary until that behavior is supported.

### P1: Multi-run loss configs can diverge silently

Reference:
- `src/prime_rl/trainer/rl/train.py:155`

The trainer builds `loss_fns` once from startup `config.losses`, while each run's orchestrator config also carries `losses`. Per-run primary loss params and custom imports are not validated against or used by the trainer.

Suggested fix: add a `MultiRunManager` validation hook requiring `orch_config.losses == trainer.config.losses`, or route loss terms per run/sample.

### P1: Legacy `[trainer.loss]` configs now fail validation

References:
- `packages/prime-rl-configs/src/prime_rl/configs/trainer.py:481`
- `configs/private/int5/ablations/ipo/dppo.toml:9`

`trainer.loss` was removed under strict config validation, but checked-in configs still use `[trainer.loss]`. These now fail validation instead of migrating.

Suggested fix: add a `mode="before"` legacy translator with a warning, or update all configs/docs/changelog as an intentional break.

### P2: Multiple primary terms validate but only the first is used

Reference:
- `src/prime_rl/trainer/rl/loss.py:300`

Configs can include more than one primary term (`rl`/`custom`), but the trainer uses only the first and silently ignores the rest. That conflicts with the named-term surface and per-env selection.

Suggested fix: validate exactly one primary term globally until per-sample named primary routing exists.

### P2: `enabled_losses=None` skips echo validation

References:
- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:533`
- `src/prime_rl/orchestrator/train_sink.py:176`

`enabled_losses=None` is documented as "all terms", but validation skips it. Two echo terms pass config validation and fail later in `_resolve_echo`.

Suggested fix: validate `None` as the full loss list for each env.

### P2: `loss_overrides` is not validated

Reference:
- `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:222`

`loss_overrides` accepts arbitrary keys and values. Typos or overrides for non-enabled/non-echo terms silently do nothing, and bad echo overrides surface only at rollout time.

Suggested fix: validate override keys against `losses`, and reject non-echo overrides for now.

### P2: Rendererless token backfill cannot support prompt-role echo

References:
- `src/prime_rl/orchestrator/trajectories.py:117`
- `src/prime_rl/orchestrator/echo.py:68`

Rendererless backfill emits no `prompt_attribution`, but prompt-role echo requires it. In SFT/MITO paths, `system`/`user`/`tool` echo can silently become a no-op.

Suggested fix: reject those configs without attribution, or add attribution to fallback tokenization.

### P2: Token export hides echo-supervised tokens

Reference:
- `src/prime_rl/trainer/rl/token_export.py:59`

Token export still gates records on `loss_mask` and exports no `echo_mask`/`echo_weight`. Prompt-only echo sequences can be skipped, and mixed exports hide which prompt tokens trained.

Suggested fix: include echo columns and export when `loss_mask OR echo_mask` is true.

### P3: Echo alpha accepts negative weights

Reference:
- `packages/prime-rl-configs/src/prime_rl/configs/losses.py:28`

Echo `alpha` allows negative finite values, turning CE into an objective that lowers selected token probability.

Suggested fix: add `ge=0` to all role alpha fields unless negative echo is intentional.

## Notes

No files were changed during the review itself. The review used local diff inspection plus four focused subagent passes covering loss/training, configs, orchestrator/runtime, and tests/docs.
