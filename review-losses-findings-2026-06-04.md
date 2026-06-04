# Review Findings: Losses Branch

Comparison base: `origin/main...HEAD`

I used `origin/main` as the comparison point because local `main` was behind the remote branch during review. No source files were changed as part of the review.

## Findings

### P1: Echo-only envs can be filtered out before echo loss is preserved

Post filters run first in `src/prime_rl/orchestrator/train_sink.py:331`, while echo-only primary masking happens later in `src/prime_rl/orchestrator/train_sink.py:351`. The default config enforces zero-advantage filtering in `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:604`, and `src/prime_rl/orchestrator/filters.py:108` drops `advantage == 0.0`.

For documented `enabled_losses = ["echo"]`, common cases like `group_size=1` can drop all samples even though echo supervision does not depend on advantage.

### P1: `enabled_losses` accepts `sft`/`opd`, but runtime does not apply them per env

The schema exposes those names in `packages/prime-rl-configs/src/prime_rl/configs/losses.py:145` and env selection in `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:218`, but runtime selection only gates echo and RL/custom primary loss in `src/prime_rl/orchestrator/train_sink.py:176`. The trainer still chooses SFT/OPD/RL from `training_mode` in `src/prime_rl/trainer/rl/loss.py:323`.

An RL env with `enabled_losses = ["sft"]` can pass validation, disable RL primary loss, and still not get SFT loss.

### P1: The public `[trainer.loss]` config surface was removed without a migration path

The new field is `trainer.losses` in `packages/prime-rl-configs/src/prime_rl/configs/trainer.py:481`, while old `type = "default"` is no longer accepted and custom loss terms now require `name` in `packages/prime-rl-configs/src/prime_rl/configs/losses.py:69`. Existing checked-in configs still use `[trainer.loss]`, for example `configs/private/int5/ablations/ipo/dppo.toml:9`.

With strict config validation, those configs now fail instead of receiving a compatibility shim or targeted deprecation error.

### P2: Backfilled rollouts lose prompt-role echo attribution

Echo annotations are built after backfill in `src/prime_rl/orchestrator/train_sink.py:221`, but the renderer backfill path in `src/prime_rl/orchestrator/trajectories.py:165` produces tokenized steps without `prompt_attribution`. Prompt-role echo only works when that attribution exists in `src/prime_rl/orchestrator/echo.py:68`.

Configured system/user/tool or prompt-assistant echo silently no-ops for backfilled rollouts.

### P2: `loss_overrides` validates override keys but not merged override values

The field is accepted in `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:222`, key validation lives in `packages/prime-rl-configs/src/prime_rl/configs/losses.py:181`, and the merged `EchoLossConfig` is only constructed later during rollout handling in `src/prime_rl/orchestrator/train_sink.py:185`.

Bad alpha types or nested shapes can pass config dry-runs and fail only once data flows.

### P2: Algorithm docs still describe the wrong DPPO clipping objective

`docs/algorithms.md:48` describes `min(ratio, delta)` style clipping, but implementation uses probability-difference masks conditioned on advantage sign in `src/prime_rl/trainer/rl/loss.py:172`.

This misstates how `dppo_mask_low/high` behave.

### P3: The SFT renderer warning can be skipped

The prompt-role echo warning checks `renderer is None` in `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:541`, but the later SFT validator forces `renderer = None` in `packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:801`.

SFT configs with prompt-role echo can become silent no-ops without the intended warning.

## Verification Notes

Test execution was blocked locally. `uv run` rejects this macOS platform for the Linux-only lockfile, and `uv run --no-sync` then fails on missing local dependencies.
