# Score-centering experiment results

Status at 2026-09-19 09:04 UTC: the corrected API4 pair is starting. No robustness conclusion yet.

## Main comparison

- Model: Qwen3-30B-A3B-Instruct-2507, pinned snapshot in `model.json`.
- Source: experimental commit `9a282107e`, with main `a1822f7a0` merged.
- Baseline: current-main IPO, eps 0.3; SLURM 892, `ipo-1plus1-api4-seed42-r2`.
- Treatment: published standalone score centering; SLURM 891, `sc-1plus1-api4-seed42-r2`.
- Each arm: one H200 trainer node plus one inference node, running concurrently.
- Concurrency: adaptive, initial 512 episodes and maximum 2048 per arm.
- Training: Terminal Lego, uniform task draws with seed 42, batch 128, group 8, 400 updates.
- Evaluation: Terminal Bench 2, 64 tasks initially, every 25 updates, and at completion.
- Inference: INT8 expert weights, FP8 dense weights, FP8 KV cache.
- No training optimization or reduction precision settings changed.

The primary contrast changes importance weighting and masking as well as centering.
It tests the published estimator against IPO. It does not isolate centering alone.
See `PROTOCOL.md` for the estimator and fixed decision rules.

## Validation and smoke evidence

Prime VM recovery checks used `uv run eval`: Lego 2/2 solved and TB2 1/2 solved.
These runtime checks used a hosted model, not the Qwen model under training.
Math, gradient, sampler-head transport, and existing CUDA loss checks passed.
After merging the latest main, 155 parser tests passed.

Both concurrent smoke pairs completed three finite optimizer updates per arm.
They also validated checkpoint saves and quantized weight refresh.

| Cache | IPO mismatch KL at update 1 | SC mismatch KL at update 1 | Serving observation |
| --- | ---: | ---: | --- |
| INT4 | 2.124 | 0.758 | Occasional NaN responses before any optimizer update |
| FP8 | 0.0071 | 0.0062 | No observed NaN responses in the diagnostic pair |

These are small smoke batches, not matched-token estimates or evidence of a treatment effect.
Both arms stayed finite in both pairs. The INT4 errors do not demonstrate training collapse.
FP8 provides a cleaner comparison but substantially reduces the measured mismatch.
A stable FP8 result cannot rule out benefits under stronger mismatch.

The INT4 pair used jobs 849/850; the FP8 pair used jobs 857/856.
Machine-readable metrics, plots, and task-level evaluation exports are under `results/smoke4` and `results/smoke5`.
Evaluation exports report failed episodes and bounds for missing rewards.
Malformed tool-call JSON is distinct from a numerical or VM-service failure.

## Superseded early pair

Jobs 861/860 ran with three inference nodes per arm and a 192-episode cap.
They completed six IPO updates and five centered updates; both remained finite.
The trainers spent most wall time waiting for rollouts; sampled KV utilization was below 1%.
At the user's request, both jobs stopped and the replacement pair starts from the same base weights.
The old logs and metrics remain separate under `results/initial-4node-attempt`.
These early updates do not establish a robustness difference.

## Cancelled 1+1 attempt

Jobs868/867 reached roughly 512 inflight episodes with a 2048 cap.
Both encountered repeated router health-check failures before any optimizer update.
Both orchestrators completed forced cleanup before their allocations were cancelled.
A matched restart with four API workers per engine passed config validation.
The user cancelled the experiment before those replacement launchers submitted jobs.
The API-worker change has not been validated under rollout load.

## Pending

Collect the main learning curves and held-out results at matched updates and token budgets.
Inspect trace failures before classifying instability.
If a separation appears, repeat with a second seed and a matched BF16 control.
Report a negative or inconclusive result if the requested separation does not occur.
