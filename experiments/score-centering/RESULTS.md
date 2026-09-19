# Score-centering experiment results

Status at 2026-09-19 10:56 UTC: both arms passed twenty-five finite updates and saved checkpoints. No robustness separation observed.

## Main comparison

- Model: Qwen3-30B-A3B-Instruct-2507, pinned snapshot in `model.json`.
- Source: SC `9a282107e`, IPO `8e2282999`; main `a1822f7a0` is merged. IPO adds dependency-sync suppression and records only.
- Baseline: current-main IPO, eps 0.3; SLURM 893, `ipo-1plus1-api4-seed42-r3`.
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

## Early current-run observations

At 10:56 UTC, both arms passed twenty-five finite optimizer updates.
IPO step 25: gradient norm 0.1005, mismatch KL 0.0044.
Centered step 25: gradient norm 0.0874, mismatch KL 0.0047.
Across updates 1–25, mean mismatch KL is 0.00430 in both arms.
Received training output tokens at update 25: IPO 21.03 million; centered 22.34 million.
These token budgets include completed training rollouts, including unused arrivals.
The exported metrics and logs contain no detected nonfinite values or numerical warnings.
Matched curves stop both arms at update 25 and are in `results/main-api4-step25`.
The earlier snapshot remains in `results/main-api4-step10`.

Both step-25 checkpoints contain readable DCP metadata and orchestrator progress.
Each trainer checkpoint occupies 341.24 GiB. All 11,475 storage references fit their shard files.
This validates metadata and file extents; no full restore was performed.
The save took about 2.5 minutes per arm. Training continued after each save.
Checkpoint checks are in `results/monitor/checkpoints-step25.json`.

Both inference pools accept refreshed weights. No stability separation is established.
Warm steps have taken roughly 1.5–6 minutes; most time is spent waiting for rollouts.
Across updates 2–9, rollout waits account for 77% of IPO time and 82% of centered time.
API4 has avoided the earlier unhealthy-worker request storm so far. Intermittent health-check misses remain.
Ten-minute training error rates fluctuate as long episodes finish; inspected bursts mainly hit the 900-second deadline.
The step-25 scheduled TB2 evaluations finished at about 11:09 UTC.

Initial TB2 evaluation solved 2/64 IPO tasks and 1/64 centered tasks.
IPO had 24 failed episodes; centered had 16. These sparse scores have substantial missingness.
Five IPO episodes and six centered episodes spanned at least one live policy refresh.
These online results are not frozen-checkpoint measurements.
See `results/main-api4-initial-eval` for scores, errors, policy spans, and missing-reward bounds.
The fixed-checkpoint Lego audit remains required before interpreting capability retention.

The step-25 evaluation used the same 64 tasks as the initial evaluation.

| Arm | Solved | Scored | Failed episodes | Episodes spanning a policy refresh |
| --- | ---: | ---: | ---: | ---: |
| IPO | 3/64 | 42 | 22 | 51 |
| Centered | 2/64 | 44 | 20 | 46 |

Among scored tasks, IPO success is 7.1% (Wilson 95%: 2.5–19.0%).
Centered success is 4.5% (Wilson 95%: 1.3–15.1%).
Assigning missing rewards either zero or one gives all-task bounds of 4.7–39.1% for IPO and 3.1–34.4% for centered.
These bounds are not confidence intervals. The sparse scores and missingness do not establish an advantage.
IPO errors comprise 13 provider errors and 9 harness errors; centered has 15 and 5.
Most provider errors concern malformed tool-call JSON. Inspected harness errors include rollout timeouts.
Recorded policy spans cover versions 25–28 for IPO and 25–29 for centered.
These evaluations measure the live training pools, not frozen step-25 checkpoints.
Task-level exports and uncertainty are in `results/main-api4-eval25`.

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
The API-worker change was unvalidated at that cancellation. The current pair has now completed twenty-five updates under load.

## Pending

Collect the main learning curves and held-out results at matched updates and token budgets.
Inspect trace failures before classifying instability.
If a separation appears, repeat with a second seed and a matched BF16 control.
Report a negative or inconclusive result if the requested separation does not occur.
