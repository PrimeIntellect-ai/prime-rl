# Score-centering experiment results

Status at 2026-09-19 13:10 UTC: runtime recovery passed. The paired step-50 restart is awaiting a four-node allocation; full restore pending. No collapse observed.

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

Both arms completed fifty finite updates before the gateway incident.
No nonfinite metrics or numerical warnings were detected in the exported metrics and logs.
Matched curves stop both arms at update 50, including mean and tail mismatch diagnostics.
See `results/main-api4-step50`; earlier snapshots remain under `main-api4-step10` and `main-api4-step25`.

| Measure | IPO | Centered |
| --- | ---: | ---: |
| Mean training success, updates 1–25 | 58.5% | 59.1% |
| Mean training success, updates 26–50 | 55.6% | 59.5% |
| Mean gradient norm, updates 26–50 | 0.119 | 0.073 |
| Largest gradient norm, updates 1–50 | 0.530 | 0.272 |
| Mean mismatch KL, updates 26–50 | 0.00435 | 0.00426 |
| Largest per-token mismatch KL estimate | 189.65 | 202.13 |
| Received training output tokens at update 50 | 42.78 million | 44.79 million |

Training success is the mean of the logged per-step arrival metrics.
Async completion and errors change the task mix. These figures are not a paired held-out estimate.
The token budgets include completed training rollouts, including unused arrivals.
Both arms encounter large rare-token mismatch despite low means.
IPO gradient peaks at updates 39, 43, and 45 subsided on the following updates.
Centered has smaller gradient peaks and a flatter reward window so far.
Neither arm meets the predefined numerical-instability or reward-collapse criterion.
These are exploratory observations from one seed, not evidence of the requested robustness separation.

Both step-50 checkpoints contain readable DCP metadata and orchestrator progress.
Each trainer checkpoint occupies 341.24 GiB. All 11,475 storage references fit their shard files.
This validates metadata and file extents; no full restore was performed.
Training continued after each save. Checks are in `results/monitor/checkpoints-step50.json`.
The same checks passed for step 25.

Both inference pools accept refreshed weights. API4 has avoided the earlier unavailable-worker storm so far.
Intermittent health-check misses remain. Most trainer time is spent waiting for rollouts.
Ten-minute error rates fluctuate as long episodes finish; inspected bursts mainly hit the 900-second deadline.
The step-50 scheduled TB2 evaluations overlapped the gateway incident and shutdown. Treat them as affected data.

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

## Gateway incident and recovery

At 12:40 UTC, both arms began receiving nginx 502 responses inside their VM harnesses.
The local inference pools remained available. No trainer numerical failure was observed.
Both orchestrators completed forced cleanup after SIGINT at 12:46 UTC.
SLURM allocations 891 and 893 were then released.
Both step-50 checkpoints were preserved with hard links before stopping.
IPO stopped after update 54; centered stopped after update 52.
Updates after step 50 and the interrupted step-50 evaluations remain in the original run directories.
They will stay separate from the checkpoint continuation.

A fresh tunnel probe returned nine successful requests and three timeouts in twelve requests.
Both persistent and fresh connections timed out. This does not identify the service root cause.
A hosted-model `uv run eval` also reproduced the 502 inside a fresh Lego VM.
Another Lego episode completed and scored. Recovery was not yet validated at 13:00 UTC.
A second eval finished at13:09 UTC with all four episodes scored and no errors.
Two subsequent tunnel probes each passed12/12 public requests.
Submitted paired step-50 continuations as IPO908 and centered909 at13:10 UTC.
Full checkpoint restore and next optimizer updates remain to be verified.
Only IPO908 received nodes; centered909 remained pending for resources.
Stopped both before episode collection to preserve concurrent execution.
The replacement `resume-pair.sbatch` reserves four nodes atomically.
Each arm runs on its own two-node subset, with explicit scoping for cleanup and startup.
A stub launch verified both disjoint scopes. Scientific settings stay fixed.
These infrastructure failures do not meet the experiment's policy-collapse criterion.
Resume both arms from their preserved step-50 states after runtime recovery.
Use new run directories to preserve the affected records and repeat the step-50 online evaluation.
Keep the same one-trainer-plus-one-inference allocation per arm.

`lineage.json` joins each parent through update 50 to its planned continuation.
Run `analyze.py --lineage experiments/score-centering/lineage.json --output <directory>` for joined curves.
Parent records stop at the checkpoint update's final trainer metric timestamp.
Later evaluation records and discarded updates are excluded from those curves.
The continuation adds received-token and active-run-time budgets to the parent boundary.
These budgets exclude recovery downtime and discarded work; original logs preserve those costs.
The joined first-50 export exactly matches the prior gradients, mismatch statistics, and budgets.
Boundary and budget checks passed on a small synthetic continuation.
Affected eval exports are under `results/gateway-incident/evals`.
Centered has only 52 of 64 step-50 eval records; twelve were interrupted before recording.
Do not use the export's recorded-episode bounds as bounds across all 64 tasks.

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
The API-worker change was unvalidated at that cancellation. The current pair has now completed fifty updates under load.

## Pending

Collect the main learning curves and held-out results at matched updates and token budgets.
Inspect trace failures before classifying instability.
If a separation appears, repeat with a second seed and a matched BF16 control.
Report a negative or inconclusive result if the requested separation does not occur.
