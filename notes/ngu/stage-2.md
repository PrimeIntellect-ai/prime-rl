# Stage 2: one SWE baseline and one NGU run

Use the established GLM Air SWE infrastructure recipe for exactly two large training runs. No calibration sweep, separate validation campaign, model scaling stage, or hyperparameter search.

The current frozen-task recipe is [configs/experiments/ngu](../../configs/experiments/ngu/README.md): train on the profiled 1,000 tasks; evaluate on full SWE-Bench Verified plus 500 disjoint SWE-rebench tasks. Its TOMLs supersede the provisional dataset/config choices below.

## Shared setup

Base recipe: `examples/advanced/glm-4.5-air/swe.toml` (the TOML is authoritative; its README describes an older topology).

- Model: `PrimeIntellect/GLM-4.5-Air-Scaleswe`, same pinned starting weights and tokenizer for both arms. Per the user, this checkpoint already trained 1,000 steps on ScaleSWE. Start fresh optimizer/scheduler state for both arms; do not resume the old ScaleSWE run.
- Training (provisional recommendation): `swerebench-v2`, verified train split (the profiling snapshot contains 6,272 tasks), bash harness, fresh Prime sandbox per episode. See [available tasksets](tasksets.md) for alternatives; no ScaleSWE training source.
- Evaluation: full `swebench-verified`, same bash harness and timeout, at initialization and every 20 optimizer steps.
- Six H200 nodes/run, assuming eight GPUs/node: two trainer nodes and four single-node inference replicas. Run the arms sequentially on the same allocation. If only four nodes are allocated, use two trainer + two inference nodes for **both** arms. Do not assume this MoE trainer fits in a one-node allocation.
- 131072 context, TP=8 with expert parallelism at inference, trainer CP=4/ulysses, custom model, router replay, activation checkpointing/offloading, Muon at LR 1e-6, default IPO loss made explicit.
- Nominal batch target 256 training traces and K=16 per sampling round. No length penalty in either arm; only binary solved reward.
- Shared default sampling settings. Preserve the recipe's uncapped per-turn output setting (EOS/context/harness limits apply); the 131k context is not a per-episode total token budget.
- Preserve global maximum trained staleness of 32 policy versions, the existing adaptive concurrency controller, and source order. Record all resolved config/defaults and dependency/model/data revisions.

`swerebench-v2`'s pinned taskset scores 1 only when all expected FAIL_TO_PASS and PASS_TO_PASS tests pass. This is compatible with binary NGU even though episodes contain multiple tool calls. A retry is a new episode from the same clean task state, never a continuation of the previous patch/sandbox. Check overlap against both the checkpoint’s prior ScaleSWE tasks and our evaluation tasks, using normalized repository + issue/PR/commit identities, not just dataset-local IDs. Retain the same filtered manifest in both arms. This check has not yet been performed.

## Only two arms

| Arm | Base round size | NGU continuation | Historical baseline | Advantage handling |
|---|---:|---:|---|---|
| Baseline | 16 | none | current static group | Centered GRPO |
| NGU | 16 | .875 | all valid attempts in the visit | Positive anchoring |

Use p=.875 because it is the highest-scoring aggregate setting in the paper's compute-matched larger math experiment; use its K=16 and the four-step history recommendation. This is a concrete transfer choice for binary SWE, not a claim that the paper established a universal optimum. Its partial-credit coding experiment used p=.95; we will not run a sweep between them. The paper's best fixed-K math score used K=32, but our baseline keeps the established SWE K=16 to isolate NGU in our actual recipe.

Store old valid reward counts after payload expiration. NGU payload history is limited to four policy versions, with the existing global staleness limit still enforced. Record the exact age convention; paper T=4 and our inclusive policy-age bound are not automatically identical.

### Binary reward only

Both arms use raw solved reward without length penalties or other shaping. All-success and all-failure static groups have zero GRPO signal. NGU retries all-failure visits probabilistically and filters all-success visits; raw historical counts determine the anchored advantages. Stage 1 no longer needs a length-shaping extension.

Apply the same cohort-preserving batch mode to both arms; report actual batch sizes/overshoot. This is a stage-1 prerequisite, since silently splitting an anchored cohort across updates is undesirable.

## Budget and comparison

Proposed duration: **24 allocated hours per run** on six nodes, i.e. 1,152 H200-hours per arm and 2,304 total. This is a budget choice, not a throughput forecast. No training has been launched. Use the same hardware topology, model, seed settings and limits in both arms. Full run-level seed plumbing is still needed; an inference seed alone does not make async scheduling deterministic.

Compare at equal allocated GPU-hours. Count online evals, filtered/retried/expired generations and trainer idle time. Also report sandbox-hours and verifier/runtime failures, since SWE compute includes substantial CPU/sandbox work. Plot step-indexed curves only as secondary diagnostics.

The TOMLs use max_steps=10000 as a guard. Arrange a graceful budget stop with a final checkpoint; there is no current TOML GPU-hour stopper. Compare the last checkpoint at or before the budget, not the best eval checkpoint. Save enough periodic checkpoints to recover matched-time comparisons; the draft retains all checkpoints, so provision storage before launch.

## Evaluation and metrics

Primary: full SWE-Bench Verified resolved rate/pass@1 at equal H200-hours. Online eval uses one attempt/task at step 0 and every 20 steps, matching the existing operational cadence. For a less noisy final estimate, evaluate each final checkpoint with four independent attempts per task under identical settings and report mean success frequency (pass@1); also report pass@4 separately. This is evaluation of the same two runs, not a separate training phase.

Difficulty profiling is now explicitly requested: before the two training runs, deploy four ETP8 replicas of the starting checkpoint and measure avg@8 on a frozen random 1,000-task SWE-rebench sample, with a one-hour solve budget. Use easy=6–8 solves, medium=3–5, hard=1–2, extra-hard=0. The complete single-job deployment and generated four-taskset overlay are in [configs/experiments/ngu](../../configs/experiments/ngu/README.md). Eval and splitting run on the allocated inference master, not a login node. Use these fixed subsets to observe training curves; retain SWE-Bench Verified for generalization.

Monitor:

| Question | Metrics |
|---|---|
| Does NGU solve more SWE tasks? | Overall resolved rate, fixed difficulty-subset resolved rates, per-task transitions, pass@1 vs GPU-hours |
| Is the effect useful per compute? | Allocated GPU-hours, generated input/output tokens, sandbox-hours, trained/generated ratio, area under learning curve |
| Is compute moving to persistent failures? | Fresh visits/retry rounds, retries per visit p50/p95/max, give-up fraction, first-success cost, accepted raw positive/negative counts |
| Is history working correctly? | Lifetime count C and reward sum S; retained cohort size; payload ages; expired payload count; no-balancing-negative drops; advantage sums/magnitudes |
| Does the pipeline remain healthy? | Trainer wait/utilization, inference throughput, active visits/buffer memory, queue age, errors/cancellations, truncations, output length/turns, ratio masking and gradient norms |

Count each physical episode once even if its reward contributes later. Keep raw generation metrics separate from accepted-training metrics; success rates among NGU-accepted groups are selection-biased.

Decision: prefer NGU if it gives a practically useful final resolved-rate gain and/or a better compute learning curve, without unacceptable instability or sandbox cost. Report paired task-level confidence intervals and the fixed difficulty subsets, but do not infer seed-to-seed reliability from one run per arm. No additional sweep is part of this proposal.

## Artifacts and readiness

- `swe-baseline.toml`: complete baseline draft derived from the existing recipe.
- `swe-ngu.toml`: matching complete draft; NGU fields are proposed stage-1 schema.
- `ngu-overlay.toml`: same NGU settings as a reusable overlay; use a distinct run name and runtime label if composing it over the baseline.

Removed the math configs and separate GSM8K pilot from this plan. Verification of stage-1 logic and a capped launch sanity check remain engineering checks, not extra scientific comparison runs.

The copied recipes omit automatic sandbox deletion and automatic resume, and give each arm distinct run names and use automatic runtime labels. They preserve numerical dtype defaults. Submodules/dependencies are currently absent, so only TOML syntax has been checked. Before launch: implement NGU plus batching/seed support; resolve both configs; pin task/model revisions; verify binary rewards and clean retry state; choose the actual shared output path and H200 SLURM partition. Do not launch either run during this preparation task.

Model access note: the supplied Hugging Face repository is configured exactly as requested. An unauthenticated model-card request returned an authentication error; its revision and published metadata have not been independently verified. The 1,000-step provenance is user-provided.
