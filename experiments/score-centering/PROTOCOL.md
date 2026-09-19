# Score centering on agentic terminal RL

## Question

Does score centering improve robustness to quantized inference in current prime-rl?
Compare current-main IPO (`eps=0.3`) against the paper's standalone score-centering loss.
Treat baseline collapse as a hypothesis, not a required outcome.

## Sources and revisions

- Paper: https://arxiv.org/html/2609.20807v1
- Reference code: https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321
- Baseline main: `a1822f7a0a8f0e4abcb4b1d07a7f6bba6c101d54` (merged before experimental training).
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision recorded in `model.json`.
- Training taskset: Terminal Lego (`PrimeIntellect/Terminal-Lego-15k`), with checkout revision archived before launch.
- Held-out evaluation: Terminal Bench 2, 64 tasks at step 0 and every 25 steps.
- Online evaluation uses the live inference pool; episodes may span weight refreshes.
  Export each episode's recorded policy span. Use the fixed-checkpoint audit for frozen-policy comparisons.

## Method

The paper subtracts the expected token score under the sampler distribution.
It approximates the unlogged tail as proportional to the trainer distribution.
We log 128 head tokens without renormalizing their probabilities.
The sampled token's logprob is recorded separately.

The primary treatment uses the paper's unweighted score-centering estimator.
Its policy loss is `-advantage * (sampled_logp - correction)`.
The correction has detached head coefficients `q_head - rho * p_head`, where
`rho = max(q_tail, 1e-6) / max(p_tail, 1e-6)`.
Both arms retain the same squared-log-ratio regularizer (coefficient 0.001).
Thus the primary contrast tests the complete published estimator against current IPO.
It changes importance weighting and masking as well as score centering.
It cannot attribute a difference to centering alone without an uncentered PG ablation.

An optional IPO-plus-centering ablation is implemented in `ipo-centered.toml`.
The reference code rejects absolute-probability masks because its head-only shortcut
assumes constant importance weights across the modeled tail.
Our extension evaluates IPO's mask across the full modeled vocabulary.
Its correction coefficients are `-p * 1[abs(p-qhat) > eps]`, with stopped gradients.
Subtracting `p` everywhere preserves the gradient because its expected score is zero.
The primary treatment decision was fixed before observing any training updates.
The stopped smoke2 pair checked IPO-plus-centering plumbing only; it completed no updates.

Both arms collect the same head metadata and use the same chunked LM head.
The baseline's objective is unchanged. The treatment uses the published score estimator.
No trainer optimization or reduction precision settings change.

## Resource and configuration limits

Run both arms concurrently. Each arm uses one H200 trainer node and one inference node.
Set adaptive concurrency to initial 512 and maximum 2048 episodes per arm.
Never exceed eight experiment nodes, including simultaneous preflight allocations.
Use the shared checkpoint, task draw sequence, sampling parameters, and run length.
Sample Lego tasks uniformly with replacement using seed 42 in both arms.
Use the existing pool sampler with one pool of weight 1; rewards cannot alter sampling.
This avoids sequential task difficulty changes that could mimic a reward collapse.
Fix this choice before the main runs; smoke runs used sequential tasks.
Async completion and admission can still change which draws enter each optimizer step.
Use 32K context, batch 128, group 8, AdamW at 1e-6, and at most four stale steps.
Use temperature 1 with no top-p, top-k, or min-p truncation.
Disable router replay in both arms.
Quantize inference experts to INT8 and dense layers to FP8. Use FP8 KV cache.
The INT4-cache smoke pair produced occasional NaN serving responses before updates.
Both trainers remained finite, so those errors do not demonstrate optimizer instability.
A matched FP8-cache diagnostic removed the observed NaN responses and reduced mismatch KL
from roughly 0.3–2.1 to 0.004–0.007 on initial smoke batches.
Choose FP8 cache before the main comparison to avoid that serving failure.
This tests a smaller measured mismatch; stable training cannot rule out benefits at stronger mismatch.
Confirm quantization and successful weight refresh in runtime logs before analysis.

## Stages

1. Verify gradients, zero constant-reward drift, and metadata packing.
2. Validate Prime VM runtime and tasksets with bounded `uv run eval` runs.
3. Run CUDA loss tests and a concurrent three-update smoke pair.
4. Run a concurrent 400-update pair from the shared initial checkpoint.
5. If a separation appears, repeat the pair with a second sampling seed.
6. If both arms stay stable, report that result. Any stronger stress test changes both arms together.

Keep smoke results separate from the main comparison.
Archive all config changes and failed attempts. Do not choose checkpoints after inspecting eval scores.
Infrastructure errors are not evidence of training collapse.

## Measurements and decision rules

Report training success, held-out success, output length, truncation, and task errors.
Report entropy, mismatch KL, gradient norm, mask fraction, staleness, and sampler head mass.
Compare matched optimizer steps and generated-token counts, with wall time as a secondary view.
Show task-level uncertainty for evaluation and variation across any repeated seeds.

Call a run numerically unstable if loss, gradient norm, or model weights become nonfinite.
Call it a reward collapse if a 25-step success average drops below half its best prior
25-step average for at least 50 steps, with a corresponding held-out decline.
Require nontrivial initial success (at least 10%) before applying this relative criterion.
Inspect traces for repetition, invalid tool calls, and lost task competence.
A timeout, sandbox failure, or missing reward signal alone does not meet this criterion.

Support the requested claim only if the baseline becomes unstable and the centered arm
remains finite and preserves capability under the same mismatch and training budget.
A single pair is exploratory evidence. If both fail or both succeed, report that outcome.
The paper's most severe experiments use a different engine, optimizer, and short math sequences.
This experiment tests transfer to production-style agentic RL rather than reproducing those curves exactly.

## Supplemental unseen-task audit

Decision recorded at 2026-09-19 06:55 UTC, after the initial TB2 evaluation.
IPO solved 1/51 scored tasks, with 13/64 failed episodes.
SC solved 5/43 scored tasks, with 21/64 failed episodes.
The estimates are sparse and have different missingness. They do not establish a treatment effect.

Keep the running pair and its scheduled TB2 evaluations unchanged.
After both runs finish, select up to 128 Lego tasks absent from either arm's training dispatch logs.
Exclude every dispatched task, including failed, rejected, and cancelled episodes.
Map task indices using the pinned dataset and its complete source-order manifest.
Sort eligible task names by SHA256("score-centering-heldout-v1:" + task_name), then take the first 128.
Archive the eligible set, excluded set, selected set, dataset revision, and run revisions before scoring.
If fewer than 64 tasks remain, report that this audit lacks the planned sample size.

Evaluate the initial model and both final checkpoints on the same selected tasks, four attempts per task.
Use the same quantization, context, agent budgets, and sampling settings as the main pair.
Pin standalone audit concurrency to 128 for each model.
Use task-level paired uncertainty and report errors plus missing-reward bounds.
Do not select tasks or checkpoints using evaluation scores.
This is a supplemental audit selected by training exposure, not an original random holdout split.
Its conclusion applies to the unseen task set; retain the original TB2 results and limitations.

## Resource revision on 2026-09-19

The user requested one trainer plus one inference node per arm and a cap near 2000.
Restart both arms from the same base weights with initial concurrency 512 and cap 2048.
Archive the early four-node runs (`ipo-seed42`, `sc-seed42`) separately; do not pool their updates with the new pair.
The early runs had no saved checkpoint and spent most trainer time waiting for rollouts.
Inference KV use was below 1% in the inspected sample with the 192-episode cap.
All other training, loss, quantization, task, and evaluation settings stay fixed.
New run names are `ipo-1plus1-seed42` and `sc-1plus1-seed42`.

The first 1+1 attempt (jobs 868/867) failed serving health checks before any optimizer update.
Both arms hit repeated unhealthy-worker errors under the larger episode load.
Restart both with four API workers per engine to parallelize Python response processing.
Retain the same model, GPU engines, loss, sampling, initial concurrency, and inflight cap.
Use fresh run names `ipo-1plus1-api4-seed42` and `sc-1plus1-api4-seed42`.

At the 08:55 restart, jobs888/887 exposed a hard-coded single-API-worker CLI override in the per-rank launcher.
Both jobs stopped before rollout collection. The helper now reads PRL_INFERENCE_API_SERVER_COUNT, defaulting to its previous value1.
The experiment explicitly sets4; all other configurations retain the default.
The generated two-node launch passed shell syntax validation.
A synthetic head-node benchmark measured 3.37s and28.9MB for a4096-token top129 response, versus0.026s and0.50MB for top1.
This supports a response-processing hypothesis but does not establish the live bottleneck.
The replacement names are `ipo-1plus1-api4-seed42-r2` and `sc-1plus1-api4-seed42-r2`.
