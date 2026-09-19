# Score centering on agentic terminal RL

## Question

Does score centering improve robustness to quantized inference in current prime-rl?
Compare current-main IPO (`eps=0.3`) against the paper's standalone score-centering loss.
Treat baseline collapse as a hypothesis, not a required outcome.

## Sources and revisions

- Paper: https://arxiv.org/html/2609.20807v1
- Reference code: https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321
- Baseline main: `8c2847721ada557e3c9dabb4d34af81393c27391` (merged before experimental training).
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision recorded in `model.json`.
- Training taskset: Terminal Lego (`PrimeIntellect/Terminal-Lego-15k`), with checkout revision archived before launch.
- Held-out evaluation: Terminal Bench 2, 64 tasks at step 0 and every 25 steps.

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

Run both arms concurrently. Each arm uses one H200 trainer node and three inference nodes.
Never exceed eight experiment nodes, including simultaneous preflight allocations.
Use the shared checkpoint, task order, sampling parameters, and run length.
Use 32K context, batch 128, group 8, AdamW at 1e-6, and at most four stale steps.
Use temperature 1 with no top-p, top-k, or min-p truncation.
Disable router replay in both arms.
Quantize inference experts to INT8 and dense layers to FP8. Use INT4 KV cache.
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
