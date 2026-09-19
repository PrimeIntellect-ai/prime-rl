# Score-centering experiment results

Status at 2026-09-19 06:34 UTC: the main pair is running. No robustness conclusion yet.

## Main comparison

- Model: Qwen3-30B-A3B-Instruct-2507, pinned snapshot in `model.json`.
- Source: experimental commit `ef2cfad80`, with main `a1822f7a0` merged.
- Baseline: current-main IPO, eps 0.3; SLURM 861, `ipo-seed42`.
- Treatment: published standalone score centering; SLURM 860, `sc-seed42`.
- Each arm: one H200 trainer node plus three inference nodes, running concurrently.
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

## Pending

Collect the main learning curves and held-out results at matched updates and token budgets.
Inspect trace failures before classifying instability.
If a separation appears, repeat with a second seed and a matched BF16 control.
Report a negative or inconclusive result if the requested separation does not occur.
