# PR Notes: feat/separate_is_rejection

## Design decisions

- **Importance sampling and rejection sampling are independent axes**: The refactor separates these two concerns cleanly. `compute_importance_weights` handles IS (soft reweighting with optional truncation/clipping). The `reject_by_*` functions handle rejection sampling (hard masking). They can be used independently or combined.

- **Rejection functions take `log_ratio` as input, not pre-computed weights**: Each `reject_by_*` function is self-contained — it takes `log_ratio` (and `mask` where needed) and computes whatever intermediate values it needs internally. This makes the API explicit for custom loss function authors: you don't need to know about `compute_importance_weights` or pre-compute the right intermediate to use rejection. The trade-off is redundant `exp(log_ratio)` across functions, but this is trivial compute.

- **Rejection operates on unclamped weights, IS clipping is separate**: Rejection masks must reflect the actual divergence, not already-softened values. If `compute_importance_weights` clips token weights for IS purposes, those clipped weights should not be fed into rejection — you'd be masking based on the wrong signal. The current design enforces this: `reject_by_*` functions compute their own weights from raw `log_ratio`, while `compute_importance_weights` handles IS clipping independently.

- **Detach on `masked_sum` removed**: The original `default_loss_fn` had `.detach()` on `log_importance_ratio[loss_mask].sum()` when computing the sequence importance weight. This is unnecessary because `coeff.detach()` already blocks gradient flow through all importance weights. Importance weights are not part of the optimization objective — they correct for distribution shift, not provide signal. The early detach was either defensive or an artifact of an earlier loss formulation.

## Behavioral changes

- **SAFETY_BOUNDS clamping added to token weights**: The new code clamps `log_ratio` at `[-30, 30]` before `exp()` in `compute_importance_weights` and all `reject_by_*` functions. The old code had no such clamping on token weights. Only affects values when `|log_importance_ratio| > 30` (i.e. `exp(30)` ~ 1e13), which should never occur in practice.

- **Sequence log clamp relaxed from 10 → 30**: The old code clamped `seq_log_importance_ratio` at `max=10.0` (~22026). The new `compute_importance_weights` clamps at `max=SAFETY_BOUNDS=30.0` (~1e13). This is moot when `sequence_clip_high` (default 10.0) is applied afterward, but if someone sets a very large `sequence_clip_high`, the effective upper bound on the unclipped sequence weight increases.
