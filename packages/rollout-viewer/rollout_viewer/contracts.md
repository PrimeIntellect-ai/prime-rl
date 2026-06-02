# Contracts — producer ⇄ store ⇄ viewer

The seams every workstream builds against. Types live in `schema.py` (Episode/Step/
diagnostics) and `storage.py` (StorageBackend + layout). This doc fixes the
behaviour the types don't encode: the **diagnostics sidecar** and the **watertight
join**.

## Data flow

```
trainer (W0)           sync (W1)                       store (W1)            viewer (W4/W5)
  rollouts/step_N/      strip→reduce→gzip→parquet        HF-free private       FastAPI + DuckDB + SPA
    train_rollouts.jsonl ───────────────┐               index/{run}/*.parquet  list/filter/sort/compare
    train_diagnostics.jsonl  (NEW) ──────┴── join ──────► transcripts/{run}/*  per-seat drill-down
```

## Diagnostics sidecar (produced by W0, consumed by W2)

KL / IS-ratio / entropy need the trainer's forward-pass logprobs and are computed
per-token inside the loss, today collapsed to a batch mean (`importance_ratio`,
`entropy`, `mismatch_kl`) with rollout identity already dropped at microbatch
packing. **They cannot be reconstructed from the rollout dump** (which carries only
the *old* sampling logprobs). So W0 captures them at the source.

**File:** `rollouts/step_N/train_diagnostics.jsonl`, one line per
`(trajectory_id, member_id, step_index)`:

```json
{
  "trajectory_id": "…", "member_id": "debater_a", "step_index": 0,
  "n_tokens": 9458, "status": "present",
  "importance_ratio": {"sum": …, "count": 9458, "mean": …, "p50": …, "p90": …, "p99": …, "min": …, "max": …},
  "mismatch_kl":      {"sum": …, "count": 9458, "mean": …, …},
  "entropy":          {"sum": …, "count": 9458, "mean": …, …},
  "masked_low_frac": 0.0, "masked_high_frac": 0.0
}
```

**W0 requirements (watertight):**
- Thread `(trajectory_id, member_id, step_index)` through microbatch packing so each
  packed sequence stays attributable. Use the SAME `trajectory_id` + `extras.member_id`
  + step order that appear in `train_rollouts.jsonl` — the join key must match exactly.
- Reduce the per-token `importance_ratio` / `mismatch_kl` / `entropy` **per sequence**
  (the rich `ValueSummary`: sum, count, mean, p50/p90/p99, min, max). Minimal
  `{sum, count, mean}` is allowed but "rich as hell" is the goal.
- **Correctness oracle (the prove-it test):** Σ over sequences of `summary.sum` MUST
  equal the existing batch-aggregate numerator, and Σ `count` the denominator — i.e.
  per-rollout reductions re-aggregate to the unchanged batch scalar. A test asserts this.
- `status`: `"present"` when the sequence contributed gradient; `"masked_out"` when
  fully clipped by `mask_ratio_*` (no IS/KL to report — **not** zero); never emit a
  fabricated `0.0`.
- Gated behind a config flag, **default off**, so existing runs are byte-identical.
- Do **not** touch `optimization_dtype` / `reduce_dtype` or any numerical knob.

## The join (W2)

`parser` loads `train_rollouts.jsonl` → `Episode.from_rollout_output` (steps have
`diagnostics=None`), then joins `train_diagnostics.jsonl` onto each `Step`:

- Key: `(trajectory_id, member_id, step_index)`, exact.
- **Mismatch is an error.** If an episode's step has no matching diagnostics row
  *and* the run has a diagnostics sidecar *and* the episode is not errored/filtered →
  `raise`. Do not align-by-position or fill defaults.
- A run with **no** sidecar at all → every Step's `diagnostics = StepDiagnostics(status="absent")`.
  This is the only path that sets "absent"; it is explicit, never inferred per-step.
- Carry `count`; the viewer re-aggregates `sum/count`, never mean-of-means.

## Strip (W1/W2)

The viewer artifact drops `tokens.*` id/mask/logprob arrays (~92% of raw bytes;
measured 1.74 MB → 0.14 MB/episode) AFTER their diagnostic reduction exists. Then
**gzip** the per-step transcript shards (~4× further) and write the index as
**parquet+zstd**. Order of cost levers: compression ≈ retention > token-strip > backend.

## Retention (W1)

`ρ` = fraction of episodes whose full transcript is written to the cloud store
(metrics + diagnostics for *every* episode are always written — they are tiny and
power filter/sort across all rollouts even when a transcript wasn't retained).
`ρ=1` (keep-all) is the default while under `SOFT_LIMIT_BYTES`. The sync **warns at
SOFT and refuses past HARD** (HF-free 100 GB private ceiling) — fail loud, never
silent-drop. Crossing the ceiling is the documented trigger to migrate to R2.

## Store layout (W1/W4) — see `storage.py`

```
runs.json                                  registry
index/{run_id}/episodes.parquet            INDEX_COLUMNS, one row/episode
transcripts/{run_id}/step-{step:05d}.jsonl.gz   full per-step messages
```

**Transcript shard line format (W1 writes, W4 reads — agree via this, not each other):**
each line of `step-{step:05d}.jsonl.gz` is one **stripped** `Episode` as
`Episode.model_dump()` JSON — i.e. full `steps[].prompt/completion` messages +
joined `diagnostics`, but no token-id arrays. The index row's `transcript_shard`
+ `transcript_line` (0-based) point at the episode's line in its step shard.
Order within a shard is stable (episode order in `train_rollouts.jsonl`).

HF-free is used as **dumb parquet/blob storage** — few large objects, **batched
commits** (never one-commit-per-step), squash history periodically. This dodges the
git-LFS file-count (<100k) and commit-rate limits that break HF as a database.

## Diagnostics rollup schema (W1 writes the index cell, W4 reads it — pinned)

The `diagnostics` column of `episodes.parquet` is a JSON object, one entry per
**scope** — the literal `"episode"` plus each `member_id`:

```json
{
  "episode":   {"status": {"present": 4, "masked_out": 1}, "n_steps": 5,
                "mismatch_kl": {"sum": …, "count": …, "mean": …, "p50": …, "p90": …, "p99": …, "min": …, "max": …},
                "importance_ratio": {…}, "entropy": {…}},
  "debater_a": {"status": {"present": 2}, "n_steps": 2, "mismatch_kl": {…}, …},
  "judge":     {"status": {"masked_out": 1}, "n_steps": 1}
}
```

Rules:
- A `<quantity>` block (`mismatch_kl` / `importance_ratio` / `entropy`) is **omitted**
  when no step in that scope has a `ValueSummary` for it (e.g. a fully `masked_out`
  judge). Omission ≠ zero.
- `sum` and `count` are **always present** when the block is — so cross-scope /
  cross-step / cross-run re-aggregation is `Σsum / Σcount` (never mean-of-means).
- Quantiles do not merge exactly across steps; the rollup reports a **worst-case
  envelope** (max of p90/p99, min of min, max of max) and an **exact** `mean = sum/count`.
  True cross-step quantiles need a producer-side sketch (t-digest), not a scalar.
- Frontend/`compare` access is dotted: `diag.<scope>.<quantity>.<field>`, e.g.
  `diag.episode.mismatch_kl.mean`, `diag.debater_a.importance_ratio.p99`.
- **`compare` MUST aggregate `SUM(sum)/SUM(count)`** for diagnostics, and use
  `COUNT(<expr>)` (NULL-skipping) as the sample `n` for plain metrics — never
  `AVG()` over per-episode means.

## Retention selection (W1 writes, W4 reads — pinned)

With `ρ < 1`, the retained set is **deterministic by source order**: the first
`ceil(ρ·n)` episodes of each step (byte-stable across re-syncs). Metrics + diagnostics
are written for **every** episode regardless of ρ. For a non-retained episode the
index row carries its `transcript_shard` but `transcript_line = -1` (the
`NOT_RETAINED` sentinel). **W4 MUST treat `transcript_line < 0` as "transcript not
retained"** — return an explicit not-retained response, never fetch line `-1` and
never 404-as-if-missing. (Smart retention — preferentially keep flagged / high-KL /
flipped / errored episodes instead of first-N — is a deferred product decision; only
relevant past the SOFT ceiling, ρ=1 keep-all until then.)

## Visibility conventions (the promise W3 verifies — pinned)

These are load-bearing; a renderer change that breaks them silently breaks the
visibility guarantee, so they live here, not only in code:
- **`[<member_id>]` prefix**: cross-seat content is interpolated into a *user* turn as
  plain text, prefixed at position 0 with `[<seat>]` (e.g. `[debater_b] …`), not carried
  as a separate assistant message. Leakage attribution depends on this prefix.
- **Causal, not phase-blind**: a seat's turn at index `t` sees every seat that produced
  a completion at index `< t` (sequential/causal). Phases (`propose`/`critique`/`final`)
  are NOT a blind-vs-open distinction — `debater_b`'s propose already sees `debater_a`'s
  propose. The visibility policy is the acyclic turn-order rule, generalizing to N seats.
- **`completion` ≠ injected-assistant**: a seat's own prior `Step.completion` is NOT
  byte-identical to how it reappears in the next prompt (the renderer re-materializes it,
  e.g. re-wrapping the `<think>` block). The prompt-delta matches by `(role, content)`
  membership, never by equality with the raw completion.
- Non-string (multimodal) message content must **not** silently disable leakage scanning
  — serialize its text for attribution or emit an explicit structural finding (fail loud).
