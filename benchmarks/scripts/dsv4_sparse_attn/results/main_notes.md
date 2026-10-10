# `main` baseline notes

These results are for `main`'s TileLang kernels at harness commit `6d5cf0180`. They were measured on 2026-10-10 on
`prime-nebius-puku-h200-gpu-059` (H200, SLURM job 3569, GPU 0). The node was quiet: no other process used any GPU.

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight indexer and are near-uniform over the
readable entries. A trained indexer favors recent and neighboring entries, so CSA gather locality here is
pessimistic.

## Files

- `main.json`: `bench.py --backends tilelang flashmla_fwd_ref --label main`, over the 240 grid items.
- `main_tables.md`: `bench.py --compare main.json`, with every item's op-boundary time, GPU busy time, FLOPs and
  peak memory.
- `main_stream.json`, `main_stream_tables.md`: `stream.py --backends tilelang --label main` and its tables.
- `main_summary.md`: `summarize.py main.json main_stream.json`. It holds the headline tables, the correctness
  margins and the per-item valid-slot histograms.
- `main_repeat_subset.json`, `main_repeat_compare.md`: a second run on 24 items, compared against the first.

## Findings

- TileLang forward+backward reaches 177-201 TFLOP/s on 64k-token CSA rows, which is 18-20% of the 989.5 TFLOP/s
  dense BF16 peak.
  - HCA rows reach 117-179 TFLOP/s, and sliding rows about 125 TFLOP/s.
  - Rows of tiny documents reach 50-60 TFLOP/s. Their queries have a median of about 38 valid slots, so the
    per-query fixed work dominates.
- Host overhead (op-boundary time minus GPU busy time) has a median of 692 µs per forward call and 971 µs per
  forward+backward call.
  - It exceeds the GPU time on 127 of 240 items in the forward:
    - 93 of the 96 items of 4096 tokens or fewer (all 72 cp=8 slices and 21 of 24 cp=1 rows);
    - 34 of the 108 cp=8 slices of longer rows;
    - none of the 36 longer cp=1 rows.
  - The source is TileLang's per-call adapter, which turns symbolic shapes into strings on every call (see the
    Plan 0 status notes).
- The FlashMLA forward reference takes 0.43-0.52 of TileLang's forward time on large GPU-bound items, and
  0.25-0.38 on small host-bound items.
- Dynamic stream: TileLang compiles 4 kernels on the first item when the cache is cold, and none per shape after
  that.
  - The other 31 items take 0.61 s, cold and warm alike.
  - Every process also pays about 42 s importing `prime_rl.trainer.models`.
- Correctness: every arm passed on every item. The closest margin is tilelang's `dsink` against the fp32 reference,
  at 9.56e-3 against a bound of 1e-2.
- Repeatability (`main_repeat_compare.md`):
  - Large GPU-bound items reproduce within 0.5%.
  - Small host-bound items move by up to about 9% for TileLang (short-4096-csa-cp8r4 forward: 877 then 804 µs) and
    up to 15% for the reference (single-4096-csa-cp8r4: 248 then 211 µs). That is often outside a single run's
    p20-p80 band.
  - Compare small items across runs with that host jitter in mind.
- On some large items, GPU busy time is slightly above op-boundary time (by at most 110 µs, about 0.5%). The GPU
  busy time comes from a separate profiled pass, so the two are not the same calls.
