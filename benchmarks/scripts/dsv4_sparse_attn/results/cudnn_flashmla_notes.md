# `cudnn_flashmla` results notes

These results are for the `cudnn_flashmla` backend (FlashMLA sparse prefill forward, cuDNN frontend SM90 DSA
backward) next to `tilelang` and the FlashMLA forward reference, at commit `e28cc525e`. They were measured on
2026-10-10 on `prime-nebius-puku-h200-gpu-059` (H200, SLURM job 3569, GPU 0). The node was quiet: no other process
used any GPU.

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight indexer and are near-uniform over the
readable entries. A trained indexer favors recent and neighboring entries, so CSA gather locality here is
pessimistic.

## Files

- `cudnn_flashmla.json`: `bench.py --backends tilelang cudnn_flashmla flashmla_fwd_ref --label cudnn_flashmla`,
  over the 240 grid items at the default 7 rounds of 10 calls.
- `cudnn_flashmla_tables.md`: `bench.py --compare cudnn_flashmla.json`.
- `cudnn_flashmla_stream.json`, `cudnn_flashmla_stream_tables.md`: `stream.py --backends tilelang cudnn_flashmla`
  and its tables.
- `cudnn_flashmla_summary.md`: `summarize.py cudnn_flashmla.json cudnn_flashmla_stream.json`. Its headline table
  is the tilelang baseline, and its correctness and stream sections cover every arm.
- `cudnn_flashmla_vs_main.md`: `bench.py --compare main.json cudnn_flashmla.json`, the tilelang regression check.

## Findings

- Correctness: `cudnn_flashmla` passes the gate on all 240 items.
  - Against tilelang, the largest relative errors are: out 5.2e-3, lse 4.2e-7, dq 7.1e-3, dkv 7.3e-3, and dsink
    9.5e-3 against a 1e-2 bound. The dsink margin is thin.
  - Against the fp32 dense reference (items of 4096 tokens or fewer), its dsink error is 6.7e-3. tilelang's own
    dsink error there is 9.6e-3, so the thin margin is shared by both bf16 backends.
- Speed, as the median ratio of `cudnn_flashmla` time to tilelang time over all 240 items (lower is better, range
  in brackets):
  - forward op-boundary 0.40 [0.35-0.53], forward GPU 0.54 [0.46-0.90];
  - forward+backward op-boundary 0.66 [0.51-0.77], forward+backward GPU 0.69 [0.50-0.91].
  - At 49k and 64k tokens the forward GPU ratio is 0.47-0.53. The forward+backward GPU ratio is 0.57 on CSA, 0.68-0.69
    on HCA and sliding.
  - The backward gains less than the forward. One reason is that O and dO are read twice: once by the TileLang
    delta kernel that feeds the torch-side sink gradient, and once by cuDNN's own preprocess.
- Host overhead (op-boundary minus GPU busy time) has a median of 195 µs per forward call and 673 µs per
  forward+backward call. tilelang's medians are 701 µs and 1020 µs.
- Dynamic stream (32 items, fresh caches):
  - cuDNN compiled 4 main backward kernels, one per slot-width bucket the stream reached: at items 0 (640 slots),
    1 (128), 2 (173, bucket 256) and 8 (346, bucket 512).
  - Its preprocess and postprocess kernels add 2 compiles, and the TileLang delta kernel adds 1.
  - The CuTe compiles recur in the warm process too, about 10.8 s of `cute.compile` per process, because
    `cute.compile` keeps no file cache.
  - The other 31 items take 7.9 s cold and 7.8 s warm, against tilelang's 0.62 s. That gap is the three
    mid-stream bucket compiles of about 2.5 s each.
- tilelang regression check: this run's tilelang time over `main.json`'s, per item, has a median of 1.000 on GPU
  time [0.984-1.040] for the forward and 1.000 [0.990-1.039] for forward+backward. On op-boundary time it is
  1.005 and 1.009. The backend-table refactor did not slow tilelang.
