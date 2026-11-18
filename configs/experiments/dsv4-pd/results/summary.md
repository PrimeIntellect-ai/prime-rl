# Results

Per-GPU throughput, second half of each hold. Full log: `../NOTES.md`; report and PR: #4565.

## Stress test: 2 nodes / 16 B200 per arm (agg = 2x aggregated DP8+EP, pd = P/D 1P:1D, pdmc = pd + Mooncake)

**Output tok/s/GPU** (model calls/s in parentheses):

| Concurrency | agg | pd | pdmc |
|---|---|---|---|
| 2048 | 980 (13.4) | **165 (2.2), thrashed** | 1,225 (23.9) |
| 1536 | 1,213 (14.1) | 1,872 (37.3) | 1,673 (32.0) |
| 1024 | 967 (14.3) | 1,696 (34.6) | 1,673 (34.6) |
| 768 | 853 (13.6) (partial) | 1,495 (25.0) (partial) | 1,496 (24.8) (partial) |
| 512 | 633 (13.6) | 1,083 (22.9) | – |
| 384 | 556 (12.0) | 898 (19.1) | – |
| 256 | 454 (9.8) | 702 (13.6) | – |
| 128 | 330 (5.8) | 435 (7.3) | – |
| 32 | 156 (1.8) | 158 (1.9) | 154 (1.8) |

**Recomputed prompt share / time to first token / inter-token latency:**

| Concurrency | agg | pd | pdmc |
|---|---|---|---|
| 2048 | 6.0% / 1.5 s / 135 ms | 99.8% / 459.9 s / – | 11.3% / 35.2 s / 24 ms |
| 1536 | 3.3% / 1.0 s / 77 ms | 5.9% / 4.7 s / 34 ms | 6.7% / 10.9 s / 31 ms |
| 1024 | 2.5% / 0.9 s / 64 ms | 3.5% / 1.3 s / 31 ms | 3.6% / 1.5 s / 31 ms |
| 768 | 3.0% / 0.8 s / 58 ms (partial) | 3.0% / 1.1 s / 27 ms (partial) | 3.1% / 1.3 s / 27 ms (partial) |
| 512 | 3.0% / 0.7 s / 47 ms | 2.6% / 0.8 s / 24 ms | – |
| 384 | 2.9% / 0.7 s / 40 ms | 2.7% / 0.8 s / 22 ms | – |
| 256 | 3.0% / 0.6 s / 31 ms | 2.7% / 0.7 s / 18 ms | – |
| 128 | 3.1% / 0.5 s / 21 ms | 3.0% / 0.6 s / 15 ms | – |
| 32 | 3.7% / 0.5 s / – | 3.5% / 0.5 s / – | 3.4% / 0.5 s / – |

## Tuning (in progress: 4-node arms at 3072, then 4096)

| run | episodes | GPUs | hold min | out tok/s/GPU | calls/s | recompute | prefill wait | decode wait | TTFT s | ITL ms |
|---|---|---|---|---|---|---|---|---|---|---|
| stress-agg-L1536 | 1536 | 16 | 75 | 1,213 | 14.1 | 3.3% | 0 | - | 1.0 | 77 |
| stress-pd-L1536 | 1536 | 16 | 75 | 1,872 | 37.3 | 5.9% | 147 | 4 | 4.7 | 34 |
| tune-agg_mbt32k-L1536 | 1536 | 16 | 75 | 1,075 | 11.6 | 5.7% | 0 | - | 1.5 | 89 |
| tune-pd_mbt32k-L1536 | 1536 | 16 | 75 | 200 | 2.2 | 99.9% | 1476 | 0 | 330.6 | 12 |
