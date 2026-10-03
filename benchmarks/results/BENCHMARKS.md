# Performance Benchmarks

Automated benchmark results for prime-rl using `--bench` flag.

**Last Updated:** 2026-02-01 00:43 UTC  
**Commit:** `unknown`  
**Docker Image:** `primeintellect/prime-rl-jackmin@sha256:5a146f7dfdcdf6b0e90fd3f1ed0874d22a3a0641378dc8dad46ce213d02fa2e6`

> :warning: indicates regression > 5% from baseline
> diffs shown when abs(change) >= 1.0% (except regressions, which always show diffs)

> :clock10: The Step Time shown is the time taken per micro batch. This differs from what gets displayed in the bench table which is the total step time.
## Qwen3-0.6B

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 11.1% | 11.90k | 1.38s | 12.5 GiB |
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 1xA6000 | 10.9% | 3.69k | 4.44s | 12.5 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 26.8% | 10.15k | 6.46s | 19.6 GiB |
| RL Full | 65536 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 26.4% | 9.98k | 6.57s | 16.1 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 4xA6000 | 16.2% | 7.72k | 33.94s | 12.4 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 1xA6000 | 17.1% | 2.04k | 32.12s | 19.5 GiB |
| SFT Full | 8192 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 16.8% | 26.01k | 0.32s | 31.7 GiB |
| SFT Full | 8192 | Recompute | FA2 | 1 | 1 | 1xA6000 | 13.5% | 6.60k | 1.24s | 31.2 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 23.7% | 25.49k | 0.64s | 52.8 GiB |

## Qwen3-30B-A3B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 2.9% | 6.11k | 21.44s | 74.6 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 2.8% | 5.94k | 22.06s | 74.6 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 15.4% | 12.75k | 41.13s | 105.4 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 16.6% | 35.03k | 3.74s | 106.4 GiB |

## Qwen3-4B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 6.5% | 27.54k | 4.76s | 17.0 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 14.7% | 27.44k | 4.78s | 17.1 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 13.9% | 26.03k | 5.04s | 17.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 36.1% | 29.54k | 17.75s | 36.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 35.4% | 28.97k | 18.10s | 36.1 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 16.0% | 68.36k | 1.92s | 54.6 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH200 | 28.4% | 53.14k | 2.47s | 54.6 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH100 HBM3 | 26.6% | 49.72k | 2.64s | 54.6 GiB |
| SFT Full | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 14.2% | 26.39k | 19.86s | 171.5 GiB |
