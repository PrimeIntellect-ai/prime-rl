# Performance Benchmarks

Automated benchmark results for prime-rl using `--bench` flag.

**Last Updated:** 2026-02-07 03:03 UTC  
**Commit:** `30bf494f`  
**Docker Image:** `None`

> :warning: indicates regression > 5% from baseline
> diffs shown when abs(change) >= 1.0% (except regressions, which always show diffs)

> :clock10: The Step Time shown is the time taken per micro batch. This differs from what gets displayed in the bench table which is the total step time.
## INTELLECT-3

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL LoRA(r=16) | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 13.3% | 25.56k | 5.13s | 91.9 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 19.8% | 16.69k | 7.85s | 89.7 GiB |
| RL LoRA(r=16) | 16384 | Offload | FA3 | 1 | 1 | 8xH200 | 19.2% | 16.15k | 8.11s | 84.2 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 2.5% | 4.78k | 27.45s | 90.4 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 19.6% | 17.36k | 30.20s | 161.7 GiB |
| RL LoRA(r=16) | 65536 | Offload | FA3 | 1 | 1 | 8xH200 | 25.1% | 9.76k | 53.73s | 128.8 GiB |
| RL LoRA(r=16) | 65536 | Offload | FA2 | 1 | 1 | 8xH200 | 16.2% | 6.30k | 83.23s | 130.3 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 5.8% | 5.16k | 101.60s | 153.6 GiB |
| RL LoRA(r=16) | 65536 | Offload | FA2 | 1 | 1 | 8xB200 | 5.7% | 5.04k | 104.10s | 131.5 GiB |

## Qwen3-0.6B

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 11.1% | 11.90k | 1.38s | 12.5 GiB |
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 1xA6000 | 10.9% | 3.69k | 4.44s | 12.5 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 26.8% | 10.15k | 6.46s | 19.6 GiB |
| RL Full | 65536 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 26.4% | 9.98k | 6.57s | 16.1 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 4xA6000 | 16.2% | 7.72k | 33.94s | 12.4 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 1xA6000 | 17.1% | 2.04k | 32.12s | 19.5 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 10.9% | 13.34k | 1.23s | 5.1 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 1xA6000 | 6.3% | 2.41k | 6.79s | 5.1 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 25.2% | 9.97k | 6.58s | 11.5 GiB |
| RL LoRA(r=16) | 65536 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 24.7% | 9.78k | 6.70s | 7.9 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA2 | 1 | 1 | 1xA6000 | 16.0% | 1.99k | 32.85s | 11.5 GiB |
| SFT Full | 8192 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 16.8% | 26.01k | 0.32s | 31.7 GiB |
| SFT Full | 8192 | Recompute | FA2 | 1 | 1 | 1xA6000 | 13.5% | 6.60k | 1.24s | 31.2 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 23.7% | 25.49k | 0.64s | 52.8 GiB |

## Qwen3-235B-A22B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 4.2% | 4.14k | 31.69s | 161.5 GiB |
| RL LoRA(r=16) | 16384 | Offload | FA2 | 1 | 1 | 8xB200 | 4.2% | 4.13k | 31.76s | 150.3 GiB |

## Qwen3-30B-A3B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 2.8% | 13.57k | 9.66s | 85.6 GiB |
| RL Full | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 2.5% | 11.87k | 11.04s | 86.6 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 2.9% | 6.11k | 21.44s | 74.6 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 2.8% | 5.94k | 22.06s | 74.6 GiB |
| RL Full | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 12.7% | 24.01k | 21.84s | 108.8 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 8.8% | 16.55k | 31.69s | 101.3 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 15.4% | 12.75k | 41.13s | 105.4 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 10.0% | 49.13k | 2.67s | 32.9 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 17.0% | 36.77k | 3.56s | 31.6 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 7.3% | 36.12k | 3.63s | 32.9 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 15.3% | 33.10k | 3.96s | 31.6 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 19.7% | 37.43k | 14.01s | 64.3 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 29.2% | 24.41k | 21.48s | 63.7 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 27.9% | 23.39k | 22.42s | 63.7 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 10.0% | 19.06k | 27.51s | 64.3 GiB |
| SFT Full | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 11.6% | 55.68k | 2.35s | 111.8 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 11.6% | 55.67k | 2.35s | 111.8 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 16.6% | 35.03k | 3.74s | 106.4 GiB |

## Qwen3-4B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 9.4% | 40.20k | 3.26s | 18.1 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 14.7% | 27.44k | 4.78s | 17.1 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 13.9% | 26.03k | 5.04s | 17.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 36.1% | 29.54k | 17.75s | 36.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 35.4% | 28.97k | 18.10s | 36.1 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 13.9% | 25.96k | 20.20s | 39.0 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 22.8% | 52.38k | 2.50s | 10.3 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 9.8% | 51.21k | 2.56s | 11.1 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 21.2% | 48.72k | 2.69s | 10.3 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 21.7% | 6.23k | 2.63s | 23.3 GiB |
| RL LoRA(r=16) | 16384 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 20.7% | 5.95k | 2.75s | 20.6 GiB |
| RL LoRA(r=16) | 16384 | Recompute | FA2 | 1 | 1 | 1xA6000 | 13.6% | 1.23k | 13.29s | 23.3 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 37.5% | 33.43k | 15.68s | 28.8 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 36.8% | 32.80k | 15.98s | 28.8 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 12.8% | 25.90k | 20.25s | 31.7 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 37.4% | 4.16k | 15.74s | 42.0 GiB |
| RL LoRA(r=16) | 65536 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 36.3% | 4.04k | 16.21s | 31.1 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xB200 | 15.5% | 66.04k | 1.98s | 52.5 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH200 | 28.4% | 53.14k | 2.47s | 54.6 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH100 HBM3 | 26.6% | 49.72k | 2.64s | 54.6 GiB |
| SFT Full | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 14.2% | 26.39k | 19.86s | 171.5 GiB |

## Llama-3.1-8B-Instruct

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL LoRA(r=16) | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 20.5% | 66.01k | 1.99s | 16.6 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 37.7% | 50.82k | 10.32s | 43.9 GiB |

## Llama-3.2-1B-Instruct

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL LoRA(r=16) | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 9.9% | 155.50k | 0.84s | 5.6 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 22.3% | 130.51k | 4.02s | 15.5 GiB |

## Llama-3.2-3B-Instruct

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL LoRA(r=16) | 16384 | Recompute | FA4 | 1 | 1 | 8xB200 | 16.2% | 97.78k | 1.34s | 10.0 GiB |
| RL LoRA(r=16) | 65536 | Recompute | FA4 | 1 | 1 | 8xB200 | 35.2% | 78.64k | 6.67s | 28.0 GiB |
