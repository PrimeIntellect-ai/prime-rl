# Performance Benchmarks

Automated benchmark results for prime-rl using `--bench` flag.

**Last Updated:** 2026-01-20 05:09 UTC  
**Commit:** `unknown`  
**Docker Image:** `primeintellect/prime-rl@sha256:d3e9636af265f5ba2e56cd2555534489f5ff64e04a3f645059ca7989508de419`

> :warning: indicates regression > 5% from baseline
> diffs shown when abs(change) >= 1.0% (except regressions, which always show diffs)

> :clock10: The Step Time shown is the time taken per micro batch. This differs from what gets displayed in the bench table which is the total step time.
## Qwen3-0.6B

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 11.5% (+3.6%) | 12.33k (+3.6%) | 1.33s | 12.5 GiB |
| RL Full | 16384 | Recompute | FA2 | 1 | 1 | 1xA6000 | 11.2% (+2.6%) | 3.78k (+2.6%) | 4.33s | 12.5 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 27.4% (+2.1%) | 10.36k (+2.1%) | 6.32s | 19.6 GiB |
| RL Full | 65536 | Offload | FA3 | 1 | 1 | 1xH100 HBM3 | 26.9% (+2.1%) | 10.19k (+2.1%) | 6.43s | 16.1 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 1xA6000 | 17.6% | 2.10k | 31.21s | 19.5 GiB |
| SFT Full | 8192 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 17.1% (+2.0%) | 26.52k (+2.0%) | 0.31s | 31.7 GiB |
| SFT Full | 8192 | Recompute | FA2 | 1 | 1 | 1xA6000 | 13.5% | 6.60k | 1.24s | 31.2 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 1xH100 HBM3 | 23.9% | 25.61k | 0.64s | 52.8 GiB |

## Qwen3-30B-A3B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | **2.7%** :warning: (-8.6%) | **5.59k** :warning: (-8.6%) | 23.45s | 74.6 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | **2.6%** :warning: (-7.9%) | **5.47k** :warning: (-7.9%) | 23.95s | 74.6 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 15.0% (-2.7%) | 12.40k (-2.7%) | 42.28s | 105.4 GiB |
| SFT Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 16.2% (-2.7%) | 34.09k (-2.7%) | 3.85s | 106.4 GiB |

## Qwen3-4B-Instruct-2507

| Type | SeqLen | AC | Attn | EP | CP | Hardware | MFU | TPS | Step Time | Peak Mem |
|------|--------|----|----|----|----|----------|-----|-----|-----------|----------|
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH200 | 14.0% (-4.6%) | 26.18k (-4.6%) | 5.01s | 17.1 GiB |
| RL Full | 16384 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 13.3% (-4.0%) | 24.98k (-4.0%) | 5.25s | 17.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH200 | 36.5% (+1.1%) | 29.86k (+1.1%) | 17.56s | 36.1 GiB |
| RL Full | 65536 | Recompute | FA3 | 1 | 1 | 8xH100 HBM3 | 35.3% | 28.87k | 18.16s | 36.1 GiB |
| RL Full | 65536 | Recompute | FA2 | 1 | 1 | 8xB200 | 12.8% | 23.83k | 22.00s | 36.0 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH200 | 29.0% (+2.1%) | 54.27k (+2.1%) | 2.42s | 54.6 GiB |
| SFT Full | 16384 | Recompute | FA2 | 1 | 1 | 8xH100 HBM3 | 27.1% (+2.1%) | 50.76k (+2.1%) | 2.58s | 54.6 GiB |

## Failed Benchmarks

- **Qwen/Qwen3-0.6B** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **PrimeIntellect/INTELLECT-3** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-0.6B** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 1xA6000: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 8xH100 HBM3: Non-zero exit code: 1
- **PrimeIntellect/INTELLECT-3** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-0.6B** (RL LoRA(r=16)) on 1xA6000: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-235B-A22B-Instruct-2507** (RL LoRA(r=16)) on 8xB200: Timeout
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 8xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-0.6B** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **PrimeIntellect/INTELLECT-3** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-30B-A3B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **PrimeIntellect/INTELLECT-3** (RL LoRA(r=16)) on 8xH200: Non-zero exit code: 1
- **Qwen/Qwen3-4B-Instruct-2507** (RL LoRA(r=16)) on 1xH100 HBM3: Non-zero exit code: 1
- **Qwen/Qwen3-0.6B** (RL LoRA(r=16)) on 1xA6000: Non-zero exit code: 1
