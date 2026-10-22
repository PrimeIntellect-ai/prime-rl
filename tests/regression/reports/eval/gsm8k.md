# Eval baseline: gsm8k

`internal/glm-5.3-high-throughput` · 1319 tasks × 1 rollouts · 5 repeats from [2026-10-16](https://github.com/PrimeIntellect-ai/prime-rl/actions/runs/38887591005) at `246360ddf`

Mean ± 95% CI of the mean (Student-t).

| Metric | Mean | ± 95% CI | Std | Min | Max | CV |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 97.65% | 0.22% | 0.18% | 97.50% | 97.95% | 0.2% |
| Time to finish (s) | 1626 | 254 | 204 | 1382 | 1834 | 12.6% |
| Eval window (s) | 1601 | 253 | 203 | 1361 | 1809 | 12.7% |
| Episode p50 (s) | 115.6 | 20.6 | 16.6 | 95.7 | 130.0 | 14.3% |
| Episode p90 (s) | 149.8 | 25.5 | 20.5 | 127.2 | 166.5 | 13.7% |
| Output tokens (mean) | 287 | 18 | 15 | 267 | 303 | 5.1% |
| Error rate | 0.03% | 0.05% | 0.04% | 0.00% | 0.08% | 136.9% |
| Truncation rate | 0.27% | 0.14% | 0.11% | 0.15% | 0.45% | 42.1% |

Within-run accuracy SEM (task sampling): **±0.42%** · observed run-to-run std: **±0.18%**

## Per repeat

| Repeat | Accuracy | ± SEM | Time to finish (s) | Eval window (s) | Episode p50 / p90 (s) | Errors | Truncated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 97.57% | 0.42% | 1708 | 1684 | 95.7 / 127.2 | 0.00% | 0.45% |
| 2 | 97.95% | 0.39% | 1436 | 1408 | 126.1 / 166.5 | 0.00% | 0.23% |
| 3 | 97.57% | 0.42% | 1382 | 1361 | 99.5 / 127.5 | 0.00% | 0.30% |
| 4 | 97.50% | 0.43% | 1771 | 1743 | 126.8 / 165.3 | 0.08% | 0.23% |
| 5 | 97.65% | 0.42% | 1834 | 1809 | 130.0 / 162.4 | 0.08% | 0.15% |
