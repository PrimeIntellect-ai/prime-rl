# Eval baseline: gsm8k

`internal/deepseek-v4.1-flash-1017-ci` · 1319 tasks × 1 rollouts · 5 repeats from [2026-11-03](https://github.com/PrimeIntellect-ai/prime-rl/actions/runs/41725025527) at `76ace8e64`

Mean ± 95% CI of the mean (Student-t).

| Metric | Mean | ± 95% CI | Std | Min | Max | CV |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 96.95% | 0.36% | 0.29% | 96.59% | 97.35% | 0.3% |
| Time to finish (s) | 701 | 117 | 94 | 539 | 770 | 13.4% |
| Eval window (s) | 676 | 116 | 93 | 515 | 746 | 13.8% |
| Episode p50 (s) | 58.2 | 8.9 | 7.1 | 45.8 | 62.8 | 12.2% |
| Episode p90 (s) | 67.2 | 9.3 | 7.5 | 54.2 | 72.4 | 11.2% |
| Output tokens (mean) | 169 | 2 | 2 | 166 | 171 | 1.2% |
| Error rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | – |
| Truncation rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | – |

Within-run accuracy SEM (task sampling): **±0.47%** · observed run-to-run std: **±0.29%**

## Per repeat

| Repeat | Accuracy | ± SEM | Time to finish (s) | Eval window (s) | Episode p50 / p90 (s) | Errors | Truncated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96.82% | 0.48% | 745 | 719 | 60.6 / 69.1 | 0.00% | 0.00% |
| 2 | 97.35% | 0.44% | 751 | 724 | 62.8 / 72.4 | 0.00% | 0.00% |
| 3 | 97.12% | 0.46% | 539 | 515 | 45.8 / 54.2 | 0.00% | 0.00% |
| 4 | 96.89% | 0.48% | 770 | 746 | 62.8 / 71.9 | 0.00% | 0.00% |
| 5 | 96.59% | 0.50% | 701 | 676 | 59.1 / 68.5 | 0.00% | 0.00% |
