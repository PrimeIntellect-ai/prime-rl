# Eval baseline: pi-terminalbench2-1

`internal/deepseek-v4.1-flash-1017-ci` · 82 tasks × 1 rollouts · 5 repeats from [2026-11-04](https://github.com/PrimeIntellect-ai/prime-rl/actions/runs/41763261279) at `791ba2d29`

Mean ± 95% CI of the mean (Student-t).

| Metric | Mean | ± 95% CI | Std | Min | Max | CV |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 65.85% | 3.56% | 2.86% | 63.41% | 70.73% | 4.3% |
| Time to finish (s) | 14475 | 14 | 11 | 14460 | 14486 | 0.1% |
| Eval window (s) | 14415 | 14 | 11 | 14400 | 14426 | 0.1% |
| Episode p50 (s) | 512.8 | 94.2 | 75.8 | 412.5 | 592.9 | 14.8% |
| Episode p90 (s) | 8037.9 | 1824.3 | 1467.3 | 6094.5 | 9916.6 | 18.3% |
| Output tokens (mean) | 35750 | 3616 | 2909 | 32742 | 40321 | 8.1% |
| Error rate | 1.95% | 0.83% | 0.67% | 1.22% | 2.44% | 34.2% |
| Truncation rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | – |

Within-run accuracy SEM (task sampling): **±5.26%** · observed run-to-run std: **±2.86%**

## Per repeat

| Repeat | Accuracy | ± SEM | Time to finish (s) | Eval window (s) | Episode p50 / p90 (s) | Errors | Truncated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.63% | 5.31% | 14486 | 14426 | 565.3 / 9916.6 | 2.44% | 0.00% |
| 2 | 64.63% | 5.31% | 14471 | 14411 | 592.9 / 6094.5 | 2.44% | 0.00% |
| 3 | 65.85% | 5.27% | 14460 | 14400 | 456.6 / 8051.2 | 2.44% | 0.00% |
| 4 | 70.73% | 5.06% | 14471 | 14411 | 412.5 / 7256.0 | 1.22% | 0.00% |
| 5 | 63.41% | 5.35% | 14486 | 14426 | 536.8 / 8871.0 | 1.22% | 0.00% |
