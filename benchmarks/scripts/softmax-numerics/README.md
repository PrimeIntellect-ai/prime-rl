# Softmax numerics: precision, memory and speed of the LM head

Scripts and results behind the numbers in the softmax-numerics PR. All runs used one H200 node.

## Precision near probability 1

`precision_sweep.py` puts a label at the top logit, sweeps how much probability the other tokens hold (down to
`1e-44`), the size of the largest logit (0 to 40) and the temperature (1.0 and 0.6), and compares every logprob
and cross-entropy site with an fp64 reference. `results/precision_main.txt` is main at `c0d3e7034`,
`results/precision_branch.txt` is this branch at `1374c16e9`. Columns: the scaled margin (log-odds of the label
against all other tokens) where the logprob and the label gradient first exceed 50% relative error, the largest
logprob error below a margin of 10, and how far the gradient is from summing to zero.

## Memory and speed

`memory_time_bench.py` times one LM-head forward+backward and records its peak GPU memory, for the Qwen3-1.7B head
(hidden 2048, vocab 151936, bf16 weights, token chunk 8192). Arms, run one after another on one GPU:

- `main`: main's code (`f74cb4902`, whose LM head equals `c0d3e7034`).
- `main, same compile`: main's code with `main_compiled.patch`, which compiles main's per-chunk math the way this
  branch compiles its own.
- `this branch`: `1374c16e9` (before the quack removal, so `sft_quack` still ran quack).

Peak GB / ms per forward+backward at 16384 tokens (lower is better):

| Path | main | main, same compile | this branch |
|---|---|---|---|
| `rl_full_vocab` | 2.0 / 134 | 1.5 / 99 | 1.5 / 107 |
| `rl_support_not_replayed` | 2.2 / 164 | 1.6 / 127 | 1.6 / 136 |
| `rl_replay_dense` | 2.2 / 162 | 1.6 / 125 | 1.6 / 134 |
| `rl_replay_sparse` | 0.8 / 57 | 0.7 / 33 | 0.7 / 35 |
| `sft_quack` | 8.2 / 52 | 8.2 / 51 | 8.2 / 51 |
| `sft_fallback` | 10.6 / 81 | 13.1 / 65 | 8.2 / 67 |
| `vanilla_rl` | 29.9 / 88 | 29.9 / 88 | 19.9 / 85 |
| `vanilla_rl_support` | 49.8 / 122 | 49.8 / 122 | 29.9 / 99 |
| `vanilla_sft` | 29.9 / 64 | 29.9 / 65 | 19.9 / 63 |

Peak GB / ms per forward+backward at 32768 tokens (lower is better):

| Path | main | main, same compile | this branch |
|---|---|---|---|
| `rl_full_vocab` | 2.1 / 272 | 1.6 / 201 | 1.6 / 214 |
| `rl_support_not_replayed` | 2.2 / 327 | 1.7 / 254 | 1.7 / 271 |
| `rl_replay_dense` | 2.2 / 324 | 1.7 / 250 | 1.7 / 269 |
| `rl_replay_sparse` | 0.8 / 111 | 0.8 / 63 | 0.8 / 68 |
| `sft_quack` | 8.2 / 103 | 8.2 / 102 | 8.2 / 102 |
| `sft_fallback` | 10.7 / 165 | 13.2 / 130 | 8.2 / 134 |
| `vanilla_rl` | 59.7 / 180 | 59.7 / 181 | 39.8 / 173 |
| `vanilla_rl_support` | 99.6 / 243 | 99.6 / 247 | 59.8 / 202 |
| `vanilla_sft` | 59.7 / 134 | 59.7 / 134 | 39.8 / 131 |

Peak GB / ms per forward+backward at 65536 tokens (lower is better):

| Path | main | main, same compile | this branch |
|---|---|---|---|
| `rl_full_vocab` | 2.2 / 546 | 1.7 / 408 | 1.7 / 439 |
| `rl_support_not_replayed` | 2.4 / 655 | 1.8 / 508 | 1.8 / 543 |
| `rl_replay_dense` | 2.4 / 647 | 1.8 / 500 | 1.8 / 537 |
| `rl_replay_sparse` | 1.0 / 220 | 1.0 / 125 | 1.0 / 133 |
| `sft_quack` | 8.4 / 207 | 8.4 / 206 | 8.4 / 206 |
| `sft_fallback` | 10.8 / 334 | 13.3 / 264 | 8.4 / 268 |
| `vanilla_rl` | 119.5 / 372 | 119.5 / 368 | 79.7 / 359 |
| `vanilla_rl_support` | OOM | OOM | 119.5 / 421 |
| `vanilla_sft` | 119.5 / 274 | 119.5 / 274 | 79.7 / 285 |

## RL runs

`rl_ab.sh <arm> <cp>` runs 25 steps of `examples/basic/hendrycks-sanity` (Qwen3-1.7B, 4 trainer and 4 inference
GPUs), and `analyze_rl_run.py <run dir>` prints the per-step metrics in `results/rl_*.txt`. main ran at
`f74cb4902` and this branch at `e15f5d3b5`. Both commits also log `logprob_collapse_regime`, the share of sampled
tokens with probability within `2^-24` of 1.
