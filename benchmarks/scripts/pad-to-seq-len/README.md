# pad_to_seq_len e2e checks

Commit under test: `45896d4a0`. One exclusive H200 node per run (SLURM job 5739), launched with `run.sh` from the repo root on the node, 2026-10-20. The bf16 runs were named `pad-seq-rl` and `pad-seq-sft`. Each run used an empty DeepGEMM cache. "Late" counts DeepGEMM JIT compiles (cache entry mtimes) after step 1 ended. Dynamo recompiles come from `TORCH_LOGS=recompiles` on every rank. Step times are trainer step times in seconds, lower is better.

| Run | Dynamo recompiles per rank | DeepGEMM compiles, step 1 / late | Steps 2+ median (min to max) |
| --- | --- | --- | --- |
| SFT GLM-4.5-Air 4L, FP8, flag on | 1 (step 1, layer type) | 22 / 0 | 1.7 s (1.6 to 2.0) |
| SFT GLM-4.5-Air 4L, FP8, flag off | 3 (step 1, incl. length guard) | 27 / 7 | 1.7 s (1.6 to 13.9) |
| RL Qwen3-30B-A3B, replay, FP8, flag on | 0 | 17 / 0 | 11.5 s (9.8 to 12.1) |
| RL Qwen3-30B-A3B, replay, FP8, flag off | 1 (step 1, length guard) | 26 / 28 | 20.7 s (8.6 to 55.6) |

bf16 runs with the flag on: SFT (10 steps) had only the step-1 layer-type recompile, and RL with replay (15 steps) had none on any rank, with mismatch KL 0.0001 to 0.0009.

Flag-off RL compiled through its last step, so it never reached a compile-free window. Its fastest steps (8.6 to 9.4 s) beat flag-on (about 10 to 12 s), which is the cost of the extra padding. These are short runs on cold caches, so they show the mechanism, not a steady-state speedup.

W&B: `primeintellect/feat-rl-pad-to-seq-len`.
