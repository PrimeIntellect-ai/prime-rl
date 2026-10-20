# pad_to_seq_len e2e checks

Commit under test: `45896d4a0`. One exclusive H200 node per run (SLURM job 5739), launched with `run.sh` from the repo root on the node, 2026-10-20. The bf16 runs were named `pad-seq-rl` and `pad-seq-sft`. Each run used an empty DeepGEMM cache. Dynamo recompiles come from `TORCH_LOGS=recompiles` on every rank. DeepGEMM JIT compiles are cache entries, timed by mtime against step boundaries. All four runs are FP8 (`quantization.type=fp8`, `moe.compute.type=deepgemm_fp8`).

| Run | Dynamo recompiles per rank, total (after step 1) | DeepGEMM compiles, total (after step 1) |
| --- | --- | --- |
| SFT GLM-4.5-Air 4L `--data.pad-to-seq-len` | 1 (0) | 22 (0) |
| SFT GLM-4.5-Air 4L | 3 (0) | 34 (7) |
| RL Qwen3-30B-A3B + replay `--orchestrator.pad-to-seq-len` | 0 (0) | 17 (0) |
| RL Qwen3-30B-A3B + replay | 1 (0) | 54 (28) |

bf16 runs with the flag: SFT (10 steps) had only the step-1 layer-type recompile, and RL with replay (15 steps) had none on any rank, with mismatch KL 0.0001 to 0.0009.

Step times are left out on purpose: in runs this short, compile stalls dominate them and overstate the flag's benefit.

W&B: `primeintellect/feat-rl-pad-to-seq-len`.
