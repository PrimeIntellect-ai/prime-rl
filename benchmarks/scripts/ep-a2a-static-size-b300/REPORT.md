# EP all-to-all explicit output size: B300 A/B against main

One node `primeintellect-tur-r1n6` (8x B300, sm_103), Slurm job 2816, 2026-10-22. Every arm ran 20 steps on this node, serially, with fresh Triton and Inductor caches per run (`launch.sh`). Steady state is steps 5-20.

Arms (code under test, `src/` only):

- main: `fea1267ef`.
- this PR: `fea1267ef` plus the fix commit, `src/` identical to `aefaf6825` (cherry-picked as `dd4d41f5a` on a local exp branch).
- #4068: the rebased #4068 tip `b57635d84`.
- both: `b57635d84` plus the fix commit (`e808aca61` on a local exp branch).

Workloads (`cfg/`), all ep 8, full AC, default compile (`fullgraph = false`), no offloading:

- Qwen3-30B-A3B (Thinking-2507) cut to 12 layers, fake data, balanced routing, 16 x 16384 tokens per step.
- `samsja/mini-glm-moe`, fake data, 16 x 8192 tokens per step.
- GLM-4.5-Air cut to 8 layers, real variable-length SFT data (`INTELLECT-3-SFT-10K` math), 16 x 16384 tokens per step, cp 1 and cp 2.
- MXFP8 expert compute and transport on the Qwen3-30B-A3B workload. Transport alone is rejected by config validation (`MXFP8 transport requires model.moe.compute.type='mxfp8'`), so the `mxt-*` cells in `series.sh` exited at startup and are not reported.

Breaks and recompiles come from every rank's `TORCH_LOGS=graph_breaks,recompiles` output (`summarize.py`). The only graph break on main is the EP all-to-all inside the activation checkpoint (`prime_rl_collectives.all_to_all_single` or `mxfp8_all_to_all`, "output shape depends on input Tensor data"), which sends every checkpointed MoE block to eager.

## Results

| run | step med (min-max) s | peak GiB | loss 1 | loss 20 | breaks | recompiles per rank |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B, main | 1.067 (1.063-1.075) | 32.7 | 16.5100 | 10.4700 | 1 | 0 |
| Qwen3-30B-A3B, this PR | 0.703 (0.700-0.708) | 32.6 | 16.5145 | 10.4750 | 0 | 0 |
| mini-glm-moe, main | 1.066 (1.049-1.085) | 14.3 | 11.2935 | 11.8952 | 1 | 1 |
| mini-glm-moe, this PR | 0.542 (0.533-0.565) | 14.1 | 11.3060 | 11.8914 | 0 | 1 |
| GLM-4.5-Air cp 1, main | 1.892 (1.834-1.930) | 55.7 | 11.2368 | 3.3974 | 1 | 1 |
| GLM-4.5-Air cp 1, this PR | 1.427 (1.397-1.486) | 55.1 | 11.2368 | 3.4142 | 0 | 1 |
| MXFP8, main | 1.093 (1.089-1.107) | 32.7 | 16.4994 | 10.4058 | 1 | 0 |
| MXFP8, main with `fullgraph = true` | 0.716 (0.713-0.727) | 32.6 | 16.4987 | 10.3958 | 0 | 0 |
| MXFP8, this PR | 0.716 (0.710-0.723) | 32.6 | 16.4994 | 10.4224 | 0 | 0 |

GLM-4.5-Air cp 2, real data, crossing this PR with #4068:

| run | step med (min-max) s | loss 20 | breaks | recompiles per rank | `ULYSSES_PARAMS` guard failures |
| --- | --- | --- | --- | --- | --- |
| main | 2.246 (2.183-2.344) | 3.2320 | 1 | 1 | 0 |
| #4068 | 2.231 (2.191-2.339) | 3.2379 | 1 | 1 | 0 |
| this PR | 1.718 (1.672-8.126) | 3.2660 | 0 | 3 | 2 |
| both | 1.702 (1.664-1.819) | 3.2623 | 0 | 1 | 0 |

- Without #4068, cp 2 recompiles once more on every rank in the same step (step 5, 8.1 s), on the `ULYSSES_PARAMS['cu_seqlens']` length guard, and then runs flat. There is no recompile storm: that came from ring CP's per-step parameters, which main removed in #4134. With #4068 underneath, the extra recompile is gone.
- Loss: all arms of a workload agree within 0.003 for steps 1-9. On the GLM-4.5-Air data, step 10 is a sensitive batch where two runs of identical code differed by 0.43 in an earlier repeat, and later gaps of 0.01-0.05 are the same size as that repeat spread. Step 1 is not bitwise reproducible across arms because compiled and eager MoE blocks round differently.
- MXFP8 runs with the pinned `prime-kernels` (`3101960`), whose `mxfp8_moe` kernel loads on sm_103.

GPU unit tests on this node: see `gpu_tests.sh` and the PR description.

## Reproduce

`series.sh` (bf16 sanity, MXFP8, cp 2 for this PR) and `series2.sh` (main reruns, #4068, both) call `launch.sh <tree> <run> [env] -- <sft args>` inside a node allocation; `results.md` is `summarize.py runs <run...>` over the run dirs.
