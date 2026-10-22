# CP attention params as forward args: B300 A/B against main

One node `primeintellect-tur-r1n6` (8x B300, sm_103), Slurm job 2816, 2026-10-22. Every arm ran 20 steps on this node, serially, with fresh Triton and Inductor caches per run (`launch.sh`). Steady state is steps 5-20.

Arms (code under test, `src/` only):

- main: `fea1267ef`.
- this PR: `b57635d84`.
- #4086: `fea1267ef` plus #4086's fix commit, `src/` identical to #4086's `aefaf6825`.
- both: `b57635d84` plus #4086's fix commit.

Workload (`cfg/`): GLM-4.5-Air cut to 8 layers, ep 8, cp 2 (Ulysses), real variable-length SFT data (`INTELLECT-3-SFT-10K` math), 16 x 16384 tokens per step, full AC, default compile (`fullgraph = false`), no offloading. Real data is required: each packed row has a different document layout, so `cu_seqlens` changes shape between micro batches. Breaks, recompiles and guard failures come from every rank's `TORCH_LOGS=graph_breaks,recompiles` output (`summarize.py`).

## Results

| run | step med (min-max) s | loss 20 | breaks | recompiles per rank | `ULYSSES_PARAMS` guard failures |
| --- | --- | --- | --- | --- | --- |
| main | 2.246 (2.183-2.344) | 3.2320 | 1 | 1 | 0 |
| this PR | 2.231 (2.191-2.339) | 3.2379 | 1 | 1 | 0 |
| #4086 | 1.718 (1.672-8.126) | 3.2660 | 0 | 3 | 2 |
| both | 1.702 (1.664-1.819) | 3.2623 | 0 | 1 | 0 |

- On main, the EP all-to-all breaks the graph inside the activation checkpoint, so MoE blocks run eagerly and only the dense first block compiles. Its guards on the `ULYSSES_PARAMS` global never fail here, and this PR alone matches main within noise.
- Once #4086 makes the MoE blocks compile, the compiled blocks guard on the length of the global `ULYSSES_PARAMS['cu_seqlens']`. A batch with a different document count fails that guard and recompiles every rank (step 5, 8.1 s). This PR passes `cu_seqlens` and `max_seqlen` as arguments instead, so that recompile and its slow step are gone (max step 1.82 s).
- Loss: all arms agree within 0.003 for steps 1-9. On this data, step 10 is a sensitive batch where two runs of identical code differed by 0.43 in an earlier repeat on another node, and the later gaps of 0.01-0.05 are the same size as that repeat spread.

## Reproduce

`series2.sh` calls `launch.sh <tree> <run> [env] -- <sft args>` inside a node allocation for these four arms (plus the #4086 description's main reruns); `results.md` is `summarize.py runs <run...>` over the run dirs.
