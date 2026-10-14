# EP all-to-all with an explicit output size: results

Runs from 2026-10-13 on one exclusive 8 x H200 node (job 4084). Every arm ran from a frozen detached worktree
with fresh Inductor and Triton caches. Arms in one wave ran concurrently on disjoint GPUs (see `matrix.sh`,
`traces_and_rl.sh`, `cp_combo.sh`). Raw runs are in `~/tmp/ep-a2a-static-size/runs`, `summary.json` here.

| Tree | Commit | Contents |
|---|---|---|
| main | `7c9d4744c` | main |
| branch | `446445615` | `fix(moe): pass an explicit output size to the EP all-to-all ops` |
| cp-only | `701b087f4` | tip of `fix/cp-attn-params-as-args` (based on `4f4d7e440`) |
| combo | `aab9655cb` | cp-only with `446445615` cherry-picked (detached, on no branch) |

All arms: `samsja/mini-glm-moe`, bf16 router, `ep=2`, no offloading, compile on, 20 steps. Times are the
median `time/forward_backward` over steps 5 to 20 with min and max, in seconds. Lower is better.

## Summary

- The branch is stacked on #4068 and must merge after it (see the cp=2 section). Against #4068 as its parent,
  the step takes 25% less time on Qwen3-30B-A3B, 15% less on GLM-4.5-Air with 8 layers, and 12% less on
  GLM-4.5-Air with cp=2, matching the parent with `fullgraph=true` (first section below).
- With `fullgraph=false`, main breaks the graph twice per MoE block on the all-to-all, so every MoE block runs
  eagerly. The branch has 0 breaks and its step time matches main with `fullgraph=true`.
- End to end, the step takes 25% less time on Qwen3-30B-A3B (5.97 s to 4.48 s) and 14% less on GLM-4.5-Air with
  8 layers (2.96 s to 2.54 s), in both cases matching main with `fullgraph=true` within 0.01 s.
- The mini-glm-moe proxy runs below show a 2x speedup that does not transfer: the model is host bound (the GPU
  is busy 17% of the traced step on main), so removing eager kernel launches shrinks its step far more than on
  a GPU-bound model. Use the proxy only for the graph-break, recompile and CP results.
- Recompiles are unchanged without CP (2 or 3 per frame in every arm, the switch to dynamic shapes).
- With `cp=2`, compiling the MoE blocks exposes the CP attention-params recompiles (`max_seqlen_k` guards):
  64 recompiles and the recompile limit, on the branch with fullgraph off and on main with fullgraph on.
  With fullgraph on, main and branch both crash at the limit. With the CP fix on top (combo), recompiles drop
  to 3 and the step takes 1.54 s against 2.38 s on main. Neither fix alone compiles CP+EP MoE blocks.
- Loss and grad norm match main within the same-code noise floor. Peak memory is identical.

## Stacked on #4068: end-to-end against the parent

Run on 2026-10-14 on one exclusive 8 x H200 node (job 4183), every arm alone and serial (`e2e.sh`, then
`e2e_untraced.sh`). Parent: #4068 at `0c95d3964`. Branch: the stacked fix commit `8cb304fd3`. Same workloads
and settings as the next section (cp=1, `ep=8`, 32k tokens per GPU). Median `time/step` over steps 5 to 20
with min and max, in seconds. Lower is better.

| Arm | Qwen3-30B-A3B step s | Qwen MFU | GLM-4.5-Air 8L step s | GLM MFU |
|---|---|---|---|---|
| parent, fullgraph off | 6.013 (5.961 to 6.018) | 31.4% | 2.977 (2.847 to 3.089) | 31.4% |
| parent, fullgraph off, repeat | not run | | 2.953 (2.828 to 3.084) | 31.4% |
| branch, fullgraph off | 4.531 (4.488 to 4.542) | 41.6% | 2.541 (2.424 to 18.67) | 36.5% |
| branch, fullgraph off, repeat | 4.524 (4.481 to 4.544) | 41.7% | 2.551 (2.456 to 18.16) | 36.3% |
| parent, fullgraph on | 4.529 (4.473 to 4.539) | 41.6% | 2.560 (2.431 to 19.07) | 36.4% |

- Qwen3-30B-A3B: 6.01 s to 4.53 s, 25% less time per step. GLM-4.5-Air 8 layers: 2.97 s to 2.55 s, 15% less.
  The branch matches the parent with fullgraph on in both. The cp=2 rows of the cp=2 section already measure
  this exact code: the stacked fix commit's tree is identical to `8eba3cb67`.
- Every arm is about 0.04 s slower than in the next section on another node, the same across arms.
- The GLM recompile stalls at steps 2, 4, 8 and 14 remain, identical in the branch and the parent with fullgraph
  on, so #4068 does not remove them at cp=1.
- Numerics: curves overlap on both workloads. GLM branch vs parent: 8.1e-3 loss, 7.0e-2 grad norm, inside the
  branch repeat floor (3.2e-2, 2.0e-1). Qwen branch vs parent: 3.8e-3 and 6.0e-2, at the eager vs compiled
  difference that the parent itself shows.
- Dropped to fit the walltime, since startup on this node took several minutes per run: the Qwen parent repeat
  and the trace pairs. The trace analysis and figure in the next section ran on the same fix code against main.

## End-to-end: Qwen3-30B-A3B and GLM-4.5-Air

Run on 2026-10-13 on one exclusive 8 x H200 node (job 4104), every arm alone and serial (`e2e.sh`,
`run_e2e_arm.sh`). SFT, `ep=8`, bf16 router, FA3, no offloading, seq len 16384, batch 16, micro-batch 1, so
32k tokens per GPU per step, 20 steps. Qwen3-30B-A3B is the full model on fake data with balanced routing.
GLM-4.5-Air keeps its first 8 layers (1 dense, 7 MoE) on real SFT data (`INTELLECT-3-SFT-10K`, `math`).
Median `time/step` over steps 5 to 20 with min and max, in seconds. Lower is better.

| Arm | Qwen3-30B-A3B step s | Qwen MFU | GLM-4.5-Air 8L step s | GLM MFU |
|---|---|---|---|---|
| main, fullgraph off | 5.965 (5.924 to 5.992) | 31.6% | 2.957 (2.847 to 3.049) | 31.5% |
| main, fullgraph off, repeat | 5.970 (5.928 to 5.987) | 31.6% | 2.960 (2.837 to 3.065) | 31.7% |
| branch, fullgraph off | 4.484 (4.469 to 4.508) | 42.0% | 2.544 (2.412 to 18.01) | 36.6% |
| branch, fullgraph off, repeat | 4.487 (4.461 to 4.512) | 42.0% | 2.537 (2.425 to 18.34) | 36.4% |
| main, fullgraph on | 4.486 (4.461 to 4.511) | 42.0% | 2.542 (2.409 to 18.49) | 36.5% |

- Qwen3-30B-A3B: 5.97 s to 4.48 s, 25% less time per step. GLM-4.5-Air 8 layers: 2.96 s to 2.54 s, 14% less.
  In both, the branch with fullgraph off matches main with fullgraph on within 0.01 s.
- Peak memory: 76.7 to 76.7 GiB (Qwen) and 56.2 to 55.1 GiB (GLM).
- The GLM maxima are recompile stalls at steps 2, 4, 8 and 14 (10 to 18 s each). Main with fullgraph on stalls at
  the same steps for the same time, so they come from compiled MoE blocks on real variable-length data, not
  from this change. Main with fullgraph off avoids them only because its MoE blocks never compile.
- Numerics: the loss and grad norm curves of all five arms overlap per step on both workloads, with no gap that
  grows. On GLM, compiled runs are less deterministic than eager ones: the branch repeat differs by up to 1.6e-2
  in loss, and main fullgraph on vs off differs by 5.1e-2, more than branch vs main (2.7e-2).

Trace accounting (`step_accounting.py`, all 8 ranks, last clean traced step, ms per step, lower is better):

| Row | Qwen main | Qwen branch | GLM main | GLM branch |
|---|---|---|---|---|
| eager elementwise, copy, index | 2253 | 177 | 659 | 130 |
| Inductor (fused Triton) | 0 | 497 | 0 | 141 |
| GEMMs and flash attention | 2443 | 2567 | 1265 | 1301 |
| exposed communication | 946 | 1003 | 885 | 824 |
| idle and other compute | 390 | 268 | 161 | 143 |
| traced step window | 6032 | 4512 | 2970 | 2538 |
| untraced median step | 5965 | 4484 | 2957 | 2544 |

GPU activities per step drop from 41.7k to 23.8k (Qwen) and 7.2k to 4.2k (GLM). The traced and untraced
deltas agree (Qwen -1520 vs -1481 ms, GLM -432 vs -413 ms). `qwen-layer2-forward.png` shows one layer's
forward: main fills the gaps between GEMMs, attention and all-to-alls with eager elementwise kernels, and the
branch replaces them with a few fused Inductor kernels (layer 2 forward 16.3 ms to 11.0 ms).

## End-to-end with context parallelism: GLM-4.5-Air, cp=2

Run on 2026-10-14 on one exclusive 8 x H200 node (job 4168), every arm alone and serial (`cp_e2e.sh`). Same
GLM-4.5-Air workload as above with `cp=2` (ring), `ep=4`, fullgraph off. All four trees share the base of #4068.

| Tree | Commit | Step s, run 1 | Step s, run 2 | Recompiles | CP guard failures |
|---|---|---|---|---|---|
| base | `4f4d7e440` | 3.62 (3.38 to 3.70) | not run | 3 | 0 |
| #4068 | `0c95d3964` | 3.60 (3.37 to 3.69) | 3.62 (3.42 to 3.68) | 3 | 0 |
| base + this fix | `1ec6594f2` | 53.25 (3.29 to 55.79) | not run | 64, limit hit | 1922 |
| #4068 + this fix | `8eba3cb67` | 3.18 (2.94 to 3.23) | 3.19 (2.94 to 3.28) | 4 | 0 |

- This fix alone is unsafe with CP on real data: steps 2 to 14 take 42 to 56 s each, recompiling on
  `DATA_PARAMS['max_seqlen_k']` and `cu_seqlens` sizes, until the recompile limit at step 15 drops the frames
  back to eager (3.4 s). The mini-glm-moe proxy hit the limit within a few steps and hid this.
- #4068 alone changes nothing here, because the MoE blocks still break the graph on the all-to-all.
- Both together: 3.61 s to 3.19 s, 12% less time per step, 0 all-to-all breaks, 4 recompiles, peak memory 54.2
  to 53.3 GiB. This PR must land after #4068.

## Fake-data RL trainer, 2 GPUs

`run_trainer_arm.sh`, NLL loss (`ce_loss_fn`), seq len 2048, fake batch 8.

| Arm | Graph breaks | Recompiles per frame | fwd+bwd s, run 1 | fwd+bwd s, run 2 |
|---|---|---|---|---|
| main, fullgraph off | 2 (all-to-all) | 2 | 1.90 [1.85, 1.93] | 1.90 [1.83, 2.00] |
| branch, fullgraph off | 0 | 2 | 0.95 [0.88, 0.99] | 0.93 [0.90, 0.98] |
| main, fullgraph on | 0 | 2 | 0.88 [0.88, 0.90] | 1.02 [0.98, 1.16] |
| branch, fullgraph on | 0 | 2 | 0.94 [0.91, 0.99] | 0.92 [0.89, 0.96] |

The fullgraph-on arms trade places between runs, so branch and main are equal there within run-to-run noise.

Numerics (`compare_numerics.py`, max relative difference over 20 steps):

| Pair | Loss | Grad norm |
|---|---|---|
| branch vs main, fullgraph off | 2.3e-05 | 1.1e-03 |
| branch vs main, fullgraph on | 2.1e-05 | 1.2e-03 |
| main vs main repeat (noise floor) | 1.5e-05 | 1.3e-03 |
| branch vs branch repeat (noise floor) | 2.2e-05 | 1.8e-03 |

## SFT, variable-length packed documents

`run_sft_arm.sh`, seq len 2048, batch 8. fwd+bwd in seconds.

| Arm | GPUs | Graph breaks | Recompiles per frame | CP guard failures | fwd+bwd s |
|---|---|---|---|---|---|
| main, cp=1, fullgraph off | 2 | 2 | 3 | 0 | 1.90 [1.85, 2.06] |
| branch, cp=1, fullgraph off | 2 | 0 | 3 | 0 | 0.88 [0.82, 0.92] |
| main, cp=2, fullgraph off | 4 | 2 | 3 | 0 | 2.38 [2.32, 2.88] |
| branch, cp=2, fullgraph off | 4 | 0 | 64, limit hit | 961 | 2.34 [2.31, 49.67] |
| main, cp=2, fullgraph on | 4 | 0 | 64, limit hit | 961 | crash at step 8 |
| branch, cp=2, fullgraph on | 4 | 0 | 64, limit hit | 961 | crash at step 8 |
| cp-only, cp=2, fullgraph off | 4 | 2 | 3 | 0 | 2.29 [2.26, 2.34] |
| combo, cp=2, fullgraph off | 4 | 0 | 3 | 0 | 1.54 [1.45, 1.80] |

The fullgraph-on crashes are `FailOnRecompileLimitHit: Hard failure due to fullgraph=True` after 8 steps of
about 48 s each. Without fullgraph the frames that hit the limit fall back to eager, so the branch with cp=2
matches main's step time instead of crashing. The guard failures are `DATA_PARAMS['max_seqlen_k'] == N` and
`cu_seqlens` sizes, the globals that `fix/cp-attn-params-as-args` turns into forward args.

## RL, reverse-text-moe

`run_rl_arm.sh`, 2 trainer GPUs and 1 inference GPU, fullgraph off, run serially.

| Arm | Graph breaks | Recompiles per frame | fwd+bwd s | step s |
|---|---|---|---|---|
| main | 2 | 3 | 3.34 [3.09, 3.77] | 15.83 [5.98, 21.32] |
| branch | 0 | 3 | 1.71 [1.56, 1.94] | 15.50 [11.30, 26.77] |

Step time here waits on inference, so only the trainer's forward plus backward reflects the change.

## Profiler traces

SFT, cp=1, fullgraph off, 5 steps, last step (4 micro-batches) on rank 0. Analysis commands are in
`~/tmp/profiling/ep-a2a-static-size/commands.sh`.

| Arm | Step window ms | Inductor kernels | All GPU events | Eager elementwise and copy calls |
|---|---|---|---|---|
| main | 2956 | 0 | 40463 | 28597 |
| branch | 1470 | 5328 | 21053 | 7001 |

Over the whole 5-step trace, compiled graphs run 4 times on main and 960 times on the branch (24 layers x 20
micro-batches x forward and backward). `moe-layer2-forward.png` shows layer 2's forward shrinking from 12.8 ms
to 7.0 ms: fused Inductor kernels (+4.6 ms per step) replace eager elementwise, copy and index kernels
(-20.3 ms per step). NCCL kernel time rises by 16.9 ms per step. The model is host bound (GPU busy 17% on main,
36% on the branch), so the all-to-alls most likely spend that time waiting for the peer rank, which the trace
cannot separate from transfer time.

## Not covered

- MXFP8 dispatch (`_mxfp8_all_to_all`) needs Blackwell and was not run.
- The fp32 router keeps MoE blocks eager for an unrelated reason and was not measured.
- GLM-4.5-Air ran with 8 of its 46 layers, so its absolute step times are not full-model times.
