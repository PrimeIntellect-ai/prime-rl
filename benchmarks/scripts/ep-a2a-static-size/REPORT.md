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

- With `fullgraph=false`, main breaks the graph twice per MoE block on the all-to-all, so every MoE block runs
  eagerly. The branch has 0 breaks and runs the forward plus backward in half the time, matching main with
  `fullgraph=true`.
- Recompiles are unchanged without CP (2 or 3 per frame in every arm, the switch to dynamic shapes).
- With `cp=2`, compiling the MoE blocks exposes the CP attention-params recompiles (`max_seqlen_k` guards):
  64 recompiles and the recompile limit, on the branch with fullgraph off and on main with fullgraph on.
  With fullgraph on, main and branch both crash at the limit. With the CP fix on top (combo), recompiles drop
  to 3 and the step takes 1.54 s against 2.38 s on main. Neither fix alone compiles CP+EP MoE blocks.
- Loss and grad norm match main within the same-code noise floor. Peak memory is identical.

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
- All results are on the 24-layer mini-glm-moe proxy, not a full-size model.
