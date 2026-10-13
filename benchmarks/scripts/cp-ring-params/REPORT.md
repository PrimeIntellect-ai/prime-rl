# CP attention params as forward args: results

Runs from 2026-10-13 on exclusive H200 nodes (jobs 3933, 3934, 4007 and 4023). Every arm ran from a frozen
detached worktree with a fresh Inductor cache. Scripts are in this directory, raw runs are in
`~/tmp/cp-ring-params/runs`.

| Tree | Commit | Contents |
|---|---|---|
| branch | `b13b09c08` | `fix(cp): pass ring attention params as forward args` |
| main | `4f4d7e440` | parent of the branch |

## Summary

- Ring CP with `fullgraph=True` and a bf16 router crashes on main at the recompile limit. The branch runs it,
  at 4.38 s forward plus backward.
- The branch removes the per-micro-batch recompiles wherever blocks compile under CP, on FA2, FA3 and FA4.
- With `fullgraph=False`, reverse-text-moe runs its MoE blocks eagerly, so branch and main perform the same.
- Numerics differ from main by the same order as same-code repeats (about 2x at most, no growth over 20 steps)
  for ring, ulysses and cp1, compile off and on.

## Fullgraph with a bf16 router

bf16 router, `ep=2`, offloading off (job 4007). With `fullgraph=False` the EP `all_to_all` (a custom op whose
fake returns a data-dependent size) breaks the graph inside the checkpoint, so MoE blocks run eagerly.
`fullgraph=True` also enables capture of data-dependent output shapes (`torch/_dynamo/output_graph.py:735`,
`allow_dynamic_output_shape_ops=one_graph or ...`), so the same blocks compile with no breaks. The fake-data
control shows it directly: compile on with fullgraph off gives 2 breaks at `all_to_all_single`, fullgraph on
gives 0.

Recompiles are counted on trainer rank 0. "CP guards" counts guard failures that mention `max_seqlen`,
`cu_seqlens`, `local_k` or `_PARAMS`.

| Arm (fullgraph on) | Tree | Steps | Recompiles | CP guards | Fwd+bwd median |
|---|---|---|---|---|---|
| RL reverse-text-moe, ring cp2 | main | crash after 3 (recompile limit) | 64 | 961 | 111.3 s |
| RL reverse-text-moe, ring cp2 | branch | 6/6 | 3 | 0 | 4.38 s |
| RL reverse-text-moe, cp1 | main | 6/6 | 3 | 0 | 1.62 s |
| Fake-data trainer, ring cp2 | main | crash at step 4 (recompile limit) | 115 | `DATA_PARAMS['max_seqlen_k']` | |
| Fake-data trainer, ring cp2 | branch | 20/20 | 2 | 0 | |
| Fake-data trainer, cp1 | main and branch | 20/20 | 2 | 0 | |

For comparison, the same RL ring cp2 arm with fullgraph off (MoE blocks eager) takes 7.71 s forward plus
backward.

## Every FlashAttention backend

`probe_fa.py`: Llama, 2 layers, AC plus `fullgraph=True`, 8 micro batches with different document layouts, 2
ranks (job 4023). FA4 ran through its SM90 kernels on H200. Recompiles summed over both ranks.

| Tree | FA2 ring | FA3 ring | FA4 ring | FA2 ulysses | FA3 ulysses | FA4 ulysses |
|---|---|---|---|---|---|---|
| branch | 0 | 0 | 0 | 0 | 0 | 0 |
| main | | | 11 | | 2 | |

Main's FA4 ring recompiles on `DATA_PARAMS['max_seqlen_k']` and the `cu_seqlens_k` size (8 micro batches
stay under the limit, so it does not crash here). Main's FA3 ulysses recompiles once per rank on the
`ULYSSES_PARAMS['cu_seqlens']` size, then that size is symbolic. A static analysis of the three backends found
no backend-specific traced code: the backend reaches the ring op only as a constant string.

The same probe with fullgraph off (`probe.py` and `probe_main.py`, FA2 ring): main recompiles 11 times, the
branch 0 times, and the branch without the new `dynamic_sources` patterns once per rank on
`ring_params.max_seqlen_k` before it turns symbolic.

## Performance without fullgraph

reverse-text-moe, 2 trainer GPUs and 1 inference GPU, 20 steps, no offloading of any kind, no checkpoints.
Trainer `time/forward_backward`, median with min and max over steps 5 to 20, in seconds. Lower is better.
`time/step` is not used because it includes waiting for rollouts.

| Arm | Branch | Main |
|---|---|---|
| bf16 router, ring cp2 | 7.71 (7.04 to 9.08) | 8.25 (7.03 to 8.83) |
| bf16 router, ring cp2, repeat | 7.86 (6.99 to 8.92) | 7.94 (6.92 to 8.58) |
| bf16 router, ulysses cp2 | 7.41 (6.91 to 8.11) | 7.32 (6.47 to 8.30) |
| bf16 router, cp1 (dp2) | 3.62 (3.21 to 3.86) | 3.55 (3.03 to 3.69) |
| fp32 router, ring cp2 | 8.54 (8.07 to 9.41) | 8.73 (7.71 to 9.76) |
| fp32 router, ulysses cp2 | 7.98 (7.06 to 9.04) | 7.95 (7.60 to 8.90) |

Both trees show the same 3 recompiles per arm: two for the first sequence length change and one for the dense
first layer. The MoE blocks run eagerly here, so they never read the CP globals inside a compiled frame.
Branch and main differ by no more than the repeats of the same code (main ring 8.25 s, then 7.94 s). After
step 2 no step exceeds 1.2 times its arm's median. Peak memory is 6.3 GiB in every arm.

## Numerics

Fake-data trainer runs on mini-glm-moe, 20 steps, 8 micro batches of 2048 tokens packed from documents of up
to 256 tokens, so document boundaries fall inside CP shards. cp1 runs on one GPU so every arm sees the same
batches. The fake batches carry random inference logprobs, which zero out importance-weighted RL losses, so
these arms train on masked NLL (`ce_loss_fn`). Relative differences, max over all 20 steps.

| Comparison | Loss | Grad norm |
|---|---|---|
| Same code repeated | 1.1e-5 to 1.5e-5 | 1.1e-3 to 1.2e-3 |
| Branch vs main, ring, ulysses and cp1, compile off and on | 1.9e-5 to 2.7e-5 | 1.1e-3 to 1.5e-3 |
| Branch cp2 vs branch cp1 | 1.9e-5 to 2.6e-5 | 1.0e-3 to 1.4e-3 |

Step 1 is not bitwise reproducible even with identical code (about 1e-5 on loss), so an exact match gate is
not achievable on this stack. Branch vs main reaches about twice the repeat spread, measured from only two
repeat pairs, and does not grow over steps. The same arms with the default IPO loss (loss near zero) agree on
entropy and mismatch KL to 1e-5 to 4e-5. Script: `compare_numerics.py`.

## Coverage

- SFT, ring cp2, fake variable-length packs, compile on: 10/10 steps, 1 recompile, no CP guards.
- Qwen3.5 VLM deferred ulysses path (`vlm_ulysses_check.py`): two packed documents with one image each, cp2
  logits match the no-CP reference exactly on both ranks, on branch and main. The RL VLM smoke run
  (color-codeword, Qwen3.5-4B) failed in the orchestrator before any batch reached the trainer
  (`Inline image count does not match expanded placeholder runs`, `trajectories.py:_build_mm_refs`). The
  branch does not touch orchestrator code.
- VLMs reject ring CP at config time (`VLM models require cp_style='ulysses'`), so the ring deferred VLM path
  is unreachable.

## Unrelated bugs found along the way

- FA2 ring skips the head-dim padding to a multiple of 8 that non-CP FA2 applies.
- SFT ring CP with image batches shards the input twice and fails at the `seq_lens` check.
- GPT-OSS calls FA4 directly because the FA4 custom op has no learned-sink argument, so its layers break the
  graph at attention and cannot use fullgraph.
- The color-codeword orchestrator error above.
