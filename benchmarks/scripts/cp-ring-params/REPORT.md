# CP attention params as forward args: results

Runs from 2026-10-13 on two exclusive H200 nodes (jobs 3933 and 3934). Every arm ran from a frozen detached
worktree with a fresh Inductor cache. Scripts are in this directory, raw runs are in `~/tmp/cp-ring-params/runs`.

| Tree | Commit | Contents |
|---|---|---|
| branch | `b13b09c08` | `fix(cp): pass ring attention params as forward args` |
| main | `4f4d7e440` | parent of the branch (main after the #4037 revert) |
| e3925 | `29e55e87f` | branch with #4037 reverted, which restores the #3925 `AC(compile(block))` layout |
| main3925 | `3827e8e0d` | main with #4037 reverted |

## Summary

- The branch removes the per-micro-batch recompiles on every layout where blocks compile. On the #3925
  layout without activation offloading, main recompiles 143 times, hits the recompile limit twice and takes
  61.9 s per step. The branch recompiles 17 times, none on CP fields, and takes 10.6 s per step.
- Numerics match main within the same-code noise floor for ring, ulysses and cp1, compile off and on.
- On main's current `compile(AC(block))` layout, reverse-text-moe runs its MoE blocks eagerly, so branch and
  main perform the same there.
- The #3925 layout with activation offloading still fails for ring (E0), now on a checkpoint replay error on
  one rank. It passes once `dynamic_sources` is visible to the autograd thread (see E0 below).

## Recompiles

Recompiles are counted on trainer rank 0. "CP guards" counts guard failures that mention `max_seqlen`,
`cu_seqlens`, `local_k` or `_PARAMS`.

| Arm (ring cp2, mini-glm-moe, 10 steps) | Tree | Recompiles | CP guards | Limit hits | Median step |
|---|---|---|---|---|---|
| E3: #3925 layout, no AC offload | main3925 | 143 | 4034 | 2 | 61.86 s |
| E3: #3925 layout, no AC offload | e3925 | 17 (max 2 per frame) | 0 | 0 | 10.56 s |
| E0: #3925 layout, AC offload on | main3925 | 27 (4 on autograd thread) | 30 | 1 | crash in step 1 |
| E0: #3925 layout, AC offload on | e3925 | 19 (2 on autograd thread) | 1 | 0 | rank 0 crash in step 1 |
| E0 with `TORCH_COMPILE_DYNAMIC_SOURCES` set | e3925 | 18 (1 on autograd thread) | 0 | 0 | 10.40 s, 10/10 |
| E2: #3925 layout, ulysses, AC offload on | e3925 | 19 (2 on autograd thread) | 1 | 0 | 10.74 s, 10/10 |

Lower is better in every column. All rows ran on the same node. Ulysses on main3925 was not rerun: in the
original investigation (`~/tmp/ac-compile-rootcause/REPORT.md`) it crashed with `aten.empty.memory_format`.

Probe on a dense model, where every block compiles under main's `compile(AC(block))` layout (Llama, 2 layers,
ring cp2, 8 micro batches with different document layouts, `probe.py` and `probe_main.py`):

| Tree | Recompiles | Reason |
|---|---|---|
| main | 11 | `DATA_PARAMS['max_seqlen_k'] == ...` and `DATA_PARAMS['cu_seqlens_k']` size, per micro batch |
| branch | 0 | |
| branch without the new `dynamic_sources` patterns | 1 per rank | `ring_params.max_seqlen_k`, then symbolic |

On reverse-text-moe under main's layout both trees show the same 3 recompiles: two for the first sequence
length change and one for the dense first layer. The MoE blocks break the graph inside the checkpoint (EP
`all_to_all` has a data-dependent output shape), so they run eagerly and never read the CP globals inside a
compiled frame. That is why this workload shows no `DATA_PARAMS` churn on main.

## Performance on main's layout

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

Branch and main differ by no more than the repeats of the same code (main ring 8.25 s, then 7.94 s). After
step 2 no step exceeds 1.2 times its arm's median. Peak memory is 6.3 GiB in every arm.

## Numerics

Fake-data trainer runs on mini-glm-moe, 20 steps, 8 micro batches of 2048 tokens packed from documents of up
to 256 tokens, so document boundaries fall inside CP shards. cp1 runs on one GPU so every arm sees the same
batches. The fake batches carry random inference logprobs, which zero out importance-weighted RL losses, so
these arms train on masked NLL (`ce_loss_fn`). Relative differences, max over all 20 steps.

| Comparison | Loss | Grad norm |
|---|---|---|
| Same code repeated (noise floor) | 1.1e-5 to 1.5e-5 | 1.1e-3 to 1.2e-3 |
| Branch vs main, ring, ulysses and cp1, compile off and on | 1.9e-5 to 2.7e-5 | 1.1e-3 to 1.5e-3 |
| Branch cp2 vs branch cp1 | 1.9e-5 to 2.6e-5 | 1.0e-3 to 1.4e-3 |

Step 1 is not bitwise reproducible even with identical code (about 1e-5 on loss), so an exact match gate is
not achievable on this stack. Parity is judged against the noise floor. The same arms with the default IPO
loss (loss near zero) agree on entropy and mismatch KL to 1e-5 to 4e-5. Script: `compare_numerics.py`.

## Coverage

- SFT, ring cp2, fake variable-length packs, compile on: 10/10 steps, 1 recompile, no CP guards.
- Qwen3.5 VLM deferred ulysses path (`vlm_ulysses_check.py`): two packed documents with one image each, cp2
  logits match the no-CP reference exactly on both ranks, on branch and main. The RL VLM smoke run
  (color-codeword, Qwen3.5-4B) failed in the orchestrator before any batch reached the trainer
  (`Inline image count does not match expanded placeholder runs`, `trajectories.py:_build_mm_refs`). The
  branch does not touch orchestrator code.
- VLMs reject ring CP at config time (`VLM models require cp_style='ulysses'`), so the ring deferred VLM path
  is unreachable.

## E0: what still fails on the #3925 layout

With activation offloading on, rank 0 raises `aten.empty.memory_format encountered during backward but not
found in storage`. Faulthandler stacks show rank 1 waiting inside the recompute's MoE token dispatch while
rank 0 sits in `destroy_process_group`, so the run hangs until the 900 s NCCL watchdog fires. The error comes
from Inductor autotuning a freshly compiled graph inside the checkpoint replay. Two things put a fresh graph
there:

1. Activation offloading unpacks a non-view copy, so the forward's `_base` guard fails and the recompute
   compiles its own cache entry on the autograd thread.
2. Dynamo config is thread-local in torch 2.13, so that first autograd-thread compile does not see
   `dynamic_sources` and specializes `ring_params.max_seqlen_k`. The next micro batch compiles once more.

Setting `TORCH_COMPILE_DYNAMIC_SOURCES` makes the patterns the process-wide default, visible on every
thread. Then only the first compile happens and E0 completes 10/10. Ulysses (E2) already passes with both
compiles.

## Follow-ups (not in this branch)

- Make `dynamic_sources` visible to the autograd thread, for example through the environment default,
  before re-landing #3925.
- GPT-OSS calls FA4 directly because the FA4 custom op has no learned-sink argument, so its layers break the
  graph at attention.
- The color-codeword orchestrator error above.
