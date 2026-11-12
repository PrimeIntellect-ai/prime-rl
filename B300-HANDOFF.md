# DeepSeek V4 performance stack: B300 handoff

This branch (`exp/dsv4-stack-b300`) sits on top of a five-PR stack of DeepSeek V4 (DSv4) trainer performance changes. The stack was audited and fixed for correctness on H200. The work left is to clean up the H200-only pieces, re-verify correctness on B300 (SM100), and measure what each PR is worth on B300. Correctness first. This file and the checks directory are the only things this branch adds; drop the branch once the work below is done.

## The stack

| PR | branch | what it does |
|---|---|---|
| #4354 | feat/dsv4-cudnn-compressor | pools the compressors' softmax-gated windows with cuDNN's fused CSA/HCA kernels |
| #4355 | feat/dsv4-cp-halo | under CP, exchanges a small halo with the left neighbor instead of all-gathering the row |
| #4356 | feat/dsv4-qnorm-rope | runs vLLM's fused q norm + RoPE, and fuses the indexer's q RoPE + quantization |
| #4357 | feat/dsv4-mhc-kernels | rewrites the mHC write-back backward, pads hc_head for DeepGEMM, TF32 pre-pass backward |
| #4358 | feat/dsv4-sac-module-targets | lets selective AC targets name a module path (`<op>@<module pattern>`) |

Each branch is stacked on the one above it, starting from `main`. Work on them with `git tree` (one worktree per branch) and cascade changes with `git tree propagate`.

## State after the H200 audit

- #4354 is correct and more accurate than the torch pooling it replaces. cuDNN finds each entry's document by scanning the row's document list, so it slows down on rows packed with many short documents; long-context rows hold few documents, so this does not matter for the workloads it targets. The position-bias gradient is accumulated with atomics and is not bitwise reproducible run to run.
- #4355 had a real bug: the halo path never all-gathered the pooled compressor entries, so every CP rank other than rank 0 attended to the wrong compressed entries while the loss stayed finite. It is fixed (`fix(deepseek-v4): all-gather the halo-pooled compressor entries`), with a unit test that fails without the fix. With the fix, halo matches the all-gather path bitwise in eager mode at cp 2, 4 and 8, and the loss of a small compiled model is bitwise equal on every step. Every halo performance number measured before the fix is invalid, because the buggy code skipped the entry all-gather.
- #4356 is correct: the q path is at the bf16 rounding floor, and the indexer path is bitwise equal to the unfused path. The shared `fp8_indexer` (also used by GLM models) is now model-agnostic again; the DSv4 RoPE + quantization lives in the DSv4 indexer.
- #4357 is correct. The pre-pass backward of mHC (layer pre-pass and hc_head) always runs TF32 with round-to-nearest, regardless of `matmul_precision`; everything else follows `matmul_precision`. An earlier version slowed down sharply under `matmul_precision = "highest"` because its fp32 kernel spilled registers; that kernel is gone. The env vars `PRIME_RL_DSV4_MHC_PAD_PROJECTION` and `PRIME_RL_DSV4_MHC_BWD_PRECISION` are gone.
- #4358 is correct after two fixes: module targets now work under the token-chunked checkpoint wrapper, and a target that never matches a traced operation raises after the first training forward instead of silently saving nothing.

## Do first

1. Revert the temporary SM90 commit `chore(deepseek-v4): admit SM90 in the cuDNN compressor kernels for H200 validation` (f2b06271e, not the branch tip) on feat/dsv4-cudnn-compressor with `git revert` (no history rewrite), then propagate. It lowers cuDNN's capability floor so the kernels run on H200, which cuDNN does not validate. On B300 it is unnecessary.
2. Sync and run the correctness checks below on B300 at the stack tip, after the revert. On H200 the cuDNN compressor test and the cuDNN arms of the halo test and parity matrix only exercised cuDNN because of that override, so the B300 run is the first one of the shipped code on hardware it admits. In particular, the cuDNN halo tiling has only been validated on H200.

## Correctness checks to rerun on B300

Run tests by name, not whole suites.

- Compressor: `pytest tests/unit/train/models/test_deepseek_v4_kernels.py -k cudnn_compressor_pool` and `pytest tests/unit/train/models/test_deepseek_v4.py -k "compressor_packed_matches_per_document or test_deepseek_v4_backward"`.
- CP halo unit test (needs two GPUs): `torchrun --nproc-per-node 2 -m pytest tests/unit/train/models/test_deepseek_v4.py -k test_halo_compress_returns_the_whole_rows_entries`, plus `-k test_halo_shards` without torchrun.
- CP halo parity matrix: `torchrun --nproc-per-node <cp> benchmarks/scripts/dsv4-stack-checks/cp_halo_parity.py` at cp 2, 4 and 8 (`--modes eager compile`, `--backends cudnn torch`). It compares a cp 1 reference, the all-gather path and the halo path on rows with documents crossing rank edges, for compressor entries, indexer and HCA picks, attention outputs and gradients. It exits non-zero on any failure.
- q norm + RoPE and indexer: `pytest tests/unit/train/models/test_deepseek_v4_kernels.py -k rope` and `pytest tests/unit/train/models/test_fp8_indexer.py tests/unit/train/models/test_fp8_indexer_cp_balance.py`.
- mHC: `pytest tests/unit/train/models/test_deepseek_v4_kernels.py -k "mhc or post_bda or sinkhorn"`. DeepGEMM's SM100 build of `tf32_hc_prenorm_gemm` is different code from the SM90 one that was tested.
- SAC: `pytest tests/unit/train/models/test_checkpointing.py` and `pytest tests/unit/test_configs.py -k "ac_targets or ac_none"`.
- Compile: for any change, check recompiles at cp 1 and cp 2 on real variable-length rows (`TORCH_LOGS=recompiles`), not only fixed-length synthetic rows.

## What to measure on B300

Only what is needed to decide each PR; B300 is the hardware these PRs target. Use paired runs against `main` on the same node.

- #4355: the halo speedup, from scratch. Expect the communication saving to hold (the fixed halo path still moves far less data than all-gathering the row), but the step-time effect was never measured on correct code.
- #4354: the compressor pooling gain on B300, now on validated hardware.
- #4356 and #4357: step-time effect at production shapes; per-op gains were measured on H200 only.
- #4358: whether a DSv4 config that uses module targets is worth its memory. The selected set adds a lot of activation memory at full depth. Also check what operator the MXFP8 GEMMs present to the AC policy: if it is not the FP8 GEMM op the docs example names, the example targets match nothing (the new check raises, so this fails loudly).

## Decisions still open

- `matmul_precision = "high"` for DSv4 RL. Production runs `"highest"` to match vLLM. The mHC pre-pass backward is already TF32 either way. Moving the rest to `"high"` needs a short RL run confirming mismatch KL stays at the production level.
- Whether to ship a DSv4 config that uses SAC module targets (above), or keep #4358 as unused flexibility. #4358 shares almost no code with #4354 to #4357, so it could also be based on `main` and land on its own.
- Whether halo becomes the only CP exchange for DSv4. `all_gather` stays the default until B300 shows halo is correct and no slower; if it is, drop `all_gather`, otherwise say in the `dsv4_cp_kv_exchange` docstring when each one wins.

## Known issues outside this stack (do not fix here)

- `DeepseekV4Compressor.compress` writes only the first batch row of RoPE for batch size > 1 (latent: the trainer always runs batch 1).
- The RL trainer raises an unclear `UnboundLocalError` in `rl/train.py` when a data-parallel rank receives zero micro batches.
- A bounded recompile on a stride guard of the mHC states' base tensor at steps 2 to 3 (present on `main`).
- With halo exchange on, rows shorter than the CP world size times the halo raise an assertion; loud, but undocumented.
- `SparseAttnInputs`' docstring says `build` checks the index range; it does not.

## Deferred cleanups (optional, after correctness and measurement)

- #4355: the cuDNN halo path does two ring shifts where one would do; `EntryShard.build` has two host syncs per compress rate; the HCA halo is larger than HCA alone needs.
- #4354: `_pool_cudnn` loops over the batch and stacks, for a batch size the trainer never uses.

## Before opening the PRs for review

Rewrite each PR description to match the code, describing only each PR's net change: several PRs carry commits that later commits in the same PR undo, kept so earlier measurements stay reachable, so reviewers should read the squashed diff. Specifically: drop all pre-fix halo performance numbers; the old mHC per-call speedup compared fp32 with TF32 and should not be quoted; describe the compressor as SM100-only, its speedup as holding on rows with few documents, and its position-bias gradient as nondeterministic; do not claim DSv4 saves a selected subset with SAC targets until a config does. Keep numbers approximate and say which hardware they come from.
