# Handoff: finish the DSv4 cudnn_flashmla backend on B300 (SM103)

Branch `feat/ds-v4-cudnn-flashmla`, parent `main`. The backend is done and measured on H200 (SM90). This handoff is
the Blackwell integration, to be done on a cluster with B300 access. B300 is compute capability 10.3 (SM103).

## What the branch has

- `refactor(deepseek-v4)`: the sparse attention ops (`prime_rl::dsv4_sparse_attn` and its backward) dispatch through
  backend tables (`FORWARD_BACKENDS`, `BACKWARD_BACKENDS`, `SHAPE_ERRORS`) on a `backend` op argument, threaded from
  `model.dsa_backend`.
- `feat(deepseek-v4)`: `cudnn_flashmla` = FlashMLA sparse prefill forward + cuDNN frontend SM90 DSA backward, in
  `src/prime_rl/trainer/models/kernels/deepseek_v4/dsv4_sparse_attn.py`. Slot widths round up to 128. The public LSE
  stays TileLang's (base 2, sink included). The SM90 backward gets the sink-inclusive natural-log LSE with no
  `attn_sink`, and dsink is formed in torch from the delta (TileLang `preprocess` kernel).
- `perf(deepseek-v4)`: `src/prime_rl/trainer/models/kernels/cudnn_dsa_compile.py` makes the SM90 backward compile
  its main kernel once for every slot width. It builds the kernel with `max_topk` at a power-of-two clamp of
  `max_sparse_attn_slots(config)` (16384 for DeepSeek-V4-Flash, from `deepseek_v4/attention.py`) and stores
  executors under the clamped key, in memory. Every compile logs one INFO line (`Compiled the cuDNN DSA backward
  <kernel> kernel ... in X.XXs`). A disk cache was rejected on purpose, and so is any startup warmup of shapes.
- `fix(deepseek-v4)`: the public backward op requires `max_slots` for `cudnn_flashmla`.
- This directory: the A/B harness and the H200 results (`results/README.md`), to be deleted in a final commit once
  the work is done (the user's practice: the SHA stays reachable, history is never rewritten).

`_cudnn_flashmla_shape_error` refuses anything but compute capability major 9, so today the backend raises at
model construction on Blackwell.

## H200 results (for comparison on B300)

DeepSeek-V4-Flash first 6 layers, real weights, cp 8, ep 8, one 8-GPU node:
- 8k tokens per GPU: steady step 2.80 s vs 2.86 s TileLang (about 2%), 16k tokens per GPU: 4.36 s vs 4.59 s
  (4-6%). Sparse attention GPU time 0.56x TileLang's.
- Compiles per rank: 3 cuDNN (preprocess, main, postprocess) and 1 TileLang, all in step 1, nothing later.
- Peak memory equal to or below TileLang's. Loss within the same-code noise floor.
- 128k tokens per GPU (1M sequence) runs out of memory in the shared CSA FP8 indexer (`fp8_indexer.py:203`
  allocates the full float32 score matrix), for both backends.

## What Blackwell needs (researched on H200 against the installed packages, not run)

Read the installed sources under `.venv/lib/python3.12/site-packages/` to re-verify each point. Frontend locked at
`nvidia-cudnn-frontend` 1.30.0.

1. **Shape gate.** Admit `(10, 0)` and `(10, 3)` in `_cudnn_flashmla_shape_error` (64 or 128 heads, head_dim 512,
   one KV head). `cudnn/.../_interface_sm100.py:23` has `_BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))`, and GLM's
   `sparse_mla_bwd.py` (~283) dispatches the same pair. Update the `dsa_backend` docstring in
   `packages/prime-rl-configs/src/prime_rl/configs/trainer.py`, which says DeepSeek V4 refuses anything but SM90.
2. **Forward.** FlashMLA's wheel carries SM100-family sparse prefill kernels for 64 and 128 heads at d_qk 512 (its
   error text: "only supported on SM90a and SM100f"). First check: does it run on SM103 at all. Then confirm
   `topk_length` and `attn_sink` work there, and that a width that is a multiple of 128 passes its
   "topk mod B_TOPK" check (the SM100 `B_TOPK` value was not readable from the binary).
3. **Backward adapter.** `flash_attn_bwd_sm100(q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale=None,
   topk_length=None, dq=None, dkv=None, deterministic=False, current_stream=None, workspace=None)` returns
   `(dq, dkv, d_sink)` (`_interface_sm100.py:201-216`). It differs from SM90:
   - `attn_sink` is required, float32 `(H,)`, and the LSE must be FlashMLA's natural-log LSE *excluding* the sink
     (:227). The kernel adds the sink itself and handles infinite LSE (`dsa_bwd_sm100.py` ~786-810). Two options:
     pass the sinkless LSE with the real sinks and take `d_sink` from the kernel, or keep the sink-inclusive LSE
     with `attn_sink = -inf` and keep dsink in torch. The public LSE is sink-inclusive base 2, so the first option
     needs the sinkless LSE either recomputed or saved from the forward. Measure both for accuracy against the
     reference before choosing.
   - `workspace` is mandatory: a uint8 CUDA buffer of `flash_attn_bwd_sm100_workspace_size(...)` bytes (:59).
   - Kernel choice: H=64, D=512 runs `generic_m64` (`dsa_bwd_sm100.py`). H=128 runs the tuned `h128_2cta_m64` only
     when `max_topk in (128, 512, 1024, 1152, 2048)` (:134-142), else `generic_m64`.
4. **Compile once on SM100.** The main kernel key is `(backend, deterministic, dtype, head_dim, head_dim_v,
   num_head, block_tile, max_topk, has_topk_length)` (:394), tail order reversed from SM90, with
   `max_topk = topk_idxs.shape[1]`. In `generic_m64`, `max_topk` is *not* only a clamp:
   - it bounds index loads (`if global_idx < self.max_topk`, `dsa_bwd_sm100.py:1429`), so the clamp must be at or
     above every real width or slots are silently dropped. Check what happens when the clamp exceeds the width of
     `topk_idxs` (out-of-bounds reads?) before relying on it.
   - `max_topk == 1024` and `== 2048` switch tiling (:69, :678-680), so keep the clamp off those values.
   - A clamp disables the H=128 2-CTA kernel. DeepSeek-V4-Flash has 64 heads, so it does not care.
   Extend `cudnn_dsa_compile.py` with an SM100 path (its own key normalization and kernel-class patch), keeping
   the SM90 path unchanged and GLM's calls without `topk_length` on stock behavior.
5. **Tests.** `tests/unit/train/models/test_deepseek_v4_kernels.py` gates `cudnn_flashmla` on major 9
   (`requires_cudnn_flashmla`) and the compile test reads the SM90 `compile_cache`. Widen both to Blackwell. All of
   `test_deepseek_v4_kernels.py` and `test_deepseek_v4.py` must pass on B300, including `torch.compile` tracing.

## Verification on B300

- Unit tests above.
- `compile_counts.py corpus | wide | two-clamps` from this directory: expect 3 cuDNN compiles per process, all on
  the first call, for every width up to 8320 (4 with two clamps).
- End-to-end A/B, TileLang vs cudnn_flashmla, two runs each, 8k and 16k tokens per GPU, reporting steady step
  time, tokens/s/GPU, per-rank peak memory, compile counts per rank and step, and loss parity (`compare.py`).
  `launch.sh` is puku-specific and needs adapting: `OUTPUT_ROOT`, `HF_HUB_CACHE`, the frozen worktree parent,
  the SLURM attach (`srun --jobid --overlap`), the W&B entity and project. Model and data must be reachable there
  (DeepSeek-V4-Flash-0731, `PrimeIntellect/INTELLECT-4-SFT` at revision `4c7951d04202eb4d428631d18697402c2b8b1871`).

## Open decisions for the user

- **Default backend.** The user wants `cudnn_flashmla` as the default if it is strictly better. `dsa_backend` is
  shared with GLM-5, and the backend needs the `flash-mla` extra (Linux x86_64 only) and a supported GPU. Proposed:
  a per-model default (None resolving to cudnn_flashmla for DeepSeek V4, tilelang for GLM), not yet decided.
- **Frontend upper bound.** `cudnn_dsa_compile` relies on private frontend internals and raises if they move.
  Whether to cap `nvidia-cudnn-frontend` in `pyproject.toml` is undecided.
- **Index prep in the op.** The flat-index and `topk_length` preparation runs eager inside the custom op (opaque to
  `torch.compile`), three times per layer per micro-batch. On H200 it is about 0.2% of a step. Moving it into
  compiled model code is possible but changes the op API.

## Rules

Ask the user before allocating nodes. Never push without explicit approval. No em-dashes in files. Draft PRs only.
