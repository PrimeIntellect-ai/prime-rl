"""MoE dispatch/combine + expert FFN, using `torch.distributed._symmetric_memory` and hand-written
CUDA kernels (`prime_kernels.comet_scatter`) instead of NCCL all-to-all, in the spirit of COMET
(https://arxiv.org/pdf/2502.19811) and Cursor's Mixture-of-Kittens megakernel.

Status: **wired into training** (`moe_runtime.configure_moe_runtime`, `model.moe.dispatch.type =
"comet"` -- see `CometMoEDispatchConfig`), with a real backward pass
(`autograd.CometMoELayerFunction`) verified against `TorchTokenDispatcher`'s existing,
fully-differentiable reference at both toy scale and a real training-step profile's shape
(Qwen3-30B-A3B: hidden=2048, num_experts=128, top_k=8, ep=8 --
`prime-rl-bench/benchmarks/profiling/SYNC_PROFILE_2026-08-17.md`), at 2/4/8-GPU scale.
Communication is intra-node only today: the symmetric-memory backend in use is `"CUDA"`
(CUDA-IPC/NVLink peer mapping), confirmed via `torch.distributed._symmetric_memory.get_backend()`
-- cross-node (InfiniBand) would need an NVSHMEM-backed backend plus real kernel changes (RDMA
put+signal instead of a plain pointer store), not just a config flag.

Dispatch and the expert FFN always run through a single CTA-specialized kernel
(`comet_scatter.fused_dispatch_ffn`, `autograd._run_fused_kernel_dispatch_and_ffn`) in the forward
pass, and its mirror-image backward kernel (`comet_scatter.fused_grad_combine_ffn`) does the same
for the gradient send: producer CTAs scatter, consumer CTAs compute, real intra-kernel
compute/communication overlap verified via per-CTA `clock64()` ranges (forward) and by construction
(backward reuses the identical CTA-specialization structure). This is the only architecture in this
package proven to achieve genuine overlap; there is no toggle or fallback to a sequential path.

The backward kernel computes the FFN's full gradient set directly -- input gradient
(`grad_dispatch_hidden`) via GEMMs recomputing `up`/`gate` from the saved `hidden_shadow`, and
weight gradients (`grad_up_proj`/`grad_down_proj`/`grad_gate_proj`) via a separate WMMA helper that
fp32-atomic-accumulates across tiles (one expert's tokens span multiple tiles). There is no
shadow-autograd graph anywhere in this path any more: forward runs the fused kernel once (not
twice), and backward's gradients come entirely from the fused kernel's own math, not from
`torch.autograd.grad` walking a separately-recomputed `torch._grouped_mm` graph. `differentiable_ffn.py`
survives only as the correctness *reference* the kernel is checked against, not as part of the
runtime path.

The forward kernel's three GEMMs (up/gate/down projection, `fused_dispatch_ffn_kernel`'s consumer
role) run on real Blackwell tensor cores via `tcgen05` MMA instructions with TMA-staged operands
(`bf16_gemm_bt_tile_tcgen05`, `pi::init_tmap_kmajor_3d` from `prime_kernels.flash_moe`'s vendored
`tcgen05_prelude.cuh`) -- measured ~5.5x faster than the plain-WMMA `bf16_gemm_bt_tile` it replaced,
for the up_proj GEMM shape in isolation at real bench scale (0.32ms vs 1.72ms, single CTA). The
backward kernel's GEMMs (`bf16_gemm_bt_tile` for the up/gate recompute, `bf16_gemm_nn_sum2_tile` for
the input gradient, `bf16_gemm_tn_atomic_tile` for the atomic-accumulated weight gradients) are
still plain WMMA -- porting them to tcgen05 is the next concrete lever, see Known gaps.

Correctness: verified at 1/2/4/8-GPU scale, toy and real bench shape (Qwen3-30B-A3B), for forward
output AND all four gradients (`grad_x`/`grad_top_scores`/`grad_up_proj`/`grad_down_proj`/
`grad_gate_proj`) against `TorchTokenDispatcher`. Performance: NOT a production speedup -- full
forward+backward measures ~4.8x *slower* than a plain `torch._grouped_mm`-based reference at real
bench scale (36.4ms vs 7.6ms/iter, 8 GPUs; improved from ~5.9x slower before the forward kernel's
tcgen05 conversion, still a net regression because backward's WMMA GEMMs -- now the larger share of
the kernel's total compute -- are unconverted). The overlap *mechanism* is real and verified; the
bottleneck is now squarely backward's compute, not the overlap.

Known gaps:
    - Backward's GEMMs (`bf16_gemm_bt_tile`, `bf16_gemm_nn_sum2_tile`, `bf16_gemm_tn_atomic_tile`)
      are the actual lever for a real end-to-end speedup now that forward's are on tcgen05. This is
      more work than forward was: three different GEMM shapes (A@B^T same as forward's, A@B direct
      for the input gradient, and A^T@B with cross-tile fp32 atomic accumulation for the weight
      gradients -- the atomic accumulation in particular has no obvious tcgen05-native equivalent
      and would likely still need a read-back-then-atomicAdd epilogue after a tcgen05 MMA, not a
      pure hardware accumulate).
    - Two tcgen05 usage bugs were found and fixed while building the forward conversion, both
      worth remembering for the backward port: (1) `tcgen05.alloc` internally relinquishes the
      CTA's tensor-memory allocation permit, so alloc+dealloc *per GEMM call* is illegal the second
      time it runs in the same CTA -- tensor memory must be allocated ONCE per CTA's whole lifetime
      (in `fused_dispatch_ffn_kernel`, once before the tile loop) and passed into the GEMM helper as
      a plain `taddr` parameter, not allocated inside it. (2) `tcgen05.dealloc` is `.sync.aligned`
      just like `.alloc` and needs the *whole warp* to execute it convergently, not one elected
      thread -- a single-thread dealloc doesn't reproduce in a short-lived isolated test but
      reliably hangs once the CTA has done enough real tcgen05 work beforehand. Both bugs manifest
      as a silent device-side hang (`cudaDeviceSynchronize` never returns), not a clean error, so
      isolate any new tcgen05 code in a standalone nvcc harness with device-side `printf`
      instrumentation before trusting it in the real kernel -- see git history for the harnesses
      used to find these two.
    - Earlier attempts at hand-rolling the tcgen05 *descriptor encoding* by hand (before switching
      to TMA-staged shared memory + `prime_kernels.flash_moe`'s vendored `tcgen05_prelude.cuh`)
      failed and were abandoned; that failure mode was fully explained once TMA entered the
      picture -- naively-staged (non-swizzled, hand-copied) shared memory doesn't match the
      canonical layout tcgen05's descriptors assume, and TMA is what performs that layout
      transformation during the copy, not something to hand-compute. Two independent descriptor
      *formulas* (a from-scratch derivation and `tcgen05_prelude.cuh`'s own `encode_smem_desc`)
      both work correctly once TMA does the staging, confirming the descriptor math was never the
      problem.
    - `score_before_experts` must be `False`; per-expert bias is not supported (both raise
      `NotImplementedError` rather than silently producing wrong results).
    - `model.moe.compute` is ignored for this dispatch path (DeepGEMM-FP8/MXFP8 are not wired up;
      both kernels are bf16-only).
    - Assumes every rank has the same local token count (implicit in giving every rank's
      symmetric-memory buffers the same capacity) and takes the caller's word for its buffer
      capacity being generous enough (silent truncation on overflow, not a raised error, in the
      hot path -- see `CometMoEDispatchConfig.capacity_multiplier`'s docstring).

Modules:
    buffers             -- symmetric-memory buffer allocation for the dispatch-receive and
                            combine-receive sides.
    metadata            -- routing/schedule computation: given each rank's local top-k routing
                            decision, compute alignment-padded per-(source, destination,
                            local-expert) segments so that every fixed-size GEMM tile is written
                            by exactly one producer (needed for race-free per-tile arrival
                            signaling), plus this rank's own dispatch work list. Also
                            `compute_backward_aux`, the backward-only derived quantities
                            autograd.py needs, and `padded_expert_offsets`, used only by the
                            `differentiable_ffn.py` correctness reference.
    kernels             -- the original, forward-only, CTA-specialized Triton kernels
                            (dispatch+grouped-GEMM, combine+weighted-reduce) -- still used by
                            `api.run_comet_moe_layer_with_buffers` (the naive/reference path), not
                            by the trainable path.
    flash_moe_compute   -- the forward-only `prime_kernels.flash_moe` tcgen05 kernel integration --
                            not used by the trainable path (no backward); kept for forward-only
                            comparisons.
    differentiable_ffn  -- `torch._grouped_mm`-based expert FFN, matching
                            `models.layers.moe.GroupedExperts`' own math and weight layout exactly.
                            Used only as the correctness reference the fused kernels' forward and
                            backward math are checked against (see `tools/comet_moe_*.py`); not
                            part of the runtime path.
    autograd            -- `CometMoELayerFunction`, the real forward+backward `torch.autograd.Function`
                            tying the fused dispatch+FFN kernel, its fused backward grad+FFN kernel,
                            and combine together.
    token_dispatcher    -- `CometMoETokenDispatcher`, implementing the same `TokenDispatcher`
                            protocol `token_dispatcher.TorchTokenDispatcher`/`deepep.DeepEPTokenDispatcher`
                            do, so `moe_runtime.py` can select it like any other dispatch backend.
    api                 -- high-level, forward-only entry points tying `kernels`/`flash_moe_compute`
                            together for benchmarking; not used by the trainable path.

See `tools/comet_moe_*.py` for runnable validation and benchmark scripts (metadata/dispatch
correctness, forward-only wallclock, forward+backward correctness and wallclock, and the
dispatcher-level integration check).
"""
