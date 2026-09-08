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
(`comet_scatter.fused_dispatch_ffn`, `autograd._run_fused_kernel_dispatch_and_ffn`) -- producer
CTAs scatter, consumer CTAs run the FFN, real intra-kernel compute/communication overlap verified
via per-CTA `clock64()` ranges. This is the only architecture in this package proven to achieve
genuine overlap; there is no toggle or fallback to a sequential path. Correctness: verified at real
8-GPU scale (forward output and all gradients match `TorchTokenDispatcher`). Performance: not yet
a production speedup -- the kernel's plain WMMA tensor-core GEMM isn't cutlass-competitive
(~11x slower than `torch._grouped_mm` in isolation, forward-only), and because the kernel has no
backward of its own, training uses a "fast value, differentiable shadow" pattern (the fused kernel
computes the real numbers under `no_grad`; `differentiable_ffn.py`'s `torch._grouped_mm` path
separately recomputes the same FFN, autograd-connected, purely to give `backward` a graph to walk)
that pays a full extra FFN pass every forward call -- together landing full forward+backward around
~2.3x slower end-to-end than a plain `torch._grouped_mm`-based reference at real bench scale. See
`autograd._run_fused_kernel_dispatch_and_ffn`'s docstring for the mechanics.

Known gaps:
    - The fused kernel's GEMM is the actual lever for real end-to-end speedup: it needs to be
      genuinely fast (real tcgen05 tensor cores, not the current WMMA) before this pays off in
      training. An attempt to hand-write a tcgen05 GEMM for the consumer-CTA role (mirroring
      `prime_kernels.flash_moe`'s own kernel source, since `flash_moe` itself can't be embedded in
      this kernel -- it's separately-compiled) did not converge: the unswizzled smem descriptor
      layout was tested against two independent hypotheses (both ruled out via a real reference and
      impulse-response probes, with a no-op control ruling out "reading garbage tensor memory"),
      and neither matched -- no working unswizzled B-operand example exists anywhere in this
      codebase to validate against, and the swizzled convention every real weight-operand usage
      actually relies on needs either NVIDIA's tcgen05 descriptor ISA documentation or substantially
      more search than was pursued.
    - A stream-based two-kernel alternative (chunked `wait_tiles` on a comm stream,
      `torch._grouped_mm` or `flash_moe` on the compute stream) was tried and abandoned: with
      `torch._grouped_mm` it measured *zero* actual cross-stream overlap (profiled directly: the
      comm stream's waits resolved before the compute stream's first GEMM even started) and was a
      net wallclock regression; with `flash_moe` as compute it measured a real ~1.3x average
      forward-only speedup (confirming stream-based overlap *can* work once compute is fast
      enough), but flash_moe has no backward, and wiring it into training via the same shadow
      pattern paid a fixed shadow-FFN tax that erased the forward-side win end-to-end (same ~0.40x
      regime as the WMMA path). Real training speedup needs *this* kernel's GEMM to be fast, not a
      different kernel swapped in via streams -- see git history for the concrete numbers if
      picking this back up.
    - `score_before_experts` must be `False`; per-expert bias is not supported (both raise
      `NotImplementedError` rather than silently producing wrong results).
    - The expert FFN's backward always runs via `torch._grouped_mm` bf16 (`differentiable_ffn.py`),
      ignoring `model.moe.compute` (DeepGEMM-FP8/MXFP8 are not wired up for this dispatch path).
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
                            signaling), plus this rank's own dispatch work list. Also the
                            backward-only derived quantities (`compute_backward_aux`,
                            `padded_expert_offsets`) autograd.py needs.
    kernels             -- the original, forward-only, CTA-specialized Triton kernels
                            (dispatch+grouped-GEMM, combine+weighted-reduce) -- still used by
                            `api.run_comet_moe_layer_with_buffers` (the naive/reference path), not
                            by the trainable path.
    flash_moe_compute   -- the forward-only `prime_kernels.flash_moe` tcgen05 kernel integration --
                            not used by the trainable path (no backward); kept for forward-only
                            comparisons.
    differentiable_ffn  -- `torch._grouped_mm`-based expert FFN, matching
                            `models.layers.moe.GroupedExperts`' own math and weight layout exactly.
                            Used as the backward-graph "shadow" for the fused kernel's forward (see
                            `autograd._run_fused_kernel_dispatch_and_ffn`).
    autograd            -- `CometMoELayerFunction`, the real forward+backward `torch.autograd.Function`
                            tying the fused dispatch+FFN kernel, its differentiable shadow, and
                            combine together.
    token_dispatcher    -- `CometMoETokenDispatcher`, implementing the same `TokenDispatcher`
                            protocol `token_dispatcher.TorchTokenDispatcher`/`deepep.DeepEPTokenDispatcher`
                            do, so `moe_runtime.py` can select it like any other dispatch backend.
    api                 -- high-level, forward-only entry points tying `kernels`/`flash_moe_compute`
                            together for benchmarking; not used by the trainable path.

See `tools/comet_moe_*.py` for runnable validation and benchmark scripts (metadata/dispatch
correctness, forward-only wallclock, forward+backward correctness and wallclock, and the
dispatcher-level integration check).
"""
