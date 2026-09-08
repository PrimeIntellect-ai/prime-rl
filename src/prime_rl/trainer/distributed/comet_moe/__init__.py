"""MoE dispatch/combine + expert FFN, using `torch.distributed._symmetric_memory` and hand-written
CUDA kernels (`prime_kernels.comet_scatter`) instead of NCCL all-to-all, in the spirit of COMET
(https://arxiv.org/pdf/2502.19811) and Cursor's Mixture-of-Kittens megakernel.

Status: **wired into training** (`moe_runtime.configure_moe_runtime`, `model.moe.dispatch.type =
"comet"` -- see `CometMoEDispatchConfig`), with a real backward pass
(`autograd.CometMoELayerFunction`) verified against `TorchTokenDispatcher`'s existing,
fully-differentiable reference at both toy scale and a real training-step profile's shape
(Qwen3-30B-A3B: hidden=2048, num_experts=128, top_k=8, ep=8 --
`prime-rl-bench/benchmarks/profiling/SYNC_PROFILE_2026-08-17.md`), at 2/4/8-GPU scale. A full
forward+backward step measures ~1.13-1.29x faster than that reference at that shape, on an
otherwise-idle box. Communication is intra-node only today: the symmetric-memory backend in use is
`"CUDA"` (CUDA-IPC/NVLink peer mapping), confirmed via `torch.distributed._symmetric_memory.get_backend()`
-- cross-node (InfiniBand) would need an NVSHMEM-backed backend plus real kernel changes (RDMA
put+signal instead of a plain pointer store), not just a config flag.

Known gaps:
    - Chunked compute/communication overlap exists (`autograd._run_chunked_dispatch_wait_and_ffn`,
      `CometMoEDispatchConfig.n_chunks`) but is **off by default** (`n_chunks=1`): splitting the
      dispatch-receive buffer into expert-aligned chunks and pipelining each chunk's `wait_tiles`
      (on a dedicated CUDA stream) against the *previous* chunk's expert FFN (on the main stream)
      is implemented and correct, but measured to be a net *regression* at real scale on an idle
      8-GPU box -- every `n_chunks > 1` tested (2, 4, 8) was slower than the plain-NCCL reference
      (0.54-0.84x) and got monotonically worse with more chunks, because splitting one
      `torch._grouped_mm` call per FFN stage into N separate calls costs more in fixed per-launch
      overhead than the overlap saves, at this problem's single-digit-millisecond scale. See
      `CometMoEDispatchConfig.n_chunks`'s docstring for the numbers. Left in the code (and
      exercised for correctness by `tools/comet_moe_backward_check.py`, which deliberately runs
      with `n_chunks=4`) in case a larger problem shape amortizes launch overhead well enough to
      make it worth raising -- re-measure with `tools/comet_moe_backward_wallclock.py` before
      changing the default.
    - A second, architecturally different overlap attempt is also wired in and also off by default
      (`CometMoEDispatchConfig.use_fused_kernel`): `comet_scatter.fused_dispatch_ffn` is a real
      CTA-specialized single-kernel-launch fusion of dispatch and the expert FFN (producer CTAs
      scatter, consumer CTAs run a WMMA tensor-core GEMM) -- genuine intra-kernel overlap, verified
      via per-CTA `clock64()` ranges to actually happen (unlike `n_chunks`'s stream-based attempt,
      measured to have zero real overlap). Correct at real 8-GPU scale (forward output and all
      gradients match `TorchTokenDispatcher`), but the kernel has no backward of its own -- forward
      uses a "fast value, differentiable shadow" pattern (`autograd._run_fused_kernel_dispatch_and_ffn`):
      the fused kernel computes the real (fast) numbers, and `differentiable_ffn.py`'s already-
      verified `torch._grouped_mm` path separately recomputes the same FFN on a clone of the same
      input purely to build an autograd graph, then that graph's tensor is overwritten in place
      with the fused kernel's values -- `backward()` needs no special-casing at all. Net result:
      correct, but ~2.3x slower end-to-end (16.69ms vs 7.21ms reference, full fwd+bwd, real scale)
      since forward now pays for the FFN twice (fast kernel once, differentiable recompute once)
      on top of the fused kernel itself not being cutlass-competitive yet (~11x slower than
      `torch._grouped_mm` in isolation, forward-only). Wired in for continued kernel development,
      not as a production speedup -- only SiLU (gated or ungated) is supported.
    - A stream-based variant using `prime_kernels.flash_moe` (real tcgen05, ~3.5x faster than
      `torch._grouped_mm`) as the per-chunk compute was tried and **reverted**: it measured a real
      forward-only win (~1.3x average, real 8-GPU bench scale, using the same two-stream chunked
      architecture as `n_chunks` above), confirming stream-based overlap *can* work once compute is
      fast enough -- but flash_moe can't be embedded in `use_fused_kernel`'s single persistent
      kernel (it's a separately-compiled kernel, not inlinable device code), and because it has no
      backward of its own, wiring it into training via the same "fast value, differentiable shadow"
      pattern paid a fixed shadow-FFN tax that erased the forward-side win end-to-end (~0.40x fwd+bwd,
      same regime as `use_fused_kernel`). The conclusion: real training speedup needs the single-
      kernel-fusion architecture's GEMM to be genuinely fast (real tcgen05, not WMMA) -- not a
      different kernel swapped in via streams. An attempt to hand-write a tcgen05 GEMM for
      `use_fused_kernel`'s consumer-CTA role (mirroring `flash_moe`'s own kernel source) did not
      converge: the unswizzled smem descriptor layout was tested against two independent hypotheses
      (both ruled out via a real reference and impulse-response probes, with a no-op control ruling
      out "reading garbage tensor memory") and neither matched: no working unswizzled B-operand
      example exists anywhere in this codebase to validate against, and the swizzled convention
      every real weight-operand usage actually relies on needs either NVIDIA's tcgen05 descriptor
      ISA documentation or substantially more search than was pursued.
    - `score_before_experts` must be `False`; per-expert bias is not supported (both raise
      `NotImplementedError` rather than silently producing wrong results).
    - The expert FFN always runs via `torch._grouped_mm` bf16 (`differentiable_ffn.py`), ignoring
      `model.moe.compute` (DeepGEMM-FP8/MXFP8 are not wired up for this dispatch path) -- chosen
      deliberately: `flash_moe_compute.py`'s `flash_moe` kernel is faster in isolation but
      forward-only, and the real bottleneck this package targets is communication, not FFN FLOPs.
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
                            also not used by the trainable path; kept for forward-only comparisons.
    differentiable_ffn  -- the trainable path's expert FFN: `torch._grouped_mm`-based, matching
                            `models.layers.moe.GroupedExperts`' own math and weight layout exactly.
    autograd            -- `CometMoELayerFunction`, the real forward+backward `torch.autograd.Function`
                            tying dispatch, the differentiable FFN, and combine together, plus the
                            two (default-off) overlap paths: `n_chunks` (stream-based chunking) and
                            `use_fused_kernel` (CTA-specialized single-kernel fusion).
    token_dispatcher    -- `CometMoETokenDispatcher`, implementing the same `TokenDispatcher`
                            protocol `token_dispatcher.TorchTokenDispatcher`/`deepep.DeepEPTokenDispatcher`
                            do, so `moe_runtime.py` can select it like any other dispatch backend.
    api                 -- high-level, forward-only entry points tying `kernels`/`flash_moe_compute`
                            together for benchmarking; not used by the trainable path.

See `tools/comet_moe_*.py` for runnable validation and benchmark scripts (metadata/dispatch
correctness, forward-only wallclock, forward+backward correctness and wallclock, and the
dispatcher-level integration check). `COMET_N_CHUNKS`/`COMET_USE_FUSED_KERNEL` env vars toggle the
two overlap paths in `tools/comet_moe_backward_check.py`/`tools/comet_moe_backward_wallclock.py`.
"""
