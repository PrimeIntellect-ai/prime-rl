# Kernel coverage for DeepSeek V4

Which ops have a fused kernel, from where, and whether we can use it. Surveyed 2026-09-03 against
the installed `.venv`, `deps/prime-kernels`, and Megatron-LM `dev` at `24fc94d27`, then verified
claim by claim against source. No GPU was available, so nothing was executed; numerical agreement
is untested throughout.

Forward-only is a cost, not a blocker: take the forward, write the backward. Free only for the
indexer, which needs no gradient.

`t` is the packed token count, and byte figures are per batch row at V4-Flash shapes.

**have** = usable now. **vendor** = copy source, no new dependency. **build** = write it.
**skip** = not worth it.

| # | Operation | Today | Elsewhere | fwd/bwd | Status |
|---|---|---|---|---|---|
| 1 | Sliding mask build | eager dense, 12 `t^2` B | Megatron index helper (torch) | n/a | build |
| 2 | Partial interleaved RoPE | eager, 212 calls/fwd | Megatron (Triton) | **fwd+bwd** | **vendor** |
| 3 | RMSNorm | quack, weighted only | quack (CuTe DSL) | fwd+bwd | partial |
| 4 | Compressor pooling | eager gather + softmax | Megatron (cuDNN shim) | fwd+bwd | build |
| 5 | Indexer scorer | eager fp32, 1x 64 `t^2` B | **in-repo** `fp8_indexer` (Triton) | fwd, no grad | **have** |
| 6 | Indexer top-k | `torch.topk` (ATen) | none usable | n/a | **have** |
| 7 | Sliding attention | eager dense scores | none | n/a | build, reuse row 8 |
| 8 | CSA attention | in-repo (TileLang) | Megatron (FlashMLA + cuDNN) | fwd+bwd | **have** |
| 9 | HCA attention | eager dense scores | none | n/a | build, reuse row 8 |
| 10 | mHC norm + project | eager fp32 linear | deep_gemm (CUDA), vLLM (TileLang) | **fwd only** | build bwd |
| 11 | mHC Sinkhorn | eager, 19-trip loop | Megatron (Triton) | **fwd+bwd** | **vendor** |
| 12 | mHC stream collapse | eager broadcast + sum | Megatron (Triton fwd) | fwd+bwd (bwd unfused) | skip |
| 13 | mHC scatter + mix | eager `bmm` | Megatron (Triton) | **fwd+bwd** | **vendor** |
| 14 | MoE router top-k | `torch.topk` (ATen) | none usable | n/a | **have** |
| 15 | MoE grouped GEMM | `torch._grouped_mm` | deep_gemm, torchao (CUDA) | fwd+bwd | **have** |
| 16 | MoE dispatch / combine | torch a2a, DeepEP (CUDA) | same | fwd, caller wires bwd | **have** |
| 17 | LM head + chunked CE | eager, chunked | quack (CuTe DSL) | fwd+bwd | **have** |

## Notes

**Row 1.** No fused kernel builds the band, but Megatron has the index construction:
`csa.py:59-70` `_get_window_topk_idxs_cached` and `csa.py:225-259` `get_window_topk_idxs_thd`, free
functions with no config dependency, returning int32 window indices with `-1` for invalid. Reusable
for rows 7 and 9. Megatron's own attention delegates the band to the backend; its one dense builder
is a bool mask in the eager softmax fallback. We cannot delegate, since no flash kernel takes
`head_dim 512` on sm90.

**Row 2.** `megatron-survey.md` calls this an "interleaving mismatch". Wrong, and never true at any
commit. Megatron's `fused_mla_yarn_rope_apply` rotates trailing channels by default (`rope_first=False`) and
does adjacent-pair with `remove_interleaving=True`, our convention exactly. It wants full-width
duplicated cos/sin against our half-width: four load statements to change, or zero if you `cat` at
the call site. About 520 self-contained lines. quack's `apply_rotary_emb` is already installed and
rotates leading channels, but its only hard constraint is `stride(-1) == 1` and the alignment rules
work out for our shapes, with a runtime check that fails loudly rather than silently. Two catches:
its out-of-place path returns a compact tensor, so a trailing slice gets no write-back, and its
in-place path never calls `mark_dirty`, so mutating an autograd-tracked view through it is unsound.

**Row 3.** `DeepseekV4UnweightedRMSNorm` (`hyperconnections.py:8-16`) is a separate eager fp32 norm
with no quack path, used for `q_b_norm` on every attention layer and for both mHC input norms. quack
itself is not sm90-gated for RMSNorm; that floor is ours.

**Rows 6, 14.** `quack.topk` exists with a backward but cannot serve either row: it asserts `k` and
`N` are powers of two, `k <= 128` and `N <= 4096`. Row 6 needs `k = index_topk = 512`, row 14 needs
`k = num_experts_per_tok = 6`. `torch.topk` stays.

**Row 5.** The eager scorer now runs in place under `no_grad`, one slab rather than three (`before-after.md`). `fp8_indexer.py` has the same structure as `DeepseekV4IndexerScorer`, fuses the causal
predicate and skips dead key tiles, and is GLM DSA's only indexer path in production. It is not the
same value: it quantizes q and k to UE8M0 fp8 and drops both constant scales, ignoring
`weight_scale` deliberately and applying no `index_head_dim^-0.5`. Positive constants cannot change
an argmax, so the top-k ordering is unaffected. Numerically untested, so adoption is gated on
measuring top-k set agreement against the fp32 scorer.

**Rows 7, 9.** No flash kernel takes `head_dim 512` on sm90, so there is nothing to adopt. The
in-repo CSA kernel does not care how indices were chosen, so both are index
construction rather than kernel work. Two constraints, though: the slot count must be a multiple of
64 (the forward tiles at `block_I = 64`, the backward at 32), and it is a kernel-compilation key.
That matters for HCA, where the readable entry set grows as `S / compress_rate` instead of being
capped at `index_topk`, so the slot count varies with sequence length and recompiles.

**Rows 11 to 13.** The Megatron symbols are `fused_sinkhorn`, `fused_h_aggregate` (Triton forward,
torch backward) and `fused_h_post_bda`. Row 12 stays **skip** because `torch.compile` already fuses
a broadcast and a sum. `fused_h_aggregate` is a real `torch.autograd.Function`
(`fused_mhc_kernels.py:2960-2982`) with a working backward (`:2752-2777`, cuTile or plain torch);
only its forward has a Triton kernel. That is not the same situation as row 10, where no backward is
registered at all.

**Rows 10 to 13.** vLLM has the mHC cluster fused, including `mhc_fused_post_pre` spanning two
layers. flashinfer has no such op, only `mhc_post` and two `mhc_pre` variants, and neither library's
"pre" kernel is fully fused, since the projection GEMM stays external. All of it is forward-only:
vLLM registers through `direct_register_custom_op` with no autograd kernel, and flashinfer's
`register_custom_op` is a no-op decorator. `vllm/model_executor/kernels/mhc/torch.py` is a torch
reference equivalent to our `DeepseekV4HyperConnection.forward`, Sinkhorn off-by-one included, and
its `hc_post_mult_value` is set to 2.0 by vLLM's DS V4 model, matching our hard-coded `2 *`.
flashinfer defaults that multiplier to 1.0, which would silently halve `post`.

**Row 10.** deep_gemm's op is `tf32_hc_prenorm_gemm`. It ships sm90 and sm100 impls, and does not normalize. It computes the projection and the
per-token sum of squares in one pass and defers the scale to the caller, which is valid because an
unweighted RMSNorm is a per-token scalar and the projection is linear.

**Row 17.** quack's op is `chunked_linear_cross_entropy`. The `== (9, 0)` gate is quack's, not ours: `FusedOutputLinear` has no capability check.
Off Hopper the call falls back to another chunked fused kernel, not to eager.

## Licensing

Megatron is BSD-3-Clause, not Apache-2.0 like the TileLang we vendor. Its files carry only a
copyright line, so a copy needs the full notice: add `SPDX-License-Identifier: BSD-3-Clause`, a
provenance comment naming the upstream path and `24fc94d27`, and a `LICENSE.megatron`.

## Looks available, is not

- `prime_kernels.flash_moe`: **architecture.** `arch = ["10.0a"]`, matched on `(major, minor)` and
  ignoring the suffix, so sm90 and consumer Blackwell sm120 both fail. Zero references in `src/`.
- `prime_kernels.rmsnorm`: **build config.** CUDA sources are committed, but its `[rmsnorm]` entry
  in `prime_kernels/kernels.toml` is commented out, so it is never built on any architecture and
  `is_available("rmsnorm")` raises `KeyError` before any capability check.
- Megatron `fused_proj_rms_compute_h`: **toolchain**, checked before architecture. It probes for the
  `tileiras` cuTile compiler and fails closed when absent, so it falls back to `@torch.compile` on
  any install without that toolchain, Hopper and Blackwell alike. `import cuda.tile` succeeds here
  but `tileiras` is not on PATH, so it is already inactive.
- Megatron `fused_compressor`: **architecture and dependency.** Equality-gated to `(10, 0)`, so it
  never fires on Hopper or on sm120, and it is a shim over `cudnn.csa.compressor`, which the
  installed cudnn 1.25.0 does not provide.

## Numerics, rows 10 to 13

Not a residual-dtype question. Every fused mHC path loads the bf16 residual and computes in fp32,
and so does ours: `hyperconnections.py:51` upcasts a residual that is already bf16, so its fp32
carries no information the fused paths discard. The two real differences are that
`deep_gemm.tf32_hc_prenorm_gemm` uses a **TF32** MMA, truncating the fp32 `fn` operand to about 10
mantissa bits, where we use an fp32 `F.linear`; and that the fused paths project then scale where we
normalize then project, which changes summation magnitudes. vLLM's TileLang fallback GEMM is a plain
fp32 loop with no TF32 loss. Measure before adopting.

Megatron's `fused_sinkhorn` has a separate issue: it saves `M_init` in the input dtype and rebuilds
the iteration chain from it in the backward, so its gradients differ from ours. Promoting that one
tensor to fp32 is a one-word edit.
