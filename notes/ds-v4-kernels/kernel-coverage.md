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

## Attention: can we borrow a library?

Row 8's `have` is doing a lot of work: could a library kernel replace the in-repo TileLang
`dsv4_sparse_attn` kernel outright? No, once per candidate, for reasons that don't overlap.

**Megatron-LM.** Two separate sparse-attention families live there and only one touches TileLang. The
DS-V4-hybrid CSA/HCA path (`csa.py`, `deepseek_v4_hybrid_attention.py`) rejects TileLang outright
(`transformer_config.py:1911-1915`: *"dsv4_hybrid does not support dsa_kernel_backend='tilelang'"*)
and dispatches to cuDNN-Frontend `develop` plus FlashMLA `nv_dev` instead, both git-branch-only and
absent from PyPI (`no_pypi_wheels` in Megatron's own `pyproject.toml`). The other family, DSA
(V3.2-style), does dispatch through TileLang (`ops/tilelang_sparse_mla_{fwd,bwd}.py`), but it's the
same upstream tile-ai/tilelang fork this repo already vendors, and this repo's version is arguably
ahead: dynamic shapes so one compiled kernel serves every packed length, wrapped as a proper
`torch.library.custom_op` with `register_fake`. The DS-V4-hybrid layer is itself `dev`-branch-only,
about four months old, `tensor_model_parallel_size=1`-only, and has no inference path of its own, so
it isn't a more mature target, just a differently-shaped one. Nothing here replaces the attention
core; what is portable is the index-construction algebra and the sink/RoPE/indexer-loss pieces, see
the portability shortlist in `megatron-survey.md`.

**FlashAttention (FA4).** Installed at 2.8.3, but its absorbed-MLA `head_dim_v==512` path is
compute-capability gated to sm100/sm110 in `_validate_head_dims`, capping at `head_dim<=256` on
Hopper (sm90, the deployed hardware). CSA runs at `head_dim=512`. That's a hardware wall baked into a
compile-time assert, not a shape mismatch to route around.

**cuDNN-Frontend / FlashMLA.** The exact two dependencies Megatron's own DS V4 path needs, and
they're unobtainable the same way for us: `nvidia-cudnn-frontend` pinned to an unreleased git commit,
`flash_mla` pinned to `deepseek-ai/FlashMLA@nv_dev`, neither on PyPI. Independent of the packaging
problem, cuDNN's `fused_compressor` shim is gated to `compute_capability == (10, 0)` by equality, so
it never fires on Hopper or on sm120 either.

**vLLM / flashinfer (inference-serving kernels).** vLLM ships an actual DS V4 sparse-MLA backend
(`vllm/models/deepseek_v4/sparse_mla.py`, `nvidia/flashmla.py`) with shapes matching production
exactly: `head_dim=512` (448 NoPE + 64 RoPE), `topk_tokens=index_topk`, compress ratios `{1,4,128}`
for SWA/CSA/HCA, and it runs on Hopper (`supported_compute_capability` includes major version 9).
flashinfer's parallel sparse-MLA kernel is Blackwell-consumer-only (sm120/121). Neither has any
transplant value for training: both are forward-only with no backward registered anywhere, built
around paged KV-cache blocks and CUDA-graph-captured decode/prefill scheduling with no analog in a
packed training forward pass, and the compute core is a prebuilt CUDA/cutlass extension
(`vllm._flashmla_C`, adapted from `deepseek-ai/FlashMLA`) rather than modifiable source. The reusable
idea, once more, is index construction, not the kernel.

**Triton.** No Triton kernel found anywhere in this survey implements the attention core itself
(`QK^T`, softmax, `...V`) at `head_dim=512`. Every Triton file that touched CSA-shaped attention
(Megatron's indexer-KL teacher `csa_teacher_lse.py`, its `fused_mla_yarn_rope_apply.py`, vLLM's
top-k metadata kernel, GLM DSA's `fp8_indexer.py`) does index, metadata, or scoring work around the
core, never the core. That lines up with the FlashAttention finding: nobody has a Triton, or stock
FlashAttention, implementation of sparse attention at this head_dim. Only TileLang and cutlass/CUDA
implementations exist.

**GLM DSA's `sparse_mla` (in-repo sibling).** The closest thing to a real drop-in: same repo, same
TileLang fork, already GPU-verified (`models/kernels/sparse_mla_{fwd,bwd}.py`, used by
`glm_moe_dsa/sparse_mla_attention.py`). Not a bare substitute, though. GLM DSA's MLA has a
decoupled-RoPE "tail" of score-only channels that DS V4 doesn't need (V4 has `K == V`, so all 512
channels feed both score and output), and no attention-sink term. `dsv4_sparse_attn_fwd.py`'s own
header documents this divergence. The DS V4 kernel already *is* this kernel, forked and adapted;
there's nothing left here to borrow.

Every library with a real fused kernel for DS-V4-shaped attention (`head_dim=512`, sparse top-k
gather, a learned sink) either isn't on PyPI, doesn't run on Hopper, or has no backward. The
tile-ai/tilelang fork is the one exception, which is why it's what's vendored.

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
