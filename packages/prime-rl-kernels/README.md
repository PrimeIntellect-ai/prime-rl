# prime-rl-kernels

Standalone GPU kernels used by `prime-rl`.

The NVFP4 implementation owns its grouped GEMM under
`prime_rl_kernels/nvfp4/grouped_gemm` and its quantizers under
`prime_rl_kernels/nvfp4/quantize`. The SM100 CUTLASS kernel is adapted from
MSLK. The quantizer is an owned CUDA implementation using Blackwell conversion
instructions adapted from Transformer Engine. Neither project is a runtime
dependency. The published NVIDIA CUTLASS headers are supplied by the
`nvidia-cutlass` Python package.

## Requirements

- NVIDIA GB200/B200 (SM100)
- CUDA 12.8 or newer
- PyTorch 2.11 or newer with `torch.float4_e2m1fn_x2`

## API

```python
from prime_rl_kernels.nvfp4.grouped_gemm import grouped_gemm
from prime_rl_kernels.nvfp4.quantize import quantize_activations, quantize_weights

# x:       [total_tokens, in_features], BF16
# weight:  [num_experts, in_features, out_features], BF16
# offsets: cumulative token counts, one INT32 value per expert
output = grouped_gemm(x, weight, offs=offsets)

# Standalone quantization returns packed E2M1 data, tcgen05-swizzled E4M3
# block scales, and FP32 decode scales.
x_nvfp4 = quantize_activations(x, offsets)
weight_nvfp4 = quantize_weights(weight)
```

Every grouped GEMM call derives the scales from the current BF16 tensors,
quantizes both operands, and runs the grouped GEMM.

## Quantization contract

- E2M1 values with one E4M3 block scale per 16 values.
- One FP32 decode scale per activation token:
  `token_amax / (6 * 448)`.
- One FP32 decode scale per expert weight tensor:
  `expert_amax / (6 * 448)`.
- Block scales remain true group-16 scales and are stored in the native
  128-by-4 tensor-core layout.
- Fused gate-up weights use one shared per-expert scale.
- Forward uses the owned SM100 grouped kernel. Backward either saves the exact
  packed forward operands and dequantizes them for BF16 dgrad and wgrad, or
  retains the original BF16 operands.

This is the same two-level per-token/per-expert NVFP4 recipe used by the online
vLLM path. It intentionally does not implement a 4-over-6 recipe.

## Compilation cache

The quantization and grouped-GEMM extensions are compiled lazily on first use.
Point `TORCH_EXTENSIONS_DIR` at persistent storage to reuse the resulting
shared objects across jobs:

```bash
export TORCH_EXTENSIONS_DIR=/persistent/cache/prime-rl-kernels/torch-extensions
```

PyTorch's extension lock allows distributed ranks sharing that directory to
wait on one build instead of compiling the same sources independently.

## prime-rl integration

Set the trainer model's quantization discriminator to `nvfp4`:

```toml
[trainer.model.quantization]
type = "nvfp4"
backward = "dequant_bf16"
```

The integration selects this path for `GroupedExperts` while leaving BF16
master parameters, checkpoints, optimizer state, dense linears, and weight
transfer unchanged. Weights are re-quantized from the current BF16 parameters
on every forward invocation, including activation-checkpoint recomputation.
Torch expert parallelism and the DeepEP local-expert path use the same adapter.
Set `backward = "bf16"` to retain the original BF16 operands for backward
instead of reconstructing them from the quantized forward operands.

## Current scope

- SM100 routed experts only.
- Runtime-fused gate-up projection with a shared per-expert scale.
- `torch.compile(fullgraph=False)` is supported through an intentional graph
  break around the custom operation. Full-graph capture is not yet supported.
- Only rows before the final logical offset have defined output. TorchTitan's
  physical padding rows are discarded by expert-parallel combine.

The repository is Apache-2.0 licensed; the grouped kernel retains its MSLK
license alongside the source.
