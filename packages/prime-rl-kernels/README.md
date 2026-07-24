# prime-rl-kernels

Standalone GPU kernels used by `prime-rl`.

The NVFP4 implementation owns its grouped GEMM under
`prime_rl_kernels/nvfp4/grouped_gemm`. The SM100 CUTLASS kernel is adapted from
MSLK, but it is compiled and loaded directly by `prime-rl-kernels`; Transformer
Engine and MSLK are not runtime dependencies. The published NVIDIA CUTLASS
headers are supplied by the `nvidia-cutlass` Python package.

## Requirements

- NVIDIA GB200/B200 (SM100)
- CUDA 12.8 or newer
- PyTorch 2.11 or newer with `torch.float4_e2m1fn_x2`

## API

```python
from prime_rl_kernels.nvfp4.grouped_gemm import grouped_gemm

# x:       [total_tokens, in_features], BF16
# weight:  [num_experts, in_features, out_features], BF16
# offsets: cumulative token counts, one INT32 value per expert
output = grouped_gemm(x, weight, offsets)
```

This is the only public kernel API. Every call derives the scales from the
current BF16 tensors, quantizes both operands, and runs the grouped GEMM.

## Quantization contract

- E2M1 values with one E4M3 block scale per 16 values.
- One FP32 decode scale per activation token:
  `token_amax / (6 * 448)`.
- One FP32 decode scale per expert weight tensor:
  `expert_amax / (6 * 448)`.
- Block scales remain true group-16 scales and are stored in the native
  128-by-4 tensor-core layout.
- Gate and up projections are currently quantized separately.
- Forward uses the owned SM100 grouped kernel. Backward is BF16 grouped dgrad
  and wgrad over the original BF16 tensors.

This is the same two-level per-token/per-expert NVFP4 recipe used by the online
vLLM path. It intentionally does not implement a 4-over-6 recipe.

## Compilation cache

The extension is compiled lazily the first time the grouped GEMM is called.
It contains only the dispatcher and two SM100 tile specializations. Point
`TORCH_EXTENSIONS_DIR` at persistent storage to reuse the resulting shared
object across jobs:

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
```

The integration selects this path for `GroupedExperts` while leaving BF16
master parameters, checkpoints, optimizer state, dense linears, and weight
transfer unchanged. Weights are re-quantized from the current BF16 parameters
on every forward invocation, including activation-checkpoint recomputation.
Torch expert parallelism and the DeepEP local-expert path use the same adapter.

## Benchmark

```bash
uv run python benchmarks/benchmark_grouped_nvfp4.py \
  --groups 32 --tokens-per-group 1024 --in-features 2048 --out-features 768
```

## Current scope

- SM100 routed experts only.
- Separate gate/up GEMMs; no fused gate/up weight quantization yet.
- `torch.compile(fullgraph=False)` is supported through an intentional graph
  break around the custom operation. Full-graph capture is not yet supported.
- Only rows before the final logical offset have defined output. TorchTitan's
  physical padding rows are discarded by expert-parallel combine.

See `nvfp4/grouped_gemm/NOTICE.md` for kernel provenance and licensing.
