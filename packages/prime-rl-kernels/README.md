# prime-rl-kernels

Standalone GPU kernels used by `prime-rl`.

The initial proof of concept provides grouped NVFP4 matrix multiplication for
Blackwell GPUs without a Transformer Engine dependency. It combines a Triton
NVFP4 packer with PyTorch's native grouped scaled-matmul backend and uses BF16
grouped matmuls for the straight-through backward pass.

## Requirements

- NVIDIA Blackwell (SM100 or newer)
- CUDA 12.8 or newer
- PyTorch 2.11 or newer built with NVFP4 grouped-matmul support

## API

```python
from prime_rl_kernels import grouped_nvfp4_mm

# x:       [total_tokens, in_features], BF16
# weight:  [num_experts, in_features, out_features], BF16
# offsets: cumulative token counts, one INT32 value per expert
output = grouped_nvfp4_mm(x, weight, offsets)
```

`quantize_nvfp4_activations`, `quantize_nvfp4_weights`, and
`grouped_nvfp4_mm_quantized` expose the lower-level path for callers that want
to manage packed-weight caching.

This PoC implements NVFP4 two-level scaling: E2M1 values, E4M3 scales over
16-value blocks, and one FP32 global scale per expert. This is deliberately not
yet the same recipe as vLLM's `nvfp4_per_token`, which uses one activation
global scale per token and jointly quantizes the gate/up projection. The block
scales are emitted directly in the native 128-by-4 swizzled layout. Ragged and
empty groups are supported, as is TorchTitan's physically padded token buffer.

## prime-rl integration

Set the trainer model's quantization discriminator to `nvfp4`:

```toml
[trainer.model.quantization]
type = "nvfp4"
```

The integration selects this path for `GroupedExperts` while leaving dense
linears, BF16 master parameters, checkpoints, optimizer state, and weight
transfer unchanged. Torch expert parallelism dispatches local packed tokens to
the same grouped kernel; the DeepEP local-expert path uses the same adapter.

## GB200 snapshot

The following medians were measured with 32 groups, `K=2048`, `N=768`, PyTorch
2.11, and warm caches. They measure the native grouped GEMM independently from
packing, then show both cached-weight and repack-both paths.

| Tokens per group | BF16 GEMM | NVFP4 prepacked | Activation pack | Cached-weight total | Repack-both total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.050 ms | 0.060 ms | 0.262 ms | 0.339 ms | 0.667 ms |
| 1,024 | 0.105 ms | 0.067 ms | 0.297 ms | 0.364 ms | 0.653 ms |
| 2,048 | 0.183 ms | 0.077 ms | 0.567 ms | 0.639 ms | 0.883 ms |

The prepacked kernel becomes faster than BF16 as groups get larger, but this
PoC is not yet an end-to-end training speedup: dynamic activation packing is
the bottleneck, and the prime-rl adapter currently repacks weights on every
call. Reproduce a shape with:

```bash
uv run python benchmarks/benchmark_grouped_nvfp4.py \
  --groups 32 --tokens-per-group 1024 --in-features 2048 --out-features 768
```

## Current scope

- Forward: NVFP4 grouped GEMM; backward: BF16 grouped dgrad and wgrad.
- Current per-expert amax scaling, not vLLM's per-token activation global
  scaling. A future cuDNN-backed row-scale recipe can fit behind the same API.
- No 4-over-6 recipe, dense-linear conversion, quantized checkpoint format, or
  LoRA-specific support.
- Packed-weight caching is exposed by the low-level API but is not yet wired to
  training-safe invalidation after optimizer steps.
- `torch.compile(fullgraph=False)` is supported through an intentional graph
  break around the custom operation. Full-graph capture is not yet supported.
- Only rows before the final logical offset have defined output. TorchTitan's
  physical padding rows are discarded by expert-parallel combine.
