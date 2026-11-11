# NVFP4 on Blackwell: test kit

Everything here waits for an SM100/SM103 node (B200, GB200, B300). Run the steps in order: each one gates the next.
Install prime-kernels from `feat/nvfp4-moe-perf` (stacked on `feat/nvfp4-moe-v2`) built for the node:
`PRIME_KERNELS=nvfp4_moe,mxfp8_moe PRIME_KERNELS_REQUIRE=1`.

## 1. Kernel correctness (minutes)

```bash
cd prime-kernels
pytest tests/test_nvfp4_moe.py tests/test_nvfp4_dispatch.py tests/test_mxfp8_moe_grad_quant.py
```

- `test_nvfp4_moe.py`: quantizer bytes against the documented layout, the forward against a dequantized BF16
  reference, every backward, and the weight cache reproducing uncached gradients bit for bit.
- `test_nvfp4_dispatch.py`: quantizing before the permutation gives the same bytes and GEMM output as after.
- `test_mxfp8_moe_grad_quant.py`: the one-pass gradient quantizer is byte-identical to the two passes. If only the
  row-wise half differs (torchao's CUDA and Triton quantizers round alike but are separate code), keep the one-pass
  column-wise output and the Triton row-wise one.

Then the matching-numerics check that motivates the whole recipe: the trainer's quantizer must produce FlashInfer's
bytes (what vLLM's engines run). #12 checked it on its original stack; rerun it on this one, with and without 4/6.

## 2. Kernel speed (minutes)

```bash
python benchmarks/nvfp4_moe.py --json nvfp4_moe.json
```

One expert layer forward and backward per shape and tokens per expert (T). Read:

- `nvfp4/mxfp8` against `bf16` and `mxfp8`: the recipe's speedup. Cursor reports ~1.5x for MXFP8 MoE layers with
  hand-fused kernels; NVFP4's forward should beat that if the GEMM is not starved by the quantizers.
- `+cache` against without: what quantizing weights once per step buys at each T. Expect a lot at T ~ 1k and
  little at T ~ 16k (quantizing a weight costs about as much as its GEMMs at T ~ 800).
- Components: `quantize_activations` against `nvfp4_gemm` sizes the payoff of quantizing before dispatch (it
  removes `top_k - 1` of every `top_k` activation quantizations of the gate_up input); `grad_quant_one_pass`
  against `two_pass` the gradient fusion.

## 3. RL numerics (hours per run)

Math RL on Qwen3-30B-A3B, one 8-GPU node, 200 steps per variant:

```bash
for variant in bf16 bf16-replay nvfp4 nvfp4-no-replay nvfp4-no-4over6 nvfp4-dequant-bwd nvfp4-dense-mxfp8; do
  uv run rl @ benchmarks/nvfp4/base.toml @ benchmarks/nvfp4/$variant.toml --output-dir outputs/nvfp4-$variant
done
python benchmarks/nvfp4/compare.py outputs/nvfp4-bf16-replay outputs/nvfp4-*
```

| variant | question |
|---|---|
| `bf16`, `bf16-replay` | baselines; replay's own effect on mismatch |
| `nvfp4` | the recipe: does it track `bf16-replay` in reward, and how much mismatch KL does it add? |
| `nvfp4-no-replay` | is router replay needed? |
| `nvfp4-no-4over6` | does 4/6 scaling matter? |
| `nvfp4-dequant-bwd` | MXFP8 backward against BF16 gradients on the dequantized operands |
| `nvfp4-dense-mxfp8` | dense layers in MXFP8 too (shared experts, router, LM head stay BF16) |

The bar: `nvfp4` reward within run-to-run noise of `bf16-replay`, mismatch KL of the same order (a few times
the baseline, not orders of magnitude), no grad-norm spikes. If mismatch KL is far above the baseline:

1. Check that the engines run the trainer's quantization (`FLASHINFER_NVFP4_4OVER6` set on them, `nvfp4_per_token`
   in the vLLM logs).
2. Division precision: the trainer and FlashInfer both divide with `rcp.approx`; Composer 2 found IEEE division
   mattered. That needs the same change on both sides.

## 4. RL speed

```bash
uv run rl @ benchmarks/nvfp4/base.toml @ benchmarks/nvfp4/nvfp4-cache.toml --output-dir outputs/nvfp4-nvfp4-cache
python benchmarks/nvfp4/compare.py outputs/nvfp4-bf16-replay outputs/nvfp4-nvfp4 outputs/nvfp4-nvfp4-cache
```

`fwd_bwd s` and `speedup` give the trainer-side gain; the trainer logs `NVFP4 weight cache held X GiB` after the
first step, the memory the cache costs. Numerics must match `nvfp4` exactly.

## What the results decide

- **Weight cache.** Large speedup and the memory fits: turn `cache_weights` on by default for EP runs. Large
  speedup but it does not fit (big models, low EP): quantize each FSDP shard once per step and all-gather the
  quantized bytes (an FSDP2 all-gather extension, like torchao's float8 weights): the cache's speed at the shard's
  memory, with 2-3.5x less all-gather traffic. Small speedup: leave it off and skip that work.
- **Quantize before dispatch** (`prime_kernels.nvfp4_moe.dispatch`). Locally it only saves the gate_up input's
  repeated quantizations, since the MXFP8 backward still needs the BF16 permuted tokens; worth wiring into the token
  dispatcher if `quantize_activations` is a large share of the layer. The bigger win, NVFP4 bytes through the EP
  all-to-all, needs the weight gradient to run on dequantized tokens: run `nvfp4-dequant-bwd` first to see what
  that costs in numerics. Not valid with `score_before_experts` (tokens are scaled per route before the experts).
- **Further kernel work**, by what the profile shows: quantization fused into the GEMM prologue, CUTLASS tile and
  cluster tuning for B200/B300, and (largest) a fused MoE-layer kernel in the style of Cursor's MoK.
