# NVFP4 on Blackwell: test kit

Everything here waits for an SM100/SM103 node (B200, GB200, B300). Run the steps in order: each one gates the next.
Install prime-kernels from `feat/nvfp4-moe-v2` built for the node:
`PRIME_KERNELS=nvfp4_moe,mxfp8_moe PRIME_KERNELS_REQUIRE=1`.

The recipe is one, after [Thinking Machines' end-to-end 4-bit training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/):
an NVFP4 forward on the first 5/6 of the layers (per-token scales on both GEMM inputs with 4/6 on activations,
per-expert weight scales, IEEE arithmetic), MXFP8 on the rest, and both gradient GEMMs in MXFP8 on the dequantized
forward operands with stochastic rounding. These steps check our own kernels and training first; `rl` sets vLLM's
engines as close to the recipe as vLLM can (see the gaps in step 3).

## 1. Kernel correctness and determinism (minutes)

```bash
cd prime-kernels
pytest tests/test_nvfp4_moe.py tests/test_nvfp4_dispatch.py tests/test_nvfp4_stochastic.py
```

- `test_nvfp4_moe.py`: the quantizers' bytes against the documented layout, the forward against the GEMM of the
  dequantized operands, the MXFP8 gradients against the BF16 gradients of the dequantized operands (within MXFP8
  error, also at RL-sized 1e-12 gradients), the same stochastic-rounding seed giving the same bits and another seed
  other ones, and whole-expert shards quantized apart matching the BF16 weight's GEMM bit for bit (the quantized
  all-gather's invariant).
- `test_nvfp4_dispatch.py`: quantizing before the permutation gives the same bytes and GEMM output as after.
- `test_nvfp4_stochastic.py` (also runs on CPU): stochastic rounding lands on the two E4M3 neighbours, is unbiased,
  and is reproducible from its seed.

Then the end-to-end determinism check: run two forward-backward passes of one MoE layer on the same inputs with the
same seed and compare the gradients byte for byte.

## 2. Kernel speed (minutes)

```bash
python benchmarks/nvfp4_moe.py --json nvfp4_moe.json
```

One expert layer forward and backward per shape and tokens per expert (T). Read:

- `nvfp4` against `bf16` and `mxfp8`: the recipe's speedup. Thinking Machines expect the NVFP4 forward about 3x the
  MXFP8 one in theory; their end-to-end gains over full MXFP8 were 8% in pretraining and 20% in RL.
- `+prequantized` against without: what receiving weights already quantized (the quantized all-gather) saves at
  each T.
- Components: `quantize_activations` against `nvfp4_gemm`; the fused stochastic-rounding quantizers
  (`grad_quant_stochastic`, `operand_requant_stochastic`) against round-to-nearest (`grad_quant_nearest`) and their
  PyTorch references (`*_torch`, expected ~10x slower).

## 3. RL numerics (hours per run)

Math RL on Qwen3-30B-A3B, one 8-GPU node, 200 steps per variant:

```bash
for variant in bf16 bf16-replay mxfp8 nvfp4 nvfp4-bf16-tail; do
  uv run rl @ benchmarks/nvfp4/base.toml @ benchmarks/nvfp4/$variant.toml --output-dir outputs/nvfp4-$variant
done
python benchmarks/nvfp4/compare.py outputs/nvfp4-mxfp8 outputs/nvfp4-*
```

| variant | question |
|---|---|
| `bf16`, `bf16-replay` | baselines; replay's own effect |
| `mxfp8` | the baseline Thinking Machines compare against |
| `nvfp4` | the recipe: does it track `mxfp8` in reward, loss and grad norm? |
| `nvfp4-bf16-tail` | the last sixth in BF16 instead of MXFP8 |

The bar: `nvfp4` reward within run-to-run noise of `mxfp8`, no grad-norm spikes, and mismatch KL of the same order as
`mxfp8`'s. The engines run vLLM's `nvfp4_per_token` on the NVFP4 layers and MXFP8 on the tail, with FlashInfer's IEEE
division and 4/6; what remains different (`rl` warns) is the down projection's static input scale on TRTLLM and the
unverified division of vLLM's load-time weight quantizer. A mismatch KL well above `mxfp8`'s points at those gaps.

## 4. RL speed

```bash
# EP 2 on the 4 trainer GPUs: the experts are FSDP-sharded over 2 ranks, so there is an all-gather to quantize
for variant in nvfp4 nvfp4-ag; do
  uv run rl @ benchmarks/nvfp4/base.toml @ benchmarks/nvfp4/$variant.toml --trainer.model.ep 2 \
    --output-dir outputs/nvfp4-$variant-ep2
done
python benchmarks/nvfp4/compare.py outputs/nvfp4-nvfp4-ep2 outputs/nvfp4-nvfp4-ag-ep2
```

`quantized_all_gather` must match `nvfp4` exactly (same loss step for step) and be faster. Each rank quantizes 1/shards
of the weights per all-gather instead of all of them for every GEMM, so the gain grows with the experts' FSDP group:
rerun the pair at the target scale before turning it on by default.

NVFP4 token dispatch against BF16, at EP > 1 so the tokens cross the expert all-to-all:

```bash
for variant in nvfp4 nvfp4-bf16-dispatch; do
  uv run rl @ benchmarks/nvfp4/base.toml @ benchmarks/nvfp4/$variant.toml --trainer.model.ep 4 \
    --output-dir outputs/nvfp4-$variant-ep4
done
python benchmarks/nvfp4/compare.py outputs/nvfp4-nvfp4-bf16-dispatch-ep4 outputs/nvfp4-nvfp4-ep4
```

The GEMMs see the same bytes either way, so loss and mismatch KL must match step for step (any difference is a
dispatch bug). `nvfp4` should be faster by what the all-to-all costs: ~3.5x fewer bytes each way it carries tokens to
the experts; the input gradient still returns in BF16. Repeat with DeepEP (`--trainer.model.moe.dispatch.type deepep`),
whose kernels move the packed rows as BF16 hidden states: a hidden size whose packed row (9/16 bytes per element plus
4) is not what DeepEP's buffers expect fails here first.

## What the results decide

- **Quantized all-gather.** Exact and faster: make it the default. It needs equal whole-expert FSDP shards, which the
  trainer config validates.
- **NVFP4 dispatch** (default): identical numerics and faster keeps it, and removes the temporary
  `moe.compute.dispatch` switch; slower would mean the packing costs more than the all-to-all bytes it saves at this EP.
- **Follow-ups:** Thinking Machines' 8-bit combine (a 32-wide random Hadamard transform and block-scaled 8-bit fixed
  point, GB300-specific); closing the engine gaps (a per-token down-projection input scale on the engines, e.g. the
  Mega MoE engine path).
