# DeepSeek V4 mini-checkpoint EP + dequantization smoke test

Builds a cheap, quantized local checkpoint that exercises both expert parallelism (`ep>1`) and
the fp8/MXFP4 dequantization path (`../dequantize.py`) together, without needing the real
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint (hundreds of GB, quantized fp8/MXFP4 on disk).

## 1. Generate the plain mini checkpoint

```bash
uv run python scripts/mini_moe.py --arch deepseek_v4 --output-dir /tmp/deepseek-v4-mini
```

Uses the real checkpoint's architecture dimensions (`hidden_size`, `head_dim`, `compress_rates`,
etc.) with a shrunk layer count and expert count, so every kernel-relevant shape matches
production. Also runs an HF-vs-prime-rl roundtrip `verify()` — this must pass before continuing.

## 2. Quantize it

```bash
uv run python src/prime_rl/trainer/models/deepseek_v4/dev/quantize_mini_checkpoint.py \
    --input-dir /tmp/deepseek-v4-mini --output-dir /tmp/deepseek-v4-mini-quantized
```

Converts dense linear weights to blockwise fp8 (128x128 blocks) and routed-expert weights to
packed MXFP4 (1x32 blocks), matching the real checkpoint's on-disk format. `dequantize_state_dict_`
runs unconditionally on load, so this is a no-op for any key without a `.scale` sibling
(biases, norms, `tid2eid`) and only needs the `.scale` keys to be present to kick in — no
`quantization_config` flag required.

## 3. Run SFT against it

```bash
uv run sft @ examples/advanced/deepseek-v4-flash/sft-mini-ep-check.toml \
    --model.name /tmp/deepseek-v4-mini-quantized
```

On a single allocated 8xH200 node this runs as-is (`ep=8`, matching
`examples/advanced/deepseek-v4-flash/rl.toml`). On fewer/different GPUs, override both the GPU
count and `ep` together, e.g. on a 4-GPU box:

```bash
uv run sft @ examples/advanced/deepseek-v4-flash/sft-mini-ep-check.toml \
    --model.name /tmp/deepseek-v4-mini-quantized \
    --deployment.num-train-gpus 4 --deployment.num-infer-gpus 0 --model.ep 4
```

Attention auto-resolves per GPU architecture (`resolve_auto_attn` in
`src/prime_rl/trainer/model.py`): FA3 on Hopper (SM90), FA4 on datacenter Blackwell (SM100), FA2
everywhere else (e.g. Ada or workstation Blackwell like RTX PRO 6000) — no manual flag needed.
Dequantization and the trainer's MoE path (`moe_fused_kernel=False`, `use_grouped_mm=False` in
this config) are plain PyTorch, with no Hopper-specific kernel dependency.

**What "success" looks like:** finite loss and non-zero, varying grad norms across the 20 steps,
with `loss/nan_count` staying at 0. The config's `data.type = "fake"` data is synthetic, and
`optim.lr` is tiny, so a *decreasing* loss curve is not the signal to watch for here — this is a
correctness check (does the forward/backward pass run under `ep>1` with dequantized weights),
not a convergence check.
