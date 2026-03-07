# FP8 Inference

This branch supports running vLLM with FP8 quantization and reloading PRIME-RL checkpoints into FP8 inference engines.

## Install

Some FP8 model stacks may require DeepGEMM (for example certain GLM-5-FP8 setups). If you hit a missing `deep_gemm` runtime error, install the FP8 inference dependency group:

```bash
uv sync --group fp8-inference
```

## Standalone Inference

```toml
# fp8_inference.toml
[model]
name = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
quantization = "fp8"
dtype = "bfloat16"
```

```bash
uv run inference @ fp8_inference.toml
```

## RL Quick Override

```bash
uv run rl @ configs/ci/integration/rl/start.toml --inference.model.quantization fp8
```

## Notes

- With FP8 quantization, `model.dtype` must be `auto`, `float16`, or `bfloat16`.
- When inference is running in FP8 mode, checkpoint reloads automatically convert BF16 weights to blockwise FP8 where needed.
