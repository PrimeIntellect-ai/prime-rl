# Testing MoE at Small Scale

When working on MoE architectures (GLM-4, Qwen3, MiniMax M2, etc.), you can't iterate on a 100B+ parameter model locally. This guide shows how to create a small (~0.5B) MoE model with the same architecture, run SFT to warm it up, and run RL on it — all on 1-2 GPUs.

The goal isn't performance. It's catching bugs in modeling code, state dict conversions, and training pipeline integration before running at scale.

## Available Test Models

| Model | Architecture | Experts | Active | Params | HF Link |
|-------|-------------|---------|--------|--------|---------|
| `PrimeIntellect/glm4-moe-tiny` | GLM-4 MoE | 8 | 4 | ~543M | [link](https://huggingface.co/PrimeIntellect/glm4-moe-tiny) |
| `PrimeIntellect/qwen3-moe-tiny` | Qwen3 MoE | 16 | 4 | ~500M | [link](https://huggingface.co/PrimeIntellect/qwen3-moe-tiny) |
| `PrimeIntellect/kimi-k25-tiny` | Kimi K2.5 (MLA+MoE) | 8 | 4 | ~500M | [link](https://huggingface.co/PrimeIntellect/kimi-k25-tiny) |
| `PrimeIntellect/minimax-m2-tiny` | MiniMax M2 | 8 | 4 | ~500M | [link](https://huggingface.co/PrimeIntellect/minimax-m2-tiny) |

All models have been fine-tuned on [PrimeIntellect/Reverse-Text-SFT](https://huggingface.co/datasets/PrimeIntellect/Reverse-Text-SFT) for a non-trivial distribution during RL.

## Overview

1. **Create + verify** a mini model with random weights and check HF <-> PrimeRL roundtrip
2. **SFT** to give it a non-trivial distribution
3. **RL** on reverse-text to validate the full pipeline

## Prerequisites

- At least 1 GPU for steps 1-2, 2 GPUs for step 3 (RL)
- Architecture presets are defined in `scripts/mini_moe.py`

## Step 1: Create and verify the mini model

```bash
# GLM-4 MoE
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./glm4-moe-tiny

# Qwen3 MoE
uv run python scripts/mini_moe.py --arch qwen3_moe --output-dir ./qwen3-moe-tiny

# MiniMax M2
uv run python scripts/mini_moe.py --arch minimax_m2 --output-dir ./minimax-m2-tiny
```

This creates a small MoE model with random weights, copies the tokenizer from the original model, then verifies that:
- Logits match between HF and PrimeRL implementations (`convert_to_prime`)
- The HF -> PrimeRL -> HF roundtrip is lossless (`convert_to_hf`)

To re-run verification only (e.g. after a modeling code change):

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./glm4-moe-tiny --verify-only
```

## Step 2: SFT warmup

Using the existing debug MoE SFT config with overrides for real data:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
    --model.name ./glm4-moe-tiny \
    --data.name PrimeIntellect/Reverse-Text-SFT \
    --data.type null \
    --max_steps 200 \
    --optim.lr 1e-4 \
    --ckpt.weights
```

Replace the model path with whichever architecture you're testing. Loss should drop from ~12 to ~2.5. The model won't be coherent, but it will have a non-trivial distribution so KL divergence is meaningful during RL.

The latest weight checkpoint is saved under `outputs/weights/step_<N>`. You can verify the roundtrip on it:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir outputs/weights/step_200 --verify-only
```

## Step 3: RL (reverse-text)

Requires 2 GPUs (one for inference, one for training). Each architecture has a dedicated CI config:

```bash
# GLM-4 MoE
uv run rl @ configs/ci/integration/rl_moe/glm4_moe.toml --trainer.model.impl custom

# Qwen3 MoE
uv run rl @ configs/ci/integration/rl_moe/qwen3_moe.toml --trainer.model.impl custom

# MiniMax M2
uv run rl @ configs/ci/integration/rl_moe/minimax_m2.toml --trainer.model.impl custom
```

You can also use `--trainer.model.impl hf` to test with the HuggingFace model implementation.

What to look for:
- **Training runs without crashing** — validates the full pipeline (inference server, orchestrator, trainer)
- **KL divergence is non-zero and finite** — confirms the reference model distribution is working
- **Loss is reasonable** — not NaN, not stuck at a constant value

Don't expect the reward to go up meaningfully in 20 steps on a random model.

## Step 4: RL with LoRA (reverse-text)

Same setup as Step 3, but with LoRA adapters enabled. Each architecture has a dedicated LoRA CI config:

```bash
# GLM-4 MoE
uv run rl @ configs/ci/integration/rl_moe_lora/glm4_moe.toml --trainer.model.impl custom

# Qwen3 MoE
uv run rl @ configs/ci/integration/rl_moe_lora/qwen3_moe.toml --trainer.model.impl custom

# MiniMax M2
uv run rl @ configs/ci/integration/rl_moe_lora/minimax_m2.toml --trainer.model.impl custom
```

These configs add LoRA rank-8 adapters on attention and expert layers, enable LoRA in the inference server, and save adapters separately. You can also use `--trainer.model.impl hf`.

## Adding a new architecture

To test a new MoE architecture:

1. Add modeling code under `src/prime_rl/trainer/models/<arch>/`
2. Add a preset to `scripts/mini_moe.py` with the config class, small dimensions, HF model class, PrimeRL model class, and tokenizer source
3. Run steps 1-3 above with `--arch <your_arch>`
4. Create a CI config in `configs/ci/integration/rl_moe/<arch>.toml` (copy an existing one and change the model name)
5. Create a LoRA CI config in `configs/ci/integration/rl_moe_lora/<arch>.toml` (copy an existing one and change the model name)
6. Add the new config to `MOE_CONFIGS` in `tests/integration/test_rl_moe.py` and `MOE_LORA_CONFIGS` in `tests/integration/test_rl_moe_lora.py`

The preset defines the small config:

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(
            hidden_size=1024,
            num_hidden_layers=24,
            n_routed_experts=8,
            # ...
        ),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    "qwen3_moe": { ... },
    "minimax_m2": { ... },
    # Add your new arch here
}
```
