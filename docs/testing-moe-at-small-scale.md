# Testing MoE at Small Scale

When working on MoE architectures (GLM-4, Kimi, etc.), you can't iterate on a 100B+ parameter model locally. This guide shows how to create a small (~0.5B) MoE model with the same architecture, run SFT to warm it up, and run RL on it — all on 1-2 GPUs.

The goal isn't performance. It's catching bugs in modeling code, state dict conversions, and training pipeline integration before running at scale.

## Overview

1. **Create + verify** a mini model with random weights and check HF <-> PrimeRL roundtrip
2. **SFT** to give it a non-trivial distribution
3. **RL** on reverse-text to validate the full pipeline

## Prerequisites

- At least 1 GPU for steps 1-2, 2 GPUs for step 3 (RL)
- Architecture presets are defined in `scripts/mini_moe.py`

## Step 1: Create and verify the mini model

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

This creates a ~543M parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, then verifies that:
- Logits match between HF and PrimeRL implementations (`convert_to_prime`)
- The HF -> PrimeRL -> HF roundtrip is lossless (`convert_to_hf`)

To re-run verification only (e.g. after a modeling code change):

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

## Step 2: SFT warmup

Using the existing debug MoE SFT config with overrides for real data:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
    --model.name ./mini-glm-moe \
    --data.name PrimeIntellect/Reverse-Text-SFT \
    --data.type null \
    --max_steps 200 \
    --optim.lr 1e-4 \
    --ckpt.weights
```

This fine-tunes on [PrimeIntellect/Reverse-Text-SFT](https://huggingface.co/datasets/PrimeIntellect/Reverse-Text-SFT) for 200 steps. Loss should drop from ~12 to ~2.5. The model won't be coherent, but it will have a non-trivial distribution so KL divergence is meaningful during RL.

The latest weight checkpoint is saved under `outputs/weights/step_<N>`. You can verify the roundtrip on it:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir outputs/weights/step_200 --verify-only
```

A pre-built SFT'd model is available at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe).

## Step 3: RL (reverse-text)

Requires 2 GPUs (one for inference, one for training).

```bash
uv run rl @ configs/ci/integration/rl/start.toml \
    --model.name samsja/mini-glm-moe \
    --trainer.model.impl custom \
    --inference.gpu-memory-utilization 0.7 \
    --inference.model.max-model-len 2048
```

Or to use the checkpoint from step 2:

```bash
uv run rl @ configs/ci/integration/rl/start.toml \
    --model.name outputs/weights/step_200 \
    --trainer.model.impl custom \
    --inference.gpu-memory-utilization 0.7 \
    --inference.model.max-model-len 2048
```

What to look for:
- **Training runs without crashing** — validates the full pipeline (inference server, orchestrator, trainer)
- **KL divergence is non-zero and finite** — confirms the reference model distribution is working
- **Loss is reasonable** — not NaN, not stuck at a constant value

Don't expect the reward to go up meaningfully in 20 steps on a random model.

## Kimi K2.5 (MLA + MoE)

The same workflow works for Kimi K2.5, which uses DeepseekV3's MLA (Multi-head Latent Attention) architecture:

### Step 1: Create and verify

```bash
uv run python scripts/mini_moe.py --arch kimi_k25 --output-dir ./mini-kimi-k25
```

### Step 2: SFT warmup

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
    --model.name ./mini-kimi-k25 \
    --data.name PrimeIntellect/Reverse-Text-SFT \
    --data.type null \
    --max_steps 200 \
    --optim.lr 1e-4 \
    --ckpt.weights
```

### Step 3: RL (requires 2 GPUs)

```bash
uv run rl @ configs/ci/integration/rl/start.toml \
    --model.name ./mini-kimi-k25 \
    --trainer.model.impl custom \
    --inference.gpu-memory-utilization 0.7 \
    --inference.model.max-model-len 2048
```

## Adding a new architecture

To test a new MoE architecture:

1. Add modeling code under `src/prime_rl/trainer/models/<arch>/`
2. Add a preset to `scripts/mini_moe.py` with the config class, small dimensions, HF model class, PrimeRL model class, and tokenizer source
3. Run steps 1-3 above with `--arch <your_arch>`

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
    # Add your new arch here
}
```
