# Development

This page covers workflows for developing on `prime-rl` itself — adding new model architectures, debugging modeling code, and the small-scale tooling we use to iterate on MoE families without booting up a 100B+ run.

## Table of Contents

- [Testing MoE at small scale](#testing-moe-at-small-scale)
  - [Step 1: build and verify a mini model](#step-1-build-and-verify-a-mini-model)
  - [Step 2: SFT warmup](#step-2-sft-warmup)
  - [Step 3: RL on reverse-text](#step-3-rl-on-reverse-text)
  - [Adding a new architecture](#adding-a-new-architecture)

## Testing MoE at small scale

When working on MoE architectures (GLM-4, Kimi, etc.), you can't iterate on a 100B+ model locally. The workflow below builds a ~0.5B model with the same architecture, warms it up with SFT, and runs RL — all on 1–2 GPUs. The goal is catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Step 1: build and verify a mini model

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

This creates a ~543M parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔PrimeRL roundtrip is lossless. To re-verify after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

### Step 2: SFT warmup

Use the shipped debug MoE SFT config with reverse-text data:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
  --model.name ./mini-glm-moe \
  --data.name PrimeIntellect/Reverse-Text-SFT \
  --data.type null \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt.weights
```

Loss drops from ~12 to ~2.5. The output won't be coherent, but the model now has a non-trivial distribution so KL divergence becomes meaningful in RL. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe).

### Step 3: RL on reverse-text

```bash
uv run rl @ configs/ci/integration/reverse_text_moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --trainer.model.impl custom \
  --inference.gpu-memory-utilization 0.7 \
  --inference.model.max-model-len 2048
```

What to look for:

- **No crashes.** Validates the full inference + orchestrator + trainer pipeline end-to-end.
- **Finite, non-zero KL.** Confirms the reference distribution is meaningful.
- **Loss reasonable.** Not NaN, not stuck.

Don't expect reward to climb meaningfully in 20 steps on a random model.

### Adding a new architecture

To add (e.g.) Kimi 2.5:

1. Add the modeling code under `src/prime_rl/trainer/models/<arch>/`.
2. Add a preset to `scripts/mini_moe.py` with the config class, small dimensions, HF + PrimeRL model classes, and tokenizer source:

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(hidden_size=1024, num_hidden_layers=24, n_routed_experts=8, ...),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    # add your arch here
}
```

3. Run the three steps above with `--arch <your_arch>`.
