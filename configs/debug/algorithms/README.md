# Algorithm Debug Configs

Minimal end-to-end algorithm configs against the `reverse-text` env, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Algorithm | Reference | Notes |
|---|---|---|---|
| `rl.toml` | `grpo` | none | default RL |
| `opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | uses `actor = "reference"` |
| `sft_lora.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local reference (used by everything except `rl.toml`) is **not** auto-launched — start it manually on GPU 1.

## Start the local reference

Needed for `opd*.toml` and `sft.toml` / `sft_lora.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

```bash
# RL (no reference)
uv run rl @ configs/debug/algorithms/rl.toml

# OPD (needs reference on port 8001)
uv run rl @ configs/debug/algorithms/opd.toml
uv run rl @ configs/debug/algorithms/opd_lora.toml

# SFT hard distill (needs reference on port 8001)
uv run rl @ configs/debug/algorithms/sft.toml
uv run rl @ configs/debug/algorithms/sft_lora.toml
```

See [docs/training.md](../../docs/training.md#algorithms-and-actor-selection) for how algorithms and actor selection interact.
