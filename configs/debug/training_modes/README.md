# Advantage/Actor Debug Configs

Minimal end-to-end configs for the built-in advantage patterns against the `reverse-text` env, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Advantage | Actor | Extra model | Notes |
|---|---|---|---|---|
| `rl.toml` | `grpo` | `policy` | none | |
| `opd.toml` | `opd` | `policy` | `teacher` local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | `policy` | `teacher` local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft.toml` | `sft` | `teacher` | `teacher` local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `sft_lora.toml` | `sft` | `teacher` | `teacher` local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local `teacher` model key used by OPD/SFT examples is **not** auto-launched — start it manually on GPU 1.

## Start the local teacher

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
# GRPO (policy actor only)
uv run rl @ configs/debug/training_modes/rl.toml

# OPD (needs teacher model key on port 8001)
uv run rl @ configs/debug/training_modes/opd.toml
uv run rl @ configs/debug/training_modes/opd_lora.toml

# SFT hard distill (teacher actor on port 8001)
uv run rl @ configs/debug/training_modes/sft.toml
uv run rl @ configs/debug/training_modes/sft_lora.toml
```

See [docs/training.md](../../docs/training.md#advantage-functions-and-actors) for how advantage functions and actors fit together.
