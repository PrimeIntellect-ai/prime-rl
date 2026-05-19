# Reverse Text — Debug Configs

Minimal end-to-end configs for the three training modes against the `reverse-text` env using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the student.

| Config | Mode | Teacher | Notes |
|---|---|---|---|
| `debug_rl.toml` | `rl` | none | |
| `debug_opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `debug_opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `debug_sft.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `debug_sft_lora.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `debug_sft_external.toml` | `sft` | PI inference (`openai/gpt-5-mini`) | external OAI endpoint; no local teacher |

The student inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local teacher (used by everything except `debug_rl.toml` and `debug_sft_external.toml`) is **not** auto-launched — start it manually on GPU 1.

## Start the local teacher

Needed for `debug_opd*.toml` and `debug_sft.toml` / `debug_sft_lora.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

```bash
# RL (no teacher)
uv run rl @ configs/reverse_text/debug_rl.toml

# OPD (needs teacher on port 8001)
uv run rl @ configs/reverse_text/debug_opd.toml
uv run rl @ configs/reverse_text/debug_opd_lora.toml

# SFT hard distill (needs teacher on port 8001)
uv run rl @ configs/reverse_text/debug_sft.toml
uv run rl @ configs/reverse_text/debug_sft_lora.toml

# SFT hard distill from openai/gpt-5-mini via PI inference
# (requires PRIME_API_KEY + PRIME_TEAM_ID in env; no local teacher needed)
uv run rl @ configs/reverse_text/debug_sft_external.toml
```

See [docs/training_modes.md](../../docs/training_modes.md) for what each mode does.
