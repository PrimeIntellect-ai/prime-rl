# Reverse Text — Debug Configs

Minimal end-to-end configs for the three training modes against the `reverse-text` env using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the student.

| Config | Mode | Teacher |
|---|---|---|
| `debug_rl.toml` | `rl` | none |
| `debug_opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) |
| `debug_sft.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) |

The student inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The teacher (used by `debug_opd.toml` and `debug_sft.toml`) is **not** auto-launched — start it manually on GPU 1.

## Start the teacher (only needed for opd/sft)

```bash
CUDA_VISIBLE_DEVICES=1 uv run vllm serve PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --port 8001 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager
```

## Run the debug configs

```bash
# RL (no teacher)
uv run rl @ configs/reverse_text/debug_rl.toml

# OPD (needs teacher on port 8001)
uv run rl @ configs/reverse_text/debug_opd.toml

# SFT hard distill (needs teacher on port 8001)
uv run rl @ configs/reverse_text/debug_sft.toml
```

See [docs/training_modes.md](../../docs/training_modes.md) for what each mode does.
