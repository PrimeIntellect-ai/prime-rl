# Algorithm — Debug Configs

Minimal end-to-end configs for the algorithm presets against bundled verifiers envs, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Algorithm | Frozen model | Notes |
|---|---|---|---|
| `grpo.toml` | `grpo` | none | |
| `opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill.toml` | `sft_distill` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `sft_distill_lora.toml` | `sft_distill` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill_external.toml` | `sft_distill` | PI inference (`openai/gpt-5-mini`) | external OAI endpoint; no local server |
| `self_distill.toml` | `self_distill` | none (`model = "policy"`) | SDFT against the live policy; demo from reverse-text's `answer` field |
| `echo.toml` | `echo` | none | multi-turn `alphabet-sort`; CE on observation tokens |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local frozen model (used by `opd*.toml` and `sft_distill.toml` / `sft_distill_lora.toml`) is **not** auto-launched — start it manually on GPU 1.

Frozen models are plain `[orchestrator.models.<key>]` entries; the algorithm points at them by key (`algo = { name = "opd", model = "reverse-text-rl" }`). There is no dedicated teacher slot — the same entry can serve any number of envs' algorithms.

## Start the local frozen model

Needed for `opd*.toml` and `sft_distill.toml` / `sft_distill_lora.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

```bash
# GRPO (no frozen model)
uv run rl @ configs/debug/algorithms/grpo.toml

# OPD (needs the frozen model on port 8001)
uv run rl @ configs/debug/algorithms/opd.toml
uv run rl @ configs/debug/algorithms/opd_lora.toml

# SFT distillation (needs the frozen model on port 8001)
uv run rl @ configs/debug/algorithms/sft_distill.toml
uv run rl @ configs/debug/algorithms/sft_distill_lora.toml

# SFT distillation from openai/gpt-5-mini via PI inference
# (requires PRIME_API_KEY + PRIME_TEAM_ID in env; no local frozen model needed)
uv run rl @ configs/debug/algorithms/sft_distill_external.toml

# Self-distillation against the live policy (no frozen model)
uv run rl @ configs/debug/algorithms/self_distill.toml

# ECHO (no frozen model; multi-turn env)
uv run rl @ configs/debug/algorithms/echo.toml
```

See [docs/algorithms.md](../../../docs/algorithms.md) for what each algorithm does and how to compose custom ones.
