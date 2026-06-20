# Advantage — Debug Configs

Minimal end-to-end configs for builtin advantage functions against bundled verifiers envs, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Advantages | Extra model | Notes |
|---|---|---|---|
| `grpo.toml` | `grpo` | none | |
| `max_rl.toml` | `max_rl` | none | GRPO with mean-normalized advantages (maximum-likelihood RL) |
| `opd.toml` | `opd` | local `reference` endpoint (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | local `reference` endpoint (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill.toml` | `sft` | local `reference` actor (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `sft_distill_lora.toml` | `sft` | local `reference` actor (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill_external.toml` | `sft` | PI inference `reference` actor (`openai/gpt-5-mini`) | external full-token endpoint; no local server |
| `self_distill.toml` | `opsd` | none | SDFT against the live policy; demo from reverse-text's `answer` field |
| `echo.toml` | `echo` | none | multi-turn `alphabet-sort`; CE on observation tokens |
| `mixed_grpo_opd.toml` | `grpo` + `opd` (per env) | local `reference` endpoint (`Qwen3-0.6B-Reverse-Text-RL`) | two envs, one run; heterogeneous loss channels |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local `reference` endpoint (used by `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`) is **not** auto-launched — start it manually on GPU 1.

Extra endpoints are declared under `[orchestrator.models.<key>]`. `actor = "reference"` selects an endpoint for rollouts; OPD keeps `actor = "policy"` and uses `models["reference"]` inside the advantage function.

## Start the local reference endpoint

Needed for `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

```bash
# GRPO (no extra model)
uv run rl @ configs/debug/algorithms/grpo.toml

# MaxRL (no extra model)
uv run rl @ configs/debug/algorithms/max_rl.toml

# OPD (needs the reference endpoint on port 8001)
uv run rl @ configs/debug/algorithms/opd.toml
uv run rl @ configs/debug/algorithms/opd_lora.toml

# SFT distillation (needs the reference actor on port 8001)
uv run rl @ configs/debug/algorithms/sft_distill.toml
uv run rl @ configs/debug/algorithms/sft_distill_lora.toml

# SFT distillation from openai/gpt-5-mini via PI inference
# (requires PRIME_API_KEY + PRIME_TEAM_ID in env; no local reference endpoint needed)
uv run rl @ configs/debug/algorithms/sft_distill_external.toml

# Self-distillation against the live policy (no extra model)
uv run rl @ configs/debug/algorithms/self_distill.toml

# ECHO (no extra model; multi-turn env)
uv run rl @ configs/debug/algorithms/echo.toml

# Mixed per-env advantages: GRPO + OPD in one run (needs the reference endpoint on port 8001)
uv run rl @ configs/debug/algorithms/mixed_grpo_opd.toml
```

See [docs/algorithms.md](../../../docs/algorithms.md) for what each advantage does and how to compose custom ones.
