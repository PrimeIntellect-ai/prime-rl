# Algorithm — Debug Configs

Minimal end-to-end configs for the algorithms against bundled verifiers envs, using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT` as the policy.

| Config | Algorithm | Frozen model | Notes |
|---|---|---|---|
| `grpo.toml` | `grpo` | none | |
| `max_rl.toml` | `max_rl` | none | GRPO with mean-normalized advantages (maximum-likelihood RL) |
| `opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft_distill.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | |
| `sft_distill_lora.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `self_distill.toml` | `opsd` | none (self-distills against the live policy) | SDFT; demo from reverse-text's `answer` field |
| `echo.toml` | `echo` | none | multi-turn `alphabet-sort`; CE on observation tokens |
| `mixed_grpo_opd.toml` | `grpo` + `opd` (per env) | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | two envs, one run; heterogeneous batches (with/without `ref_logprobs`) |

The policy inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with `gpu_memory_utilization=0.5`. The local frozen model (used by `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`) is **not** auto-launched — start it manually on GPU 1.

Frozen models are declared inline on the algorithm, named where the model is used — `[orchestrator.algo.teacher]` for `opd`, `[orchestrator.algo.sampling.source]` for `sft` — with `name` + `base_url`. `opsd` declares none (it self-distills against the live policy). prime-rl never hosts them; only the trainable policy's server is managed by the `rl` entrypoint.

## Start the local frozen model

Needed for `opd*.toml`, `sft_distill.toml` / `sft_distill_lora.toml`, and `mixed_grpo_opd.toml`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run inference \
  --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
  --server.port 8001 \
  --gpu-memory-utilization 0.5 \
  --model.enforce-eager
```

## Run the debug configs

Every config writes to the default `outputs/` directory, so running two back-to-back — or re-running one — fails with `FileExistsError`. Pass a distinct `--output-dir outputs/<algo>` per config (recommended for a sweep) or `--clean-output-dir` to wipe and restart.

```bash
# GRPO (no frozen model)
uv run rl @ configs/debug/algo/grpo.toml --output-dir outputs/grpo

# MaxRL (no frozen model)
uv run rl @ configs/debug/algo/max_rl.toml --output-dir outputs/max_rl

# OPD (needs the frozen model on port 8001)
uv run rl @ configs/debug/algo/opd.toml --output-dir outputs/opd
uv run rl @ configs/debug/algo/opd_lora.toml --output-dir outputs/opd_lora

# SFT distillation (needs the frozen model on port 8001)
uv run rl @ configs/debug/algo/sft_distill.toml --output-dir outputs/sft_distill
uv run rl @ configs/debug/algo/sft_distill_lora.toml --output-dir outputs/sft_distill_lora

# Self-distillation against the live policy (no frozen model)
uv run rl @ configs/debug/algo/self_distill.toml --output-dir outputs/self_distill

# ECHO (no frozen model; multi-turn env)
uv run rl @ configs/debug/algo/echo.toml --output-dir outputs/echo

# Mixed per-env algorithms: GRPO + OPD in one run (needs the frozen model on port 8001)
uv run rl @ configs/debug/algo/mixed_grpo_opd.toml --output-dir outputs/mixed_grpo_opd
```

See [docs/algorithms.md](../../../docs/algorithms.md) for what each algorithm does and how to author your own.
