# Training Mode — v1 Debug Configs

Minimal end-to-end configs for the three training modes (`rl` / `opd` / `sft`) on the **v1
env server** (the `reverse-text-v1` taskset), using `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`
as the student. The v1 mirror of `configs/debug/training_modes/`.

Training is **renderer-only**: every rollout client is the renderer client, so rollouts always
carry exact token ids + logprobs. RL/OPD roll out the student; SFT rolls out the teacher
through the renderer client too, so its teacher must be a **self-hosted vLLM** (token-in/out) —
distilling from an external chat API isn't supported. The teacher must also **share the
student's tokenizer** (e.g. same model family): the student trains on exactly the token ids the
renderer feeds the teacher and the teacher samples, so the vocabularies must match. Every config
sets `[orchestrator.renderer]` accordingly.

| Config | Mode | Teacher | Notes |
|---|---|---|---|
| `rl.toml` | `rl` | none | |
| `opd.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | teacher computes logprobs |
| `opd_lora.toml` | `opd` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |
| `sft.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | teacher rolls out, tokens backfilled |
| `sft_lora.toml` | `sft` | local vLLM (`Qwen3-0.6B-Reverse-Text-RL`) | trains a LoRA adapter (rank 8) |

The student inference server is auto-launched on GPU 0 at `http://localhost:8000/v1` with
`gpu_memory_utilization=0.5`. The local teacher (everything except `rl.toml`) is **not**
auto-launched — start it manually on GPU 1.

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
# RL (no teacher)
uv run rl @ configs/v1/training_mode/rl.toml

# OPD (needs teacher on port 8001)
uv run rl @ configs/v1/training_mode/opd.toml
uv run rl @ configs/v1/training_mode/opd_lora.toml

# SFT hard distill (needs teacher on port 8001)
uv run rl @ configs/v1/training_mode/sft.toml
uv run rl @ configs/v1/training_mode/sft_lora.toml
```

See [docs/training.md](../../../docs/training.md#training-modes-rl--opd--sft-via-orchestrator)
for what each mode does.
