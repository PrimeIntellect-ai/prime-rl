## Runs

### general-agent — Qwen3-4B-Instruct (single node, local solver)

Single-node RL run on the `general-agent-solver-local` environment with Qwen3-4B-Instruct, split 4 train / 4 infer GPUs on one H200 node.

```bash
uv run rl @ configs/general_agent/rl_qwen3_4b.toml
```

### general-agent — Qwen3-30B-A3B-Instruct (two nodes, local solver)

Two-node RL run (1 train + 1 infer, 8+8 GPUs) on the same env with the Qwen3-30B-A3B MoE. Trainer uses full-node expert parallel (`ep=8`); inference uses `tp=8`.

```bash
uv run rl @ configs/general_agent/rl_qwen3_30b_a3b.toml
```

### general-agent-behavior-learning — Qwen3-4B-Instruct RLM ablations

Four matched RLM runs for prompt and behavior-shaping ablations. They train `Qwen/Qwen3-4B-Instruct-2507` on a single node split into 4 train GPUs and 4 inference GPUs, with `batch_size = 256`, `rollouts_per_example = 8`, and `max_retries = 1`. The baseline uses the current RLM prompt, the prompt run loads explicit IPython/programmatic-control guidance from `behavior_learning/prompts/extended.md`, the behavior-shaping run gates judge rewards on solved rollouts using `openai/gpt-5-mini` through Prime inference, and the combined run enables both.
See `configs/behavior_learning/README.md` for the GPT-5.5 discovery walkthrough, uploaded eval links, and curated behavior evidence.

```bash
uv run rl @ configs/behavior_learning/rl_qwen3_4b_rlm_baseline.toml
uv run rl @ configs/behavior_learning/rl_qwen3_4b_rlm_prompt.toml
uv run rl @ configs/behavior_learning/rl_qwen3_4b_rlm_behavior_shaping.toml
uv run rl @ configs/behavior_learning/rl_qwen3_4b_rlm_prompt_behavior_shaping.toml
```

The behavior-shaping configs use `behavior_judge_model = "openai/gpt-5-mini"` and `behavior_reward_alpha = 1.0`. The judge provider defaults to Prime inference with `PRIME_API_KEY`, the behavior reward is solution-gated, and the environment fails early if the key is missing.

### general-agent — Qwen3.5-35B-A3B (four nodes, local solver)

Four-node RL run (2 train + 2 infer, 16+16 GPUs) on the same env with the Qwen3.5-35B-A3B MoE.

```bash
uv run rl @ configs/general_agent/rl_qwen35_35b_a3b.toml
```

### general-agent — Qwen3-30B-A3B-Thinking (four nodes, RLM solver)

Four-node RL run (2 train + 2 infer, 16+16 GPUs) on the `general-agent-solver-rlm` env with the Qwen3-30B-A3B Thinking-2507 MoE.

```bash
uv run rl @ configs/general_agent/rl_qwen3_30b_a3b_thinking.toml
```

### general-agent — Qwen3-30B-A3B-Thinking + length penalty (four nodes, RLM solver)

Same setup as the Thinking run above with correctness-gated length shaping enabled (`[orchestrator.advantage] length_shaping = true`) — shorter correct rollouts get up to 2× advantage amplification.

```bash
uv run rl @ configs/general_agent/rl_qwen3_30b_a3b_thinking_lp.toml
```

### general-agent — NemotronH-Nano-30B-A3B (Instruct, four nodes, local solver)

Four-node RL run (2 train + 2 infer, 16+16 GPUs) on the `general-agent-solver-local` env with the **Instruct** (non-base) NemotronH-Nano-30B-A3B. Trainer uses FA2 with `ep=8` (cp disabled — `cp_style="ring"` is rejected for Mamba/SSM models and `ulysses` is unnecessary at this scale) at `seq_len=65536`. Inference at `dp=2, tp=4` with `max_model_len=131072` (BFCL prompts overflow at 32k). Rollouts go through MITO (`use_token_client=false`, `use_renderer=false`) with the `nano_v3` reasoning parser and `qwen3_coder` tool parser server-side. In-loop eval on `bfcl-v3` and `mcp-atlas` every 50 steps.

```bash
uv run rl @ configs/general_agent/rl_nemotron_nano.toml
```

### SFT — NemotronH-Nano-30B-A3B Base (two nodes)

Two-node SFT of the **Base** Nemotron-3-Nano-30B-A3B (NemotronH hybrid MoE) on the `general_agent` subset of `PrimeIntellect/INTELLECT-5-SFT-Raw`. The non-base (Instruct) SFT was dropped — we RL directly from the Instruct checkpoint instead.

```bash
uv run sft @ configs/general_agent/sft_nemotron_nano_base.toml
```

### Standalone inference — NemotronH-Nano-30B-A3B (one node)

Single-node serving of the stock NemotronH-Nano weights (no SFT) — useful for baselines / synth. tp=8.

```bash
# Base
uv run inference @ configs/general_agent/infer_nemotron_nano_base.toml

# Regular
uv run inference @ configs/general_agent/infer_nemotron_nano.toml
```

### Standalone inference (GLM-5.1-FP8 disaggregated)

8-node disaggregated deployment (4 prefill + 4 decode) with CPU KV cache offloading. Not coupled to the RL runs above — use when serving GLM-5.1 for external clients or synth.

```bash
uv run inference @ configs/general_agent/infer_glm51.toml
```

### RLM discovery eval

Generate and run the GPT-5.5 discovery eval used to mine solved RLM behavior examples:

```bash
uv run vf-eval configs/behavior_learning/eval_rlm_gpt55_discovery.toml
```

## General tips

- To generate the SLURM script without submitting, add `--dry-run`.
- To start from scratch, add `--clean-output-dir` to wipe the output directory.
- Tail logs emitted by `--dry-run`:
  - Trainer: `tail -F <output_dir>/logs/trainer.log`
  - Orchestrator: `tail -F <output_dir>/logs/orchestrator.log`
  - Inference: `tail -F <output_dir>/logs/inference.log`
  - Per-env rollouts: `tail -F <output_dir>/logs/envs/train/general-agent-solver-local/*.log`
