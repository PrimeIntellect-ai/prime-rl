# Training Modes

PRIME-RL supports three training modes through our RL trainer, selected via `training_mode`:

- **`rl`** — reinforcement learning: student generates rollouts, no teacher
- **`opd`** — [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/): students generates rollouts, train to minimize the KL divergence between the student and teacher's logprobs for each token in the rollout
- **`sft`** — supervised fine-tuning on teacher-generated rollouts

> Note: PRIME-RL also has a dedicated `sft` entrypoint for more traditional supervised fine-tuning from a HF dataset. When using the `sft` training mode on the orchestrator, teacher rollouts are generated on-the-fly and used for training.

The mode determines who generates rollouts, what role the teacher plays, and what must be configured.

| Mode | Student | Teacher |
|---|---|---|
| `rl` | required | forbidden |
| `opd` | required | required (local vLLM) |
| `sft` | required | required (any OAI-compatible endpoint) |

**SFT vs OPD teachers** differ in what the orchestrator asks of them. SFT only calls `/v1/chat/completions` to generate rollouts — any OpenAI-compatible endpoint works (PI inference, OpenAI, Anthropic, a local vLLM). OPD additionally needs token-level logprobs scored over the student's tokens, which today only vLLM's `/inference/v1/generate` with `prompt_logprobs` exposes — so the OPD teacher must be a vLLM server.

### Reference configs

Debug-scale configs for all three modes (and LoRA variants) live in [`configs/debug/training_modes/`](../configs/debug/training_modes/):

- `rl.toml` / `opd.toml` / `opd_lora.toml`
- `sft.toml` / `sft_lora.toml` (local vLLM teacher)
- `sft_external.toml` (PI inference teacher)

See [`configs/debug/training_modes/README.md`](../configs/debug/training_modes/README.md) for run commands.

## Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_mode` | `"rl"` | One of `rl`, `opd`, `sft`. Propagates to `orchestrator.training_mode` and (for sft) `trainer.loss.type`. |
| `deployment.num_teacher_gpus` | `None` | Number of GPUs for the teacher vLLM server. Auto-starts when set. OPD only. |
| `trainer.loss.teacher_tau` | `0.0` | Distillation strength. Must be `> 0` in OPD. |
| `trainer.loss.adv_tau` | `1.0` | Weight for the RL advantage signal. Set `0` for pure distillation. |
| `orchestrator.verification.enabled` | `true` | Enable/disable verification. Set to `false` for pure distillation with `adv_tau = 0`. |
