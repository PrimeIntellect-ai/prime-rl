# Training Modes

prime-rl supports three training modes, selected via `training_mode`:

- **`rl`** — standard reinforcement learning from rewards
- **`opd`** — on-policy distillation: RL with an extra KL term toward a teacher's logprobs ([Thinking Machines blog post](https://thinkingmachines.ai/blog/on-policy-distillation/))
- **`sft`** — hard distillation: supervised fine-tuning on teacher-generated rollouts

The mode determines who generates rollouts, what role the teacher plays, and what must be configured.

## Mode comparison

| | **rl** | **opd** | **sft** |
|---|---|---|---|
| **Student does** | generate rollouts → get trained on them | generate rollouts → get trained on them | serve inference (for evals + weight sync); get trained on teacher's rollouts |
| **Teacher does** | nothing (must be unset) | score student rollouts (token-level logprobs) | generate rollouts |
| **Loss** | reward-based (advantage) | reward + KL to teacher logprobs (`teacher_tau > 0`) | pure NLL on teacher tokens (hard distill) |
| **Student inference** (`[inference]`) | **required** | **required** | **required** |
| **Teacher inference** (`[teacher_inference]`) | forbidden | **required, must be vLLM** | not used (teacher is external) |
| **`[orchestrator.teacher]`** | must be `None` | auto-wired from `[teacher_inference]` | **required** — `client.base_url` + `model.name` of external endpoint |
| **`num_teacher_gpus`** | unset | **required** (`> 0`) | unset (teacher is external) |
| **Teacher endpoint type** | n/a | **local vLLM only** | **any OpenAI-compatible** (PI inference, OpenAI, Anthropic, local vLLM…) |
| **Weight sync (trainer → ?)** | → student inference | → student inference (teacher frozen) | → student inference (teacher frozen) |
| **Evals** | student | student | student |

## Key implications

**OPD's teacher cannot be an external API.** `compute_teacher_logprobs` (`src/prime_rl/orchestrator/utils.py`) calls vLLM's `/inference/v1/generate` with `prompt_logprobs=1`. That endpoint is vLLM-specific; PI inference, OpenAI, etc. return 404. For OPD, set `num_teacher_gpus` and let `[teacher_inference]` spin up a local vLLM.

**SFT's teacher is just chat completions.** It only needs `/v1/chat/completions`. Point `[orchestrator.teacher.client]` at anything OpenAI-compatible. No local GPU needed for the teacher.

**RL forbids any teacher.** Even a stray `[orchestrator.teacher]` block fails validation.

**Student model name is always the model being trained.** In SFT this is *not* the rollout-generating model — that's the teacher. The student model field still determines tokenizer, trainer init weights, and what gets saved as checkpoints.

## Minimal config per mode

```toml
# rl
training_mode = "rl"
[inference]
```

```toml
# opd — auto-launched local teacher
training_mode = "opd"
[deployment]
num_teacher_gpus = 1            # spin up a teacher vLLM
[orchestrator.teacher]          # empty block; client + model auto-wired
[trainer.loss]
teacher_tau = 0.5
[inference]
# Override [teacher_inference.model] to use a different teacher model than the student.
```

```toml
# sft — external teacher (PI inference)
training_mode = "sft"
[orchestrator.teacher.client]
base_url = ["https://api.pinference.ai/api/v1"]
[orchestrator.teacher.model]
name = "qwen/qwen3-30b-a3b-instruct-2507"
[inference]
```

## OPD details

### Using an external (already-running) teacher

Skip `num_teacher_gpus` and point at the existing endpoint. The teacher **must** be a vLLM server (for the `/inference/v1/generate` + `prompt_logprobs` endpoint):

```toml
training_mode = "opd"
[trainer.loss]
teacher_tau = 0.5

[orchestrator.teacher.client]
base_url = ["http://teacher-server:8000/v1"]

[orchestrator.teacher.model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### Pure distillation (no verification)

For agentic environments where verification is expensive (code execution, tool use, multi-turn interactions), skip verification and use only the teacher signal:

```toml
training_mode = "opd"
[deployment]
num_teacher_gpus = 2

[trainer.loss]
teacher_tau = 1.0
adv_tau = 0.0                   # disable reward-based learning

[orchestrator.verification]
enabled = false                 # skip expensive verification
```

The student learns to match the teacher without needing any reward signal.

### Monitoring

The `teacher_kl` metric shows the KL divergence from teacher to student. Lower means the student is closer to the teacher.

## SFT details

### VLM support

Image input is supported in SFT mode when the student is a VLM:

- Prompts can include OpenAI-style image items in `message.content`, e.g. `{"type": "image_url", "image_url": {"url": "data:image/..."}}`
- The orchestrator extracts and preprocesses images from trajectory prompts and attaches `pixel_values` / `image_grid_thw` to training samples
- No teacher token IDs / logprobs are required; reconstruction still happens from messages

Notes:
- This path currently expects `data:image/...` payloads in message content
- The teacher rollout endpoint must also handle the same multimodal prompts during generation

### Reference configs

- `configs/reverse_text/debug_sft.toml` (local vLLM teacher)
- `configs/reverse_text/debug_sft_external.toml` (PI inference teacher)

## Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_mode` | `"rl"` | One of `rl`, `opd`, `sft`. Propagates to `orchestrator.training_mode` and (for sft) `trainer.loss.type`. |
| `deployment.num_teacher_gpus` | `None` | Number of GPUs for the teacher vLLM server. Auto-starts when set. OPD only. |
| `trainer.loss.teacher_tau` | `0.0` | Distillation strength. Must be `> 0` in OPD. |
| `trainer.loss.adv_tau` | `1.0` | Weight for the RL advantage signal. Set `0` for pure distillation. |
| `orchestrator.verification.enabled` | `true` | Enable/disable verification. Set to `false` for pure distillation with `adv_tau = 0`. |
