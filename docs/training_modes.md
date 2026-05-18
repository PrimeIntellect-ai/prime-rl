# Training Modes

prime-rl supports three training modes, selected via `orchestrator.training_mode`:

- **`rl`** — standard reinforcement learning from rewards
- **`opd`** — on-policy distillation: RL with an extra KL term toward a teacher's logprobs
- **`sft`** — hard distillation: supervised fine-tuning on teacher-generated rollouts

The mode determines who generates rollouts, what role the teacher plays, and what must be configured.

## Mode comparison

| | **rl** | **opd** | **sft** |
|---|---|---|---|
| **Student does** | generate rollouts → get trained on them | generate rollouts → get trained on them | get trained on teacher's rollouts; optionally serve inference for evals |
| **Teacher does** | nothing (must be unset) | score student rollouts (token-level logprobs) | generate rollouts |
| **Loss** | reward-based (advantage) | reward + KL to teacher logprobs (`teacher_tau > 0`) | pure NLL on teacher tokens (hard distill) |
| **Student inference** (`[inference]`) | **required** | **required** | **optional** — only if you want evals or weight-sync the student |
| **Teacher inference** (`[teacher_inference]`) | forbidden | **required, must be vLLM** | not used (teacher is external) |
| **`[orchestrator.teacher]`** | must be `None` | auto-wired from `[teacher_inference]` | **required** — `client.base_url` + `model.name` of external endpoint |
| **`num_teacher_gpus`** | unset | **required** (`> 0`) | unset (teacher is external) |
| **Teacher endpoint type** | n/a | **local vLLM only** | **any OpenAI-compatible** (PI inference, OpenAI, Anthropic, local vLLM…) |
| **Weight sync (trainer → ?)** | → student inference | → student inference (teacher frozen) | → student inference if configured; teacher never touched |
| **Evals** | student | student | only if `[inference]` is set (then student evals) |

## Key implications

**OPD's teacher cannot be an external API.** `compute_teacher_logprobs` (`src/prime_rl/orchestrator/utils.py`) calls vLLM's `/inference/v1/generate` with `prompt_logprobs=1`. That endpoint is vLLM-specific; PI inference, OpenAI, etc. return 404. For OPD, set `num_teacher_gpus` and let `[teacher_inference]` spin up a local vLLM.

**SFT's teacher is just chat completions.** It only needs `/v1/chat/completions`. Point `[orchestrator.teacher.client]` at anything OpenAI-compatible. No local GPU needed for the teacher.

**SFT student inference is optional but enabling it changes behavior.** If you set `[inference]`, you get (a) student-side evals during training, and (b) weight sync from trainer to student inference (so the student inference pool reflects training progress). Without `[inference]`, the run is teacher-rollout-only with no online evals.

**RL forbids any teacher.** Even a stray `[orchestrator.teacher]` block fails validation.

**Student model name is always the model being trained.** In SFT this is *not* the rollout-generating model — that's the teacher. The student model field still determines tokenizer, trainer init weights, and what gets saved as checkpoints.

## Minimal config per mode

```toml
# rl
training_mode = "rl"
[inference]
```

```toml
# opd
training_mode = "opd"
[deployment]
num_teacher_gpus = 1
[orchestrator.teacher]      # empty block; auto-wired from teacher_inference
[trainer.loss]
teacher_tau = 0.5
[inference]
# Override [teacher_inference.model] to use a different teacher model.
```

```toml
# sft
training_mode = "sft"
[orchestrator.teacher.client]
base_url = ["https://api.pinference.ai/api/v1"]
[orchestrator.teacher.model]
name = "qwen/qwen3-30b-a3b-instruct-2507"
[inference]                 # optional — drop if you don't want student-side evals
```

See [On-Policy Distillation](on_policy_distillation) for OPD/SFT-specific details (pure distillation, VLM support, parameters).
