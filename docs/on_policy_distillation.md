# On-Policy Distillation

On-policy distillation uses a teacher model to provide dense token-level feedback during RL training. The student generates rollouts, and the teacher's logprobs guide the student to stay close to stronger behavior while still learning from rewards.

For more details, see [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) by Thinking Machines.

## Quick Start

Add `num_teacher_gpus` to `[deployment]` and set `teacher_tau > 0`:

```toml
[deployment]
num_teacher_gpus = 2

[trainer.loss]
teacher_tau = 0.5
```

This automatically starts a teacher inference server using the same model as inference. To use a different teacher model:

```toml
[deployment]
num_teacher_gpus = 2

[teacher_inference.model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

[trainer.loss]
teacher_tau = 0.5
```

## Using an External Teacher Server

If the teacher is already running elsewhere:

```toml
[trainer.loss]
teacher_tau = 0.5

[orchestrator.teacher_model.client]
base_url = ["http://teacher-server:8000/v1"]

[orchestrator.teacher_model.model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

## Pure Distillation (No Verification)

For agentic environments where verification is expensive (code execution, tool use, multi-turn interactions), you can skip verification entirely and use only the teacher signal:

```toml
[deployment]
num_teacher_gpus = 2

[trainer.loss]
teacher_tau = 1.0
adv_tau = 0.0  # Disable reward-based learning

[orchestrator.verification]
enabled = false  # Skip expensive verification
```

This runs pure on-policy distillation: the student learns to match the teacher without needing any reward signal.

## SFT Distillation ("Hard Distillation") From Teacher Rollouts

Use this mode when you want to train from teacher-generated completions directly (hard distillation), without teacher token-level logprobs.

```toml
[trainer.loss]
type = "sft"

[orchestrator]
use_token_client = false

[orchestrator.teacher_rollout_model.client]
base_url = ["https://your-openai-compatible-endpoint/v1"]
skip_model_check = true

[orchestrator.teacher_rollout_model.model]
name = "teacher-model-name"
```

In this mode:
- Rollouts are generated from `orchestrator.teacher_rollout_model`
- The orchestrator uses text-level reconstruction with the student tokenizer
- The RL trainer optimizes masked NLL (`trainer.loss.type = "sft"`)
- Omit `[inference]` (no local inference server required)

### Image Input (VLM) Support

Yes, image input is supported in SFT/hard-distillation mode when the student model is multimodal (VLM).

- Prompts can include OpenAI-style image items in `message.content`, e.g. `{"type": "image_url", "image_url": {"url": "data:image/..."}}`
- The orchestrator extracts and preprocesses images from trajectory prompts and attaches `pixel_values`/`image_grid_thw` to training samples
- No teacher token IDs/logprobs are required; reconstruction still happens from messages

Notes:
- This path currently expects `data:image/...` payloads in message content
- The teacher rollout endpoint still needs to be able to handle the same multimodal prompts during generation

Reference configs:
- `configs/alphabet_sort/sft_distill_hard_qwen4b_lora_prime_teacher.toml`
- `examples/alphabet_sort/sft_distill_hard.toml`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `deployment.num_teacher_gpus` | `None` | Number of GPUs for teacher server. Auto-starts server when set. |
| `trainer.loss.teacher_tau` | `0.0` | Distillation strength. Set `> 0` to enable. |
| `trainer.loss.adv_tau` | `1.0` | Weight for RL advantage signal. Set `0` for pure distillation. |
| `orchestrator.verification.enabled` | `true` | Enable/disable verification. Set to `false` for pure distillation with `adv_tau = 0`. |

## Monitoring

The `teacher_kl` metric shows the KL divergence from teacher to student. Lower values mean the student is closer to the teacher.
