# On-Policy Distillation

On-policy distillation uses a teacher model to provide dense token-level feedback during RL training. The student generates rollouts, and the teacher's logprobs guide the student to stay close to stronger behavior while still learning from rewards.

For more details, see [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) by Thinking Machines.

## Quick Start

Add `teacher_gpu_ids` and set `teacher_tau > 0`:

```toml
teacher_gpu_ids = [2, 3]

[trainer.loss]
teacher_tau = 0.5
```

This automatically starts a teacher inference server using the same model as inference. To use a different teacher model:

```toml
teacher_gpu_ids = [2, 3]

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
teacher_gpu_ids = [2, 3]

[trainer.loss]
teacher_tau = 1.0
adv_tau = 0.0  # Disable reward-based learning

[orchestrator.buffer]
skip_verification = true  # Skip expensive verification
```

This runs pure on-policy distillation: the student learns to match the teacher without needing any reward signal.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher_gpu_ids` | `None` | GPUs for teacher server. Auto-starts server when set. |
| `trainer.loss.teacher_tau` | `0.0` | Distillation strength. Set `> 0` to enable. |
| `trainer.loss.adv_tau` | `1.0` | Weight for RL advantage signal. Set `0` for pure distillation. |
| `orchestrator.buffer.skip_verification` | `false` | Skip verification. Use with `adv_tau = 0`. |

## Monitoring

The `teacher_kl` metric shows the KL divergence from teacher to student. Lower values mean the student is closer to the teacher.
