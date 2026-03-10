# Bring Your Own Algorithms

Prime-RL supports custom implementations for key algorithmic components, allowing you to experiment with different RL objectives and techniques.

## 1. Custom Loss Functions

The loss is computed **per-sequence** (per-sample). You provide a function that computes the loss for a single sequence, and the framework handles iteration and aggregation.

### Interface

```python
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def my_custom_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
    ...
```

#### LossInputs

```python
@dataclass
class LossInputs:
    trainer_logprobs: Float[Tensor, "seq"]      # Log probs from current policy
    inference_logprobs: Float[Tensor, "seq"]    # Log probs from reference policy
    teacher_logprobs: Float[Tensor, "seq"] | None  # Optional teacher log probs
    advantages: Float[Tensor, "seq"]            # Per-token advantages
    loss_mask: Bool[Tensor, "seq"]              # Mask for valid tokens
```

#### LossOutputs

```python
@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]         # Scalar loss for this sequence
    metrics: dict[str, Tensor]      # Metrics to log
```

### Built-in loss types

You can use these without writing custom code by setting `type` in `[trainer.loss]`:

| Type | Description | Key config |
|------|-------------|------------|
| `default` | IPO-style loss (DPPO-Binary TV + KL); supports teacher distillation | `ipo_mask_low`, `ipo_mask_high`, `adv_tau`, `teacher_tau`, `kl_tau` |
| `ppo_clip` | Token-level PPO clipped surrogate; optional KL penalty vs reference policy | `clip_eps` (default: 0.2), `kl_coef`, `entropy_coef` |
| `reinforce` | REINFORCE policy gradient: -log π(a\|s) * A | `entropy_coef` |
| `custom` | Your own function via `import_path` and `kwargs` | `import_path`, `kwargs` |

Example: PPO clip with optional KL penalty:

```toml
[trainer.loss]
type = "ppo_clip"
clip_eps = 0.2
kl_coef = 0.01
```

Example: REINFORCE:

```toml
[trainer.loss]
type = "reinforce"
```

### Example: Custom PPO-style loss (advanced)

If you need a variant not covered by the built-in `ppo_clip`, you can still plug in a custom loss:

```python
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    surr1 = ratio * inputs.advantages
    surr2 = clipped_ratio * inputs.advantages

    loss = -torch.min(surr1, surr2)[inputs.loss_mask].sum()

    return LossOutputs(
        loss=loss,
        metrics={"clip_frac": (ratio != clipped_ratio)[inputs.loss_mask].float().mean()},
    )
```

### Custom loss configuration

```toml
[loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"
kwargs = { clip_eps = 0.2 }
```

---

## 2. Custom Advantage Functions

Advantages are computed **per-example** (grouped by `rollouts_per_example`). You provide a function that computes advantages for a batch of examples.

### Interface

```python
from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs

def my_custom_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
    ...
```

#### AdvantageInputs

```python
@dataclass
class AdvantageInputs:
    rewards: Float[Tensor, "num_examples rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_examples rollouts_per_example"]
```

#### AdvantageOutputs

```python
@dataclass
class AdvantageOutputs:
    advantages: Float[Tensor, "num_examples rollouts_per_example"]
```

### Built-in advantage types

You can use these by setting `type` in the orchestrator’s advantage config (e.g. `[orchestrator.advantage]` or the config key your entrypoint uses):

| Type | Description | Key config |
|------|-------------|------------|
| `default` | Reward minus per-problem baseline (GRPO-style); optional length-weighted mean | `length_weighted_mean` |
| `normalized` | Zero-mean, unit-variance per problem | `eps` |
| `reinforce` | Raw rewards as advantages (no baseline); use with REINFORCE loss | — |
| `custom` | Your own function via `import_path` and `kwargs` | `import_path`, `kwargs` |

Example: normalized advantages:

```toml
[orchestrator.advantage]
type = "normalized"
eps = 1e-8
```

Example: REINFORCE (raw rewards):

```toml
[orchestrator.advantage]
type = "reinforce"
```

### Example: Custom normalized advantage (advanced)

```python
import torch
from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs

def normalized_advantage(inputs: AdvantageInputs, eps: float = 1e-8) -> AdvantageOutputs:
    """Normalize advantages to zero mean and unit variance per example."""
    mean = inputs.rewards.mean(dim=1, keepdim=True)
    std = inputs.rewards.std(dim=1, keepdim=True)
    advantages = (inputs.rewards - mean) / (std + eps)
    return AdvantageOutputs(advantages=advantages)
```

### Custom advantage configuration

```toml
[advantage]
type = "custom"
import_path = "my_module.normalized_advantage"
kwargs = { eps = 1e-8 }
```

---

## Default behavior

If you do not set a custom or built-in type:

- **Loss**: Uses `default` (IPO-style masked importance sampling with KL and optional teacher).
- **Advantage**: Uses `default` (reward minus per-example baseline).

See `LossConfig` and `AdvantageConfig` in the codebase for all parameters.

## Tips

- Your functions receive structured inputs via dataclasses with jaxtyping annotations
- Return metrics as scalars or 1D tensors - they'll be aggregated automatically
- Use the `loss_mask` / tensor shapes to handle variable-length sequences
- Test your custom functions with the provided test patterns before training
