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

### Example: PPO Clipped Loss

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

### Configuration

```toml
[loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"
kwargs = { clip_eps = 0.2 }
```

---

## 2. Custom Advantage Functions

Advantages are computed **per-group** (one example × N rollouts). You provide a function that computes advantages for a single group; the framework calls it once per group and stitches the results back together. Groups may have fewer than `group_size` rollouts when some rollouts in the group errored (partial-group training).

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
    # All rollouts for a single example (one group).
    rollouts: list[vf.RolloutOutput]
```

Each `vf.RolloutOutput` carries the full rollout (`reward`, `trajectory`, etc.), so custom advantages can read any metadata they need (e.g. completion-token counts, turn counts, tool calls).

#### AdvantageOutputs

```python
@dataclass
class AdvantageOutputs:
    advantages: list[float]   # one entry per rollout in the input group
```

### Example: Normalized Advantage

```python
import statistics
from prime_rl.orchestrator.advantage import AdvantageInputs, AdvantageOutputs

def normalized_advantage(inputs: AdvantageInputs, eps: float = 1e-8) -> AdvantageOutputs:
    """Normalize advantages to zero mean and unit variance within the group."""
    rewards = [r["reward"] for r in inputs.rollouts]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return AdvantageOutputs(advantages=[(r - mean) / (std + eps) for r in rewards])
```

### Configuration

```toml
[advantage]
type = "custom"
import_path = "my_module.normalized_advantage"
kwargs = { eps = 1e-8 }
```

---

## Default Implementations

If no custom function is specified:

- **Loss**: Uses `default_loss_fn` (masked importance sampling with KL against the inference policy, and optional masking strategies)
- **Advantage**: Uses `default_advantage_fn` (reward minus per-example baseline, a.k.a. DR-GRPO without std normalization)

See `LossConfig` and `AdvantageConfig` for available parameters.

## Tips

- Your functions receive structured inputs via dataclasses with jaxtyping annotations
- Return metrics as scalars or 1D tensors - they'll be aggregated automatically
- Use the `loss_mask` / tensor shapes to handle variable-length sequences
- Test your custom functions with the provided test patterns before training
