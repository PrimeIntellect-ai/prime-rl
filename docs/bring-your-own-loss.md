# Bring Your Own Loss

Prime-RL supports custom loss functions, allowing you to experiment with different RL objectives beyond the default loss.

## How It Works

The loss is computed **per-sequence** (per-sample). You provide a function that computes the loss for a single sequence, and the framework handles:
- Iterating over all sequences in a batch
- Aggregating metrics across sequences
- Scaling the total loss

## Interface

Your custom loss function must accept `LossInputs` and return `LossOutputs`:

```python
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def my_custom_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
    ...
```

### LossInputs

```python
@dataclass
class LossInputs:
    trainer_logprobs: Float[Tensor, "seq"]      # Log probs from current policy
    inference_logprobs: Float[Tensor, "seq"]    # Log probs from reference policy
    teacher_logprobs: Float[Tensor, "seq"] | None  # Optional teacher log probs
    advantages: Float[Tensor, "seq"]            # Per-token advantages
    loss_mask: Bool[Tensor, "seq"]              # Mask for valid tokens
```

All tensors have shape `(seq,)` where `seq` is the sequence length for that sample.

### LossOutputs

```python
@dataclass
class LossOutputs:
    loss: Float[Tensor, ""]         # Scalar loss for this sequence
    metrics: dict[str, Tensor]      # Metrics to log (scalars or 1D tensors)
```

## Writing a Custom Loss

Here's an example of a simple PPO-style clipped loss:

```python
# my_project/losses.py
import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs

def ppo_clip_loss(inputs: LossInputs, clip_eps: float = 0.2) -> LossOutputs:
    """PPO clipped surrogate objective."""
    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    surr1 = ratio * inputs.advantages
    surr2 = clipped_ratio * inputs.advantages

    # Loss for this sequence (sum over valid tokens)
    loss = -torch.min(surr1, surr2)[inputs.loss_mask].sum()

    # Metrics to track
    metrics = {
        "clip_frac": (ratio != clipped_ratio)[inputs.loss_mask].float().mean(),
        "ratio_mean": ratio[inputs.loss_mask].mean(),
    }

    return LossOutputs(loss=loss, metrics=metrics)
```

## Configuration

To use a custom loss, configure it in your training config:

```toml
[loss]
path = "my_project.losses.ppo_clip_loss"
kwargs = { clip_eps = 0.2 }
```

Or in Python:

```python
from prime_rl.trainer.rl.config import CustomLossConfig

loss_config = CustomLossConfig(
    path="my_project.losses.ppo_clip_loss",
    kwargs={"clip_eps": 0.2},
)
```

The `path` is a fully-qualified Python import path to your loss function. The `kwargs` dict is unpacked and passed to your function.

## Default Loss

If you don't specify a custom loss, Prime-RL uses `LossConfig` which runs the default `prime_rl_loss`. This implements a GRPO-style objective with importance ratio clipping and various masking strategies. See the `LossConfig` class for all available parameters.

## Tips

- Your loss function receives one sequence at a time - don't worry about batching
- Return metrics as scalars (0-dim tensors) or 1D tensors - they'll be aggregated automatically
- The `loss_mask` indicates which tokens are valid (part of the response, not padding)
- Use `inputs.loss_mask` to filter tokens when computing your loss
