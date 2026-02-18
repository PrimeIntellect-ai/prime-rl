# SAPO (Soft Adaptive Policy Optimization)

SAPO is a smooth and adaptive policy-gradient method for reinforcement learning that replaces hard clipping with temperature-controlled soft gating for improved training stability and sample efficiency.

## Overview

SAPO addresses the brittleness of hard clipping in group-based policy optimization by introducing:

1. **Smooth Soft Gating**: Instead of hard clipping that zeroes gradients outside a fixed band, SAPO uses a smooth sigmoid-based gate that gradually attenuates off-policy updates
2. **Token-Level Adaptivity**: SAPO selectively down-weights individual off-policy tokens while preserving learning signals from near-on-policy tokens
3. **Asymmetric Temperature Control**: Different temperatures for positive and negative advantage tokens improve stability

## Mathematical Formulation

SAPO maximizes the following objective:

```
J(θ) = E[1/G * Σ(i=1 to G) 1/|y_i| * Σ(t=1 to |y_i|) f_i,t(r_i,t(θ)) * Â_i,t]
```

Where:
- `r_i,t(θ) = π_θ(y_i,t | q, y_i,<t) / π_θ_old(y_i,t | q, y_i,<t)` is the token-level importance ratio
- `f_i,t(x) = (4/τ_i,t) * σ(τ_i,t * (x - 1))` is the soft gate function
- `σ(x) = 1/(1 + e^(-x))` is the sigmoid function
- `τ_i,t = τ_pos` if `Â_i,t > 0`, else `τ_neg`

The soft gate creates a smooth trust region centered at `r = 1` (on-policy), with gradients weighted by:

```
w_i,t(θ) = 4 * p_i,t(θ) * (1 - p_i,t(θ))
```

where `p_i,t(θ) = σ(τ_i,t * (r_i,t(θ) - 1))`.

## Key Properties

### 1. Sequence-Level Coherence

Under common conditions (small on-policy steps, low intra-sequence variance), SAPO's token-level gates concentrate to a smooth sequence-level gate, similar to GSPO but with continuous trust region.

### 2. Token-Level Adaptivity

When sequences contain heterogeneous tokens, SAPO selectively down-weights only off-policy tokens while preserving gradients from near-on-policy tokens, improving sample efficiency.

### 3. Asymmetric Temperature Design

Using `τ_neg > τ_pos` causes gradients on negative tokens to decay more rapidly, addressing distinct stability profiles:
- **Positive tokens**: Update increases probability of sampled token
- **Negative tokens**: Update diffuses to many unsampled tokens, potentially causing instability

## Configuration

To use SAPO, set `type = "sapo"` in your loss configuration:

```toml
[loss]
type = "sapo"

# SAPO-specific parameters
tau_pos = 1.0    # Temperature for positive advantage tokens
tau_neg = 1.05   # Temperature for negative advantage tokens (recommended: tau_neg >= tau_pos)

# Shared parameters
adv_tau = 1.0
teacher_tau = 0.0  # Optional: for teacher KL divergence
kl_tau = 0.0       # Optional: for KL penalty
```

### Hyperparameters

- **`tau_pos`** (default: 1.0): Temperature controlling the decay rate of soft gate for positive advantage tokens. Larger values create faster decay.
  
- **`tau_neg`** (default: 1.05): Temperature for negative advantage tokens. Should typically be ≥ `tau_pos` for improved stability.

- **`adv_tau`** (default: 1.0): Scaling factor for advantages

- **`teacher_tau`** (default: 0.0): Scaling factor for teacher KL divergence (if using teacher model)

- **`kl_tau`** (default: 0.0): Scaling factor for KL penalty term

## Usage Example

### Basic RL Training with SAPO

```bash
# Using the example configuration
uv run rl @ examples/reverse_text/rl_sapo.toml
```

### Custom Configuration

```toml
[loss]
type = "sapo"
tau_pos = 1.0
tau_neg = 1.05
adv_tau = 1.0
```

## Comparison with Default Loss (AIPO)

| Feature | AIPO (Default) | SAPO |
|---------|----------------|------|
| Gating Mechanism | Hard clipping/masking | Smooth soft gate |
| Trust Region | Discrete bands | Continuous sigmoid |
| Gradient Behavior | Binary (in/out) | Smooth attenuation |
| Sample Efficiency | Lower (hard masks) | Higher (preserves signals) |
| Stability | Brittle with outliers | More robust |
| Asymmetric Design | No | Yes (τ_pos ≠ τ_neg) |

### When to Use SAPO

- **Training instability**: SAPO's smooth gating provides more stable updates
- **High variance ratios**: Especially in MoE models where token ratios vary widely
- **Sample efficiency**: SAPO preserves more learning signals from partially off-policy sequences
- **Long sequences**: Better handles heterogeneous tokens within sequences

### When to Use Default Loss

- **Simpler baselines**: AIPO is well-understood and simpler to tune
- **Legacy compatibility**: Existing configurations tuned for hard clipping

## Metrics

SAPO logs the following metrics to W&B:

- `soft_gate_weight`: Average soft gate weight across tokens (1.0 = fully on-policy, → 0 = heavily gated)
- `importance_ratio`: Mean token-level importance ratio
- `teacher_kl`: KL divergence to teacher model (if configured)

These metrics help monitor:
- How much gating is applied
- Policy divergence from behavior policy
- Alignment with teacher model

## Implementation Details

SAPO is implemented as a loss function in `src/prime_rl/trainer/rl/loss.py`:

```python
def sapo_loss_fn(inputs: LossInputs, loss_config: SAPOLossConfig) -> LossOutputs:
    # Compute importance ratio r_i,t(θ)
    token_importance_ratio = torch.exp(trainer_logprobs - inference_logprobs)
    
    # Select temperature based on advantage sign
    tau = torch.where(advantages > 0, tau_pos, tau_neg)
    
    # Compute soft gate: f_i,t(r) = (4/τ) * sigmoid(τ * (r - 1))
    gate_logit = tau * (token_importance_ratio - 1.0)
    soft_gate = (4.0 / tau) * torch.sigmoid(gate_logit)
    
    # Weighted loss
    loss = -(soft_gate * scaled_advantages.detach() * trainer_logprobs)[loss_mask].sum()
    ...
```

## References

- **Paper**: Gao et al., "Soft Adaptive Policy Optimization" (2025)
- **ArXiv**: [arXiv:2511.20347](https://arxiv.org/abs/2511.20347)
- **Used in**: Qwen3-VL model series by Alibaba

## Tips and Best Practices

1. **Start with defaults**: `tau_pos = 1.0`, `tau_neg = 1.05`
2. **Monitor soft_gate_weight**: Values near 1.0 indicate on-policy, values → 0 indicate heavy gating
3. **Adjust tau_neg for stability**: Increase if training is unstable (try 1.1-1.2)
4. **Combine with teacher models**: Set `teacher_tau > 0` for additional regularization
5. **Use with async training**: SAPO handles off-policy naturally, works well with `max_async_level > 1`

## Troubleshooting

**Training is still unstable**: Try increasing `tau_neg` (e.g., 1.1, 1.2) to more aggressively dampen negative gradients.

**Performance is worse than default**: Ensure `tau_neg >= tau_pos`. Try adjusting temperatures or check if your task benefits from soft vs hard gating.

**Metrics show very low soft_gate_weight**: Policy may be too off-policy. Reduce `max_async_level` or increase temperatures.

**Slow convergence**: Try decreasing temperatures (but stay > 0.5) to allow larger updates.

## Extending SAPO

SAPO can be combined with other techniques:

- **Custom loss functions**: Use SAPO as inspiration for custom loss implementations
- **Teacher distillation**: Set `teacher_tau > 0` to incorporate teacher guidance
- **Multi-task learning**: SAPO's stability benefits multi-task scenarios

See [bring-your-own-algorithms.md](bring-your-own-algorithms.md) for creating custom loss functions.
