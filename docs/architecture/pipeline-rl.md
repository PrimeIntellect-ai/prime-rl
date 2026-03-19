# Pipeline RL

PRIME-RL supports a variety of popular RL training objectives for post-training language models: GRPO, GSPO, OPO, RLOO, and [CISPO](https://arxiv.org/abs/2506.13585).

## How It Works

At each step, we sample $N$ prompts from our dataset. For each prompt $x$, we sample a group of rollouts $\{y_i\}^G_{i=1}$ and use a verifier to assign scores $s_i$ to each $y_i$. These scores are turned into advantages using an advantage function, and the policy is updated using the chosen loss objective.

The default pipeline:

1. **Inference** generates rollouts from the current (or recent) policy
2. **Orchestrator** collects rollouts, computes advantages, and packs batches
3. **Trainer** updates the policy using the loss objective

## Loss Functions

The loss is computed per-sequence. The default loss uses masked importance sampling with KL against the inference policy. See [Custom Algorithms](../custom-algorithms.md) for how to plug in your own loss.

## Advantage Functions

Advantages are computed per-example (grouped by `rollouts_per_example`). The default advantage function uses reward minus per-example baseline (DR-GRPO without std normalization). See [Custom Algorithms](../custom-algorithms.md) for custom advantage functions.

## Async RL

By default, PRIME-RL runs asynchronous off-policy training to maximize throughput. See [Async RL](async-rl.md) for details on how this works and the loss objective that handles the distribution shift.
