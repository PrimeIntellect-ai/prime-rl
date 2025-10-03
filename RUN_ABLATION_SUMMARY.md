# Granite Reverse-Text RL Ablation Snapshot

## What We Ran
1. Warmed `ibm-granite/granite-3.0-1b-a400m-instruct` for 100 SFT steps to create a steady reverse-text starter. 
2. Extended RL for 50 GRPO steps under three settings: router replay + recompute, no replay + recompute, and no replay + no recompute.
3. Benchmarked each checkpoint with the reverse-text evaluator (1000 prompts × 3 rollouts).

## Training (50 GRPO steps)
- Replay + recompute: mean 0.1663, peak 0.2639 @ step 34.
- No replay + recompute: mean 0.1436, peak 0.2597 @ step 46.
- No replay + no recompute: mean 0.1372, peak 0.2671 @ step 45.

## Evaluation (reverse-text environment)
- Replay + recompute: 0.227 ± 0.083
- No replay + recompute: 0.223 ± 0.084
- No replay + no recompute: 0.231 ± 0.085

## Checkpoints
- replay+recompute → rewardhacker00/granite-reverse-text-rl-50
- no-replay+recompute → rewardhacker00/granite-reverse-text-rl-noreplay-50
- no-replay+no-recompute → rewardhacker00/granite-reverse-text-rl-norecompute-50

## Visual
- “Granite RL Reward Trajectories” shows all three curves on one plot (router replay offers slightly steadier convergence, but all land in the same reward band).

## Next
- Repeat the same ablation recipe on the Wordle environment with the Granite 1B checkpoint, then scale to larger Granite variants if gains remain marginal.
