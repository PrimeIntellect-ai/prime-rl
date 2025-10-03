# Granite Reverse-Text RL Ablation Snapshot

## Training (50 GRPO steps)
- Replay + recompute: mean reward 0.1663, peak 0.2639 @ step 34.
- No replay + recompute: mean 0.1436, peak 0.2597 @ step 46.
- No replay + no recompute: mean 0.1372, peak 0.2671 @ step 45.

## Evaluation (reverse-text environment)
- 1k examples × 3 rollouts:
  - Replay + recompute: 0.227 ± 0.083
  - No replay + recompute: 0.223 ± 0.084
  - No replay + no recompute: 0.231 ± 0.085

## Deployed Checkpoints
- Replay + recompute → rewardhacker00/granite-reverse-text-rl-50
- No replay + recompute → rewardhacker00/granite-reverse-text-rl-noreplay-50
- No replay + no recompute → rewardhacker00/granite-reverse-text-rl-norecompute-50

## Visual
- "Granite RL Reward Trajectories" compares training reward curves for all three runs.
