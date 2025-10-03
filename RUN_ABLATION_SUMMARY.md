# Granite Reverse-Text RL Ablation Snapshot

## What We Ran
1. Warmed `ibm-granite/granite-3.0-1b-a400m-instruct` for 100 SFT steps on reverse-text.
2. Ran 50 GRPO steps under three routing/logprob configurations.
3. Evaluated each checkpoint on the reverse-text benchmark with 1000 prompts × 3 rollouts.

## Metrics at a Glance
| Configuration | Train Mean | Train Peak (step) | Eval Avg | Eval Std |
| --- | ---: | --- | ---: | ---: |
| Replay + Recompute | 0.1663 | 0.2639 (34) | 0.227 | 0.083 |
| No Replay + Recompute | 0.1436 | 0.2597 (46) | 0.223 | 0.084 |
| No Replay + No Recompute | 0.1372 | 0.2671 (45) | 0.231 | 0.085 |

## Deployed Checkpoints
- Replay + recompute → `rewardhacker00/granite-reverse-text-rl-50`
- No replay + recompute → `rewardhacker00/granite-reverse-text-rl-noreplay-50`
- No replay + no recompute → `rewardhacker00/granite-reverse-text-rl-norecompute-50`

## Visual
- “Granite RL Reward Trajectories” overlays all three reward curves; replay keeps the trajectory steadier, but final rewards cluster closely.

## Next Up
- Port the same ablation recipe to Wordle with Granite 1B, then revisit with larger Granite variants to probe scale effects.
