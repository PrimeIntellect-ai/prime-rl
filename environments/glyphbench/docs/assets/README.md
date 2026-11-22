# README assets

## Game animations

`gallery/` contains nine original game recordings used in the README gallery, one
per suite. They show sample trajectories rendered from the Unicode grid, without
the legend or status display. These illustrate the interface, not trained-policy
performance. The recordings retain their original frames and timing; the README
sets their display height to 160 pixels.

The gallery uses Snake, MultiRoom, Corridor, ChopTrees, Space Invaders, CoinRun,
Breakout, CraftaxFull, and NetHack. Use `scripts/record_random_gifs.py --grid-only`
to record similar trajectories.

## Paper figures

Figures from *GlyphBench: A Playground for Language Model Reinforcement Learning*.

| File | Source in the paper |
|---|---|
| `overview.png` | Figure 1, `glyphbench_figure1_aligned_arrows.pdf`, rasterized at 2,400 pixels wide. |
| `multitask-learning.png` | `multitask_learning.pdf`, rasterized at 2,000 pixels wide. |
| `replay-ui.png` | `replay_ui_bankheist.png`, the original replay illustration. |

The learning figure reports one Qwen3.5-4B run on 100 tasks through update 250.
The training curve includes a trailing nine-step mean; fixed evaluation uses
new seeds of the same tasks. Reasoning Gym bars use 9,300 attempts per model
at each output-token budget.
