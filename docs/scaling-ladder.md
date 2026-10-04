# Scaling Ladder

A scaling ladder trains one pretraining recipe at several widths, fits how the loss falls with compute, and projects the loss of a larger target run before it is launched. Two tools in `tools/` do this; the formulas live in `prime_rl/utils/scaling.py`.

## Generate the rungs

```bash
uv run python tools/scaling_ladder.py pretrain.toml ladder/ --widths 512 768 1024 1536 2048 --tokens-per-param 20
```

`pretrain.toml` is an ordinary `sft` config. Its `model.name` is the base shape. For each width the tool writes:

- `ladder/<name>-d<width>/model/config.json`: the base HF config at that width. Heads, KV heads, layers and the MLP / expert / shared-expert intermediate sizes scale linearly with the width (MLP sizes round to multiples of 128). Head dim, vocabulary, number of experts and top-k stay fixed.
- `ladder/<name>-d<width>/sft.toml`: an overlay with the rung's model (`model.init = "scratch"`, tokenizer of the base model), `data.batch_size`, `max_steps`, `optim.lr` / `betas2` / `eps`, a WSD schedule (`scheduler.type = "linear"` with warmup and decay), `run.name` and the W&B group.
- `ladder/ladder.json`: one row per rung for the fit.

`--layers` sets each rung's depth (one value per width) and `--batch-tokens` sets its tokens per batch; both default to the rules below.

It prints the ladder:

```
    size layers  batch    steps  tokens  active   total    FLOPs       lr
    d512     12     64     4020   1.05B   52.7M    775M 1.46e+18 3.93e-03
     ...
   d2048     48    512    26040   54.6B   2.73B   30.5B 1.52e+21 1.76e-03
```

`active` is the active non-embedding parameter count and sets the token budget (`tokens = tokens_per_param * active`). `total` includes every expert and the input embedding. `FLOPs` is the training compute with the trainer's MFU estimator (`prime_rl.utils.flops.forward_flops`, three times the forward pass).

Launch a rung with the recipe and its overlay:

```bash
uv run sft @ pretrain.toml @ ladder/pretrain-d1024/sft.toml
```

## Hyperparameters

`Heuristic` in `prime_rl/utils/scaling.py` sets every rung's hyperparameters from the token budget `D`, the width `d` and the tokens per batch `B`:

| Quantity | Formula |
|---|---|
| tokens per batch | `B = 6.6 · D^0.5`, rounded to a power-of-two number of sequences |
| peak LR | `min(0.05, 0.0876 · D^-0.346 · d^-0.345 · B^0.5)` |
| Adam β2 | `clip(0.999^(B / 131072), 0.95, 0.9999)` |
| Adam ε | `9.68e-18 · sqrt(D / B)` |
| schedule | 1% linear warmup, constant, linear decay over the last 20% to 5% of the peak LR |

The LR applies to every parameter group. With `optim.type = "muon"` the Muon and AdamW groups share one LR (Muon runs with `adjust_lr = "rms_norm"`), and β2 / ε apply to the AdamW part.

The coefficients are placeholders. The LR, β2 and ε terms come from Marin's MoE AdamW fit at sequence length 8192, and the batch term is a square-root rule. Refit them from LR and batch sweeps of your recipe before relying on the ladder.

## Fit and project

Once at least three rungs have finished:

```bash
uv run python tools/fit_scaling_law.py ladder/ladder.json <entity>/<project>
```

Each rung's W&B run is found by its name. The fit uses the held-out `val/loss` when every run logs it, else the train `loss/mean` averaged over a window of 1% of the steps; `--metric` picks another key (e.g. `val/loss/<source>`). All rungs share one schedule shape. So at each fraction `f` of training the finished rungs' losses are fitted as `L_f(C) = E + A · C^-α` over training compute `C`. The tool prints the final-loss fit and, for every unfinished rung, the projected loss at each fraction next to the loss it has logged so far. Add the target run's width to `--widths` to get its config and its projection.
