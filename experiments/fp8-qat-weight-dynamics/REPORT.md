# SFT Weight Dynamics Under FP8 Quantization

## Context

A DeepSeek-V4-Flash SFT checkpoint served in FP8 lost most of its fine-tuning. The bf16 base sits exactly on the FP8 grid, and most SFT updates are smaller than half an FP8 bin, so round-to-nearest serving maps them back to the base value.

## Questions

1. Does FP8 quantization-aware training (QAT) push more SFT updates across FP8 bin boundaries? In QAT the forward pass uses the FP8-rounded weight and the gradient passes straight through to the fp32 master. That could make each weight's update direction persist longer, and could carry weights across more bins if the minima of the loss seen through FP8-rounded weights lie further from the start than the bf16 minima.
2. Does starting on the FP8 grid or off it change how many updates survive? An on-grid start (the base snapped to the UE8M0 grid) is the DSv4 situation. An off-grid start (the original bf16 base) has weights at random positions within their bins.

## What we ran

| setting        | value                                                                                               |
|----------------|-----------------------------------------------------------------------------------------------------|
| model          | Qwen3-30B-A3B-Base                                                                                  |
| data           | `PrimeIntellect/INTELLECT-3-SFT-10K`, math split, identical order in every run                      |
| length         | 1000 steps at 8k x 16 tokens per step                                                               |
| optimizer      | Muon, lr 1e-5 with cosine decay to 1e-6, weight decay 0.01                                          |
| hardware       | one node of 8 H200s per run                                                                         |
| starts         | off-grid (original bf16 base), on-grid (UE8M0 copy)                                                 |
| training modes | bf16, `qat-ue8m0` (power-of-two weight scales), `qat-amax` (`amax/448` weight scales, H200 default) |
| FP8 coverage   | attention q/k/v/o and all routed experts                                                            |

The six runs are the product of the two starts and the three training modes.

| run | start on UE8M0 grid | train     | W&B run                                                                                                                              |
|-----|---------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------|
| 1   | True                | bf16      | [pilot-q30b-base_ongrid-ue8m0_train-bf16_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/yea2f7rm) |
| 2   | False               | bf16      | [q30b-base_offgrid_train-bf16_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/jvg7v18b)            |
| 3   | True                | qat-ue8m0 | [q30b-base_ongrid-ue8m0_train-qat-ue8m0_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/zwui1swj)  |
| 4   | False               | qat-ue8m0 | [q30b-base_offgrid_train-qat-ue8m0_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/ww43b8cq)       |
| 5   | True                | qat-amax  | [q30b-base_ongrid-ue8m0_train-qat-amax_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/phc49c2i)   |
| 6   | False               | qat-amax  | [q30b-base_offgrid_train-qat-amax_muon-1e-5_1000st_8k-x16](https://wandb.ai/primeintellect/fp8-weight-dynamics/runs/zkrnl5vq)        |

The on-grid copy snaps the FP8-covered weights to the UE8M0 grid with the trainer's own cast, which changes 92.7% of them by 2.7% RMS and leaves the starting loss unchanged. Each run's fp32 master weights were exported at steps 250, 500, 750 and 1000 and compared with the run's own start. Serving is simulated with 128x128 block FP8 e4m3 casts on two grids: UE8M0 (power-of-two scales) and `amax/448`.

## Data

All numbers are over the 29.9B FP8-covered weights at step 1000.

- `P(bf16 Δ)`: share of weights whose bf16 value differs from the start.
- `P(fp8 Δ)`: share of weights whose served value differs from the served start.
- `P(fp8 Δ | bf16 Δ)`: share of bf16 changes still visible after serving.
- `P(fp8 Δ | no bf16 Δ)`: share of weights unchanged in bf16 whose served value changes anyway.
- `update_kept`: the fraction of the true update the served model carries along the update's direction, `sum(u * Δ) / sum(Δ^2)`, with `Δ` the fp32 master minus the start and `u` the served final minus the served start. It is unbounded: 1 means the served model carries the full update on average, 0 means none of it, above 1 overshoots, and below 0 points against the update. `P(fp8 Δ | bf16 Δ)` counts weights, while `update_kept` weighs them by size: off-grid, few weights cross a bin but each crossing jumps a full bin, so `update_kept` stays near 1.
- Own grid (marked `*`): the FP8 grid an arm trained against, UE8M0 for `qat-ue8m0` and `amax/448` for `qat-amax`. Serving on it uses the same quantizer as the training forward pass. bf16 arms have no own grid and are shown on both.

Round-to-nearest serving:

| start on UE8M0 grid | train     | grid       | P(bf16 Δ) | P(fp8 Δ) | P(fp8 Δ∣bf16 Δ) | P(fp8 Δ∣no bf16 Δ) | update_kept |
|---------------------|-----------|------------|-----------|----------|-----------------|--------------------|-------------|
| True                | bf16      | UE8M0      | 0.818     | 0.158    | 0.193           | 0                  | 0.407       |
| True                | bf16      | amax/448   | 0.818     | 0.438    | 0.480           | 0.249              | 0.820       |
| True                | qat-ue8m0 | UE8M0 *    | 0.826     | 0.169    | 0.204           | 0                  | 0.438       |
| True                | qat-amax  | amax/448 * | 0.820     | 0.444    | 0.486           | 0.254              | 0.823       |
| False               | bf16      | UE8M0      | 0.815     | 0.240    | 0.294           | 0                  | 0.998       |
| False               | bf16      | amax/448   | 0.815     | 0.438    | 0.485           | 0.228              | 0.999       |
| False               | qat-ue8m0 | UE8M0 *    | 0.817     | 0.241    | 0.295           | 0                  | 0.999       |
| False               | qat-amax  | amax/448 * | 0.817     | 0.439    | 0.485           | 0.231              | 0.999       |

The `amax/448` rows differ systematically from the UE8M0 rows. The Serving grid part of Analysis explains why.

Stochastic-rounding serving: means over 3 seeds (they agree to better than 1e-4). Stochastic rounding casts the start and the final weights with independent draws, so an off-grid weight that never changed still lands on a different grid point with probability `2p(1-p)` for a fractional bin position `p`. That averages to 1/3, which is the 0.332 in `P(fp8 Δ | no bf16 Δ)`.

| start on UE8M0 grid | train     | grid       | P(fp8 Δ∣bf16 Δ) | P(fp8 Δ∣no bf16 Δ) | update_kept |
|---------------------|-----------|------------|-----------------|--------------------|-------------|
| True                | bf16      | UE8M0      | 0.297           | 0                  | 1.000       |
| True                | qat-ue8m0 | UE8M0 *    | 0.307           | 0                  | 1.000       |
| True                | qat-amax  | amax/448 * | 0.594           | 0.449              | 1.000       |
| False               | bf16      | UE8M0      | 0.451           | 0.332              | 1.000       |
| False               | qat-ue8m0 | UE8M0 *    | 0.451           | 0.332              | 1.000       |
| False               | qat-amax  | amax/448 * | 0.601           | 0.487              | 1.000       |

### Per-family weight shifts

On-grid start, bf16 training, UE8M0 round-to-nearest serving:

| family      | elements | P(bf16 Δ) | P(fp8 Δ∣bf16 Δ) | update_kept |
|-------------|----------|-----------|-----------------|-------------|
| expert.gate | 9.66B    | 0.817     | 0.193           | 0.405       |
| expert.up   | 9.66B    | 0.819     | 0.193           | 0.404       |
| expert.down | 9.66B    | 0.819     | 0.194           | 0.409       |
| attn.q      | 0.40B    | 0.739     | 0.173           | 0.412       |
| attn.k      | 0.05B    | 0.816     | 0.211           | 0.475       |
| attn.v      | 0.05B    | 0.807     | 0.204           | 0.461       |
| attn.o      | 0.40B    | 0.824     | 0.202           | 0.432       |
| ALL         | 29.90B   | 0.818     | 0.193           | 0.407       |

### Update size relative to bin width

RMS of `Δ` over all FP8-covered weights, and the share of bf16-changed weights by `|Δ|/bin`, with the bin width taken from the start's UE8M0 grid:

| start on UE8M0 grid | train     | RMS Δ    | <0.1  | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-2   | >2    |
|---------------------|-----------|----------|-------|----------|----------|-------|-------|-------|
| True                | bf16      | 2.63e-04 | 0.342 | 0.312    | 0.161    | 0.091 | 0.047 | 0.047 |
| True                | qat-ue8m0 | 2.84e-04 | 0.327 | 0.311    | 0.166    | 0.095 | 0.050 | 0.050 |
| True                | qat-amax  | 2.67e-04 | 0.339 | 0.312    | 0.162    | 0.091 | 0.047 | 0.048 |
| False               | bf16      | 2.63e-04 | 0.329 | 0.315    | 0.165    | 0.093 | 0.048 | 0.049 |
| False               | qat-ue8m0 | 2.64e-04 | 0.329 | 0.315    | 0.165    | 0.093 | 0.048 | 0.049 |
| False               | qat-amax  | 2.64e-04 | 0.329 | 0.315    | 0.165    | 0.093 | 0.048 | 0.049 |

Whether `P(fp8 Δ | bf16 Δ)` under UE8M0 round-to-nearest serving can be read off this table depends on the start:

- On-grid start: a weight sits on a grid point, half a bin from the rounding boundary, so it changes its served value if and only if `|Δ|/bin > 0.5`, the last three columns. For bf16 these sum to 0.185, against a measured 0.193. The excess comes from block scales that moved.
- Off-grid start: a weight sits at a random position in its bin, so it crosses with probability about `min(|Δ|/bin, 1)`, and every column contributes. Weighting each bucket by its midpoint gives 0.301 for bf16, against a measured 0.294.

The table does not apply to `amax/448` serving, whose bins differ, or to stochastic rounding, where the served value depends on random draws rather than on a fixed boundary.

### Update persistence

During training, a tracker followed a fixed random sample of 65,536 elements in each of 8 expert weight tensors (one data-parallel rank's shard of the stacked gate/up and down projections at layers 0, 16, 32 and 47). After each optimizer step `t` it forms the single-step update on those elements, `u_t = w_t - w_{t-1}` with `w` the fp32 master, and records the cosine between updates `k` steps apart, `cos_k = (u_t · u_{t-k}) / (|u_t| |u_{t-k}|)`. A higher cosine at lag `k` means update directions persist longer. Muon's momentum (0.95) makes short lags high in every arm, so persistence differences show at long lags. Mean over steps 200 to 800 and the expert tensors (the 8 tracked attention tensors show the same pattern):

| start on UE8M0 grid | train     | lag 1 | lag 4 | lag 16 | lag 64 |
|---------------------|-----------|-------|-------|--------|--------|
| True                | bf16      | 0.938 | 0.782 | 0.412  | 0.046  |
| True                | qat-ue8m0 | 0.939 | 0.788 | 0.426  | 0.066  |
| True                | qat-amax  | 0.938 | 0.783 | 0.414  | 0.049  |
| False               | bf16      | 0.938 | 0.783 | 0.413  | 0.047  |
| False               | qat-ue8m0 | 0.938 | 0.783 | 0.413  | 0.046  |
| False               | qat-amax  | 0.938 | 0.783 | 0.413  | 0.047  |

### Training loss

Mean training loss over 50-step windows. Every run sees the same batches in the same order, and QAT runs report the loss of their FP8 forward pass:

| start on UE8M0 grid | train     | steps 1-50 | steps 451-500 | steps 951-1000 |
|---------------------|-----------|------------|---------------|----------------|
| True                | bf16      | 0.6203     | 0.3900        | 0.3081         |
| True                | qat-ue8m0 | 0.6387     | 0.4152        | 0.3599         |
| True                | qat-amax  | 0.6258     | 0.3953        | 0.3214         |
| False               | bf16      | 0.6194     | 0.3898        | 0.3080         |
| False               | qat-ue8m0 | 0.6211     | 0.3909        | 0.3103         |
| False               | qat-amax  | 0.6206     | 0.3911        | 0.3105         |

## Analysis

1. **QAT: barely any effect on which updates survive.** QAT changes `P(fp8 Δ | bf16 Δ)` and `update_kept` by about 0.03 at most, though from the on-grid start it moves the master weights up to 8% further.
2. **Start: it decides how much of the update survives.** Under UE8M0 round-to-nearest serving, the on-grid start keeps `update_kept` near 0.41 in every training mode, against near 1 off-grid.

### QAT (question 1)

From the on-grid start, `qat-ue8m0` raises `P(fp8 Δ | bf16 Δ)` on its own grid from 0.193 to 0.204 and `update_kept` from 0.407 to 0.438, and `qat-amax` moves `amax/448` `update_kept` from 0.820 to 0.823. From the off-grid start, both QAT arms match bf16. QAT from the on-grid start does make updates persist somewhat longer: with Muon's fixed per-step size, `qat-ue8m0` moves its masters 8% further (RMS `Δ` 2.84e-4 against 2.63e-4) and `qat-amax` 1.4% further, but that is too small to change survival much. The update-persistence table points the same way (see Other observations). Off-grid, all three arms move within 0.2% of each other. The `|Δ|/bin` histogram shows the same small shift: on-grid `qat-ue8m0` puts 0.195 of its changes at half a bin or more, against 0.185 for bf16.

### Start (question 2)

In every training mode, the on-grid start keeps `update_kept` near 0.41 under UE8M0 round-to-nearest serving, against near 1 off-grid, which reproduces the DSv4 erasure at this scale. The `|Δ|/bin` histogram shows the mechanism: about 82% of changes are smaller than half a bin, and from an on-grid start round-to-nearest erases exactly those. Stochastic rounding gives `update_kept` 1.000 in every arm, and the pattern holds in every weight family (`update_kept` 0.40 to 0.48).

UE8M0 round-to-nearest serving, with the start on ("on") or off ("off") the UE8M0 grid:

| train     | P(fp8 Δ∣bf16 Δ) on | P(fp8 Δ∣bf16 Δ) off | P(fp8 Δ) on | P(fp8 Δ) off | update_kept on | update_kept off |
|-----------|--------------------|---------------------|-------------|--------------|----------------|-----------------|
| bf16      | 0.193              | 0.294               | 0.158       | 0.240        | 0.407          | 0.998           |
| qat-ue8m0 | 0.204              | 0.295               | 0.169       | 0.241        | 0.438          | 0.999           |
| qat-amax  | 0.195              | 0.295               | 0.160       | 0.241        | 0.412          | 0.998           |

### Serving grid

The `amax/448` rows behave differently from the UE8M0 rows, for three reasons:

- `P(fp8 Δ | no bf16 Δ)`: a shift in a block's largest weight changes its `amax/448` scale by a non-power-of-two factor, so the whole block re-rounds on a shifted grid. 26 to 32% of blocks change scale, and `P(fp8 Δ | no bf16 Δ)` is 0.23 to 0.25. UE8M0 scales change only by powers of two, which keep every e4m3 value representable, so UE8M0 has none.
- `P(fp8 Δ | bf16 Δ)` and `P(fp8 Δ)`: the same re-rounding also moves some changed weights to a new grid point, so `P(fp8 Δ | bf16 Δ)` is about 0.48 for either start. `P(fp8 Δ)` mixes the two conditionals, `P(fp8 Δ) = P(bf16 Δ) P(fp8 Δ | bf16 Δ) + (1 - P(bf16 Δ)) P(fp8 Δ | no bf16 Δ)`, so it rises with them.
- `update_kept`: the on-grid start sits on the UE8M0 grid, not the `amax/448` one, so under `amax/448` serving it is mostly off-grid and `update_kept` is 0.82, against 0.41 on UE8M0. Re-rounding moves weights in directions unrelated to their updates, which adds noise to `update_kept` but no bias, so off-grid `update_kept` stays at 0.999.

### Other observations

- Update persistence: five of the six arms have identical update cosines at every lag, to within 0.001. On-grid `qat-ue8m0` alone sits above them, by a gap that grows with lag: +0.006 at lag 4, +0.014 at lag 16 and +0.020 at lag 64. So QAT lengthens persistence only when its grid matches an on-grid start, and only slightly. These differences look too small on their own to account for the 8% larger displacement, so most of that likely comes from slow drift beyond lag 64, which the tracker does not measure.
- Training loss follows the erasure: from the on-grid start, QAT ends with mean loss 0.013 higher for `qat-amax` and 0.052 higher for `qat-ue8m0` than bf16, while every other arm is within 0.003. That is the same ordering as `update_kept` on each arm's own grid. A QAT run trains through its own quantizer, so it suffers the erasure during training, not only at serving time.
- QAT from an off-grid start is close to free: both off-grid QAT arms match bf16 in `P(fp8 Δ | bf16 Δ)`, `update_kept`, displacement and persistence, and end with mean loss within 0.003 of bf16.

## Limitations

- One seed per arm.
- About 130K tokens per step against 34.6M in the DSv4 SFT, so each step carries far more gradient noise.
- The math split runs out near step 830, so the last ~170 steps repeat data in every arm.
- One model, Muon only, and H200 only. On SM100 the trainer uses UE8M0 scales for activations and gradients too.
- Serving is simulated with casts, not run through vLLM.

## Open questions

**Hypothesis: stochastic rounding at serving time helps when the base sits on the serving grid and hurts slightly when it does not.** From an on-grid start, round-to-nearest erases most of the update (`update_kept` 0.41), while stochastic rounding keeps all of it on average and adds rounding noise only to weights that moved. From an off-grid start, round-to-nearest already keeps the update (`update_kept` 0.998), so stochastic rounding adds noise to every weight, doubling the rounding variance, with nothing to recover.

**Test:** compare held-out loss of the step-1000 exports with weights fake-quantized under round-to-nearest and under stochastic rounding (several seeds), for both starts.
