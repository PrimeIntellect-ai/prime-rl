# Verification of the after-backward normalization

Compares main (7448986fc) against the fix (3e6758eb4) with the harness in the parent directory. Create detached worktrees `<prefix>-main` and `<prefix>-fix` at those commits, set `GN_WORKTREES=<prefix>`, and run each `wave_*.sh` on one 8-GPU node. Outputs go to `$GN_VERIFY_OUT` (default `./grad-norm-verify`).

- `wave_a.sh`, `wave_b.sh`: one-step fixed-data RL gradient dumps (single, CP 2, DP 2, with main repeats), the two `test_loss.py` tests, and SFT with 8 accumulation steps.
- `wave_c.sh` to `wave_f.sh`: fake samples with overlapping rl and ce weight streams (`GN_MC_MODE`), so the component counts differ and are not powers of two.
- `wave_g.sh`: the same code with its loss multiplied by c before backward and the gradients divided by c after (`GN_LOSS_RESCALE`), as a noise floor for moving a loss scale across backward.
- `wave_d.sh` also runs SFT with 6 accumulation steps (`sft_bs6.toml`).

Compare RL dumps with `../compare_grads.py` or `norm_ratio.py <out>/rl a:b ...`, and SFT runs with `sft_compare.py <out>/sft a:b ...`. The outputs used in the PR are in `results/`.
