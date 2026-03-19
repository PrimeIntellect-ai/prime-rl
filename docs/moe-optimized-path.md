# Optimized MoE Path

Prime-RL keeps the `main` MoE compute path as the baseline:

- `model.moe_use_grouped_mm = true`
- Torchtitan expert parallel when `model.ep > 1`

The opt-in MoE optimization layer now only covers the routed-token helpers that showed a clear win on top of that baseline: Triton routing, scatter, gather, and the fused routed-path reorder.

## Config knobs

The optional overrides live under `[model.moe_optim]`.

```toml
[model]
moe_use_grouped_mm = true

[model.moe_optim]
routing = "triton"
scatter = "triton"
gather = "triton"
routed_ffn = "fused"
```

Available backends:

- `routing`: `torch`, `triton`
- `scatter`: `torch`, `triton`
- `gather`: `torch`, `triton`
- `routed_ffn`: `torch`, `fused`

Notes:

- `moe_use_grouped_mm` keeps the existing grouped-mm baseline from `main`. Set it to `false` only for debugging or microbenchmark comparisons.
- Explicit Triton selections are treated as hard requirements and raise if Triton or CUDA support is unavailable.
- `routed_ffn = "fused"` currently fuses scatter-index computation with token scattering and is intended to be used with the Triton routed path.

## Example config

Use `configs/debug/moe/sft/optimized_path.toml` as the optimized reference config. It keeps the baseline grouped-mm/EP path and adds:

- `flash_attention_3`
- Triton routing, scatter, and gather
- fused routed MoE path
- `ep = 8`

Hardware notes:

- `flash_attention_3` is intended for Hopper-class GPUs.
- If you are not on Hopper, use `flash_attention_2` or `sdpa` and keep the same MoE overrides.

## Comparison configs

Use `configs/qwen3_8b/sft_compare.toml` as the shared base for Qwen3-30B-A3B comparisons.

Compose it with one of these overrides:

- `configs/qwen3_8b/sft_compare_baseline.toml`: the strongest pre-change baseline, keeping grouped MM, compile, and the default torch routed helpers
- `configs/qwen3_8b/sft_compare_optimized.toml`: the same trainer setup plus Triton routing, scatter, gather, and `routed_ffn = "fused"`

Example commands:

```bash
uv run sft @ configs/qwen3_8b/sft_compare.toml @ configs/qwen3_8b/sft_compare_baseline.toml
uv run sft @ configs/qwen3_8b/sft_compare.toml @ configs/qwen3_8b/sft_compare_optimized.toml
```

## Authoritative H200 procedure

Treat benchmark numbers as authoritative only after entering the target H200 shell and confirming the hostname.

1. Start the benchmark shell with `srun --jobid=3286 --pty bash`.
2. Confirm the prompt shows `fares@ltc-idc3-hgx8-h200-11`, or run `whoami && hostname` and confirm `fares` on `ltc-idc3-hgx8-h200-11`.
3. `cd /home/fares/prime-rl`
4. Run the baseline compare config.
5. Run the optimized compare config.
6. Compare the JSON outputs written by the two runs.

Example session:

```bash
srun --jobid=3286 --pty bash
hostname
cd /home/fares/prime-rl
uv run sft @ configs/qwen3_8b/sft_compare.toml @ configs/qwen3_8b/sft_compare_baseline.toml
uv run sft @ configs/qwen3_8b/sft_compare.toml @ configs/qwen3_8b/sft_compare_optimized.toml
```
