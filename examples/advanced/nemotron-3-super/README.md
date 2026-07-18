# Nemotron-3-Super

Multi-node SWE RL for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` — a hybrid-Mamba MoE run
at 131k context across 4 trainer + 1 inference node (ulysses CP, expert parallel).

| config | task |
|---|---|
| `swe.toml` | SWE (ScaleSWE + SWE-Bench-Verified) |

```bash
uv run rl @ examples/advanced/nemotron-3-super/swe.toml
```

Requires the `mamba-ssm` dependency group; the config's `slurm.pre_run_command` syncs it.
