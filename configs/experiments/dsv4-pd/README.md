# DeepSeek-V4-Flash P/D disaggregation: stress test and tuning

Experiment record for [PR #4565](https://github.com/PrimeIntellect-ai/prime-rl/pull/4565) (P/D for DeepSeek-V4-Flash with
online MXFP8). This branch also merges the NVFP4 inference work of
[PR #4564](https://github.com/PrimeIntellect-ai/prime-rl/pull/4564) for the P/D + NVFP4 runs. Results are in
`results/summary.md` and `NOTES.md` (the running log); paths in the scripts point at the original experiment directory
(`/home/phi/exp/dsv4-pd`).

## Workload

PI-DeepSWE (113 tasks) with the RLM harness at reasoning effort max (passed as `chat_template_kwargs`, since vllm-router
0.2.4 rejects a top-level `max`), group size 16, 24 h rollout timeout, Prime sandboxes: `configs/stress/stress_eval.toml`.
Concurrency is pinned through the eval's control API by `scripts/stress_driver.py`, which records per-minute engine
metrics (generated / prompt / recomputed prefill tokens, GPU and external cache hits, queues, latencies).

## Layout

- `configs/stress/`: the equal-compute arms on 2 nodes (`agg`: 2x aggregated DP8+EP, `pd`: 1P:1D, `pdmc`: 1P:1D + 1 TB
  Mooncake on prefill) and the eval config.
- `configs/tune/`: tuning arms (prefill/scheduler step size, 3P:1D vs 4x aggregated on 4 nodes, P/D + NVFP4).
- `configs/pd/`: P/D bring-up configs (dummy weights, 1-node split, 2-node, Mooncake, HiSparse, AIME26 eval).
- `router-templates/_launch_router.sh.j2`: the router launch used by every arm (circuit breaker off, retries, lenient
  health checks); the arms point `slurm.template_path` at a template copy that includes it.
- `scripts/`: `stress_driver.py` (staircase driver), `level_loop.sh` / `tune_round.sh` (one fresh eval burst per level),
  `analyze_stress.py` / `analyze_tune.py` (per-level, per-GPU tables), `build_report.py` + `report/template.html`
  (the shared report page), `pd_smoke.py` / `mooncake_*check.py` (P/D and Mooncake correctness checks).
