# Debate run configs (generated)

Self-contained debate run configs are produced from factored fragments by a
generator, so the **debater × judge × schedule** matrix has no duplication and a
new model/judge/schedule is a one-file change.

## Layout
- `base.toml` — shared run + protocol invariants (tables only; no identity/envs/judge).
- `debaters/<name>.toml` — the trained policy: `[model]`, `[trainer.model]` (LoRA
  targets), `[deployment]` (topology), `[inference]` (MoE/routed-experts/kernels),
  `[orchestrator.train.sampling]`. The heavy, per-model block.
- `judges/<name>.toml` — `[orchestrator.multi_agent.fixed.judge]` (winner → reward).
- `grader-deepseek.txt` — the ONE GT-correctness grader (DeepSeek), injected into
  every config's single-agent eval. Change it here, regenerate, all runs update.
- `gen.py` — the generator + the schedule spec.
- `generated/` — emitted, launch-ready configs (committed). **Build artifacts —
  never hand-edit; edit a fragment or `gen.py` and regenerate.**

## Use
```
uv run python configs/debate/gen.py                 # regenerate all configs
uv run rl @ configs/debate/generated/qwen35-a3b__qwen9b-or__pcd4.toml   # launch one
```

## Add to the matrix
- New judge → drop `judges/<name>.toml` (model + base_url + sampling), regenerate.
- New debater → drop `debaters/<name>.toml` (the per-model block; also confirm
  LoRA targets, MoE-vs-dense knobs, topology, sampling, and that the reasoning
  parser / vLLM patches cover the architecture), regenerate.
- New schedule → add a `(prompts_ref, schedule)` entry to `SCHEDULES` in `gen.py`.

Metric keys are env-name-unified, so runs overlay on shared keys (twc_3way,
position_bias, single-agent GT-acc); compare per-member turn/slot counts only
within-schedule (they scale with the 4–6 slot differences). Debate-eval
`group_size` is fixed at 8 across schedules for comparable twc.
