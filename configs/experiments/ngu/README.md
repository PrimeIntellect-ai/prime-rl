# NGU SWE experiment

Train `PrimeIntellect/GLM-4.5-Air-Scaleswe` on 800 SWE-rebench tasks, 200 per difficulty tier. The tiers rank 987 tasks solved by stored initial-policy rollouts; their mean pass rates are 99.7%, 73.2%, 42.6%, and 13.3%. Evaluate only SWE-bench Verified and a disjoint 500-task SWE-rebench holdout every 20 steps.

| Config | Sampling | Batch target |
|---|---|---|
| `train-static.toml` | K=8 (48 complete groups) | 384 |
| `train-ngu.toml` | Start at K=4; add 4 after an all-failure round with probability .8 | 384 |

Both use 2 trainer + 6 TP8/EP inference H200 nodes, Muon LR 3e-6, default IPO (.3 epsilon, zero KL), a 0.1 output/input/turn length penalty, `max_off_policy_steps=64`, and W&B project `ngu-ablations`. Compare equal GPU-hours.

```bash
export PYTHONPATH="$PWD/tools/ngu/tasksets${PYTHONPATH:+:$PYTHONPATH}"
uv run rl @ configs/experiments/ngu/train-static.toml --slurm.project-dir "$PWD"
uv run rl @ configs/experiments/ngu/train-ngu.toml --slurm.project-dir "$PWD"
```

`eval.toml` and `inference.toml` are the original profiling configs. Required taskset code and pinned manifests live in `tools/ngu/`.
