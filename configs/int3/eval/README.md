Configs and start commands for running INTELLECT-3 evals.

To run the (default) debug eval (should be super quick, just 50 rollouts on Qwen3-0.6B), run:

```bash
sbatch configs/int3/eval/run.sh
```

To run the mini evals (~50 rollouts at 2k context) on GLM-4.5-Air, run:

```bash
MODEL_NAME=zai-org/GLM-4.5-Air EVAL_CONFIG=configs/int3/eval/mini-eval.toml sbatch configs/int3/eval/run.sh
```

To run the quick evals (~500 rollouts at 32k context) on GLM-4.5-Air, run:

```bash
MODEL_NAME=zai-org/GLM-4.5-Air EVAL_CONFIG=configs/int3/eval/quick-eval.toml sbatch configs/int3/eval/run.sh
```

To run the full evals (~2000 rollouts at 80k context) on GLM-4.5-Air, run:

```bash
MODEL_NAME=zai-org/GLM-4.5-Air EVAL_CONFIG=configs/int3/eval/full-eval.toml sbatch configs/int3/eval/run.sh
```

Further:
- To get more nodes, use the `--nodes` or `-N` SLURM flag. We will automatically parallelize the evaluation across all nodes.
- We automatically push evaluation results to the W&B project `intellect-3-evals`.
- We automatically push rollouts as private datasets to the HF Hub using format `PrimeIntellect/<MODEL_NAME>-<EVAL_MODE^>-Evals` with one subset per eval environment.
