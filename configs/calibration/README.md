# Calibration pack

This pack checks one handshake before longer GPQA debate runs:

1. `gpqa_debate_5step.toml` trains debate for five steps and saves
   `outputs/isambard/calibration/gpqa_debate_5step/weights/step_5`.
2. `gpqa_single_shot_eval_after_ckpt.toml` loads that saved checkpoint as a
   normal vLLM model and scores open-ended GPQA in a single-turn setting.

`gpqa_debate_5step_bs512_g16.toml` is the throughput profile for a 16-GPU
allocation: one train node plus three TP=4 inference replicas, `batch_size=512`,
and vLLM throughput scheduler knobs.

Run both from the Isambard checkout after binding the shell to the repo:

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl-main
source scripts/env/activate-prime-rl.sh
```

Inside a held two-node allocation, launch the debate calibration plainly:

```bash
HF_HUB_OFFLINE=0 uv run --no-sync rl @ configs/calibration/gpqa_debate_5step.toml
```

For the 16-GPU throughput profile, hold four nodes and launch:

```bash
HF_HUB_OFFLINE=0 uv run --no-sync rl @ configs/calibration/gpqa_debate_5step_bs512_g16.toml
```

After `weights/step_5/STABLE` exists, run the single-shot eval from a GPU node:

```bash
HF_HUB_OFFLINE=0 uv run --no-sync --env-file .env baseline-eval \
  --config configs/calibration/gpqa_single_shot_eval_after_ckpt.toml
```

For reruns, prefer overriding both the debate `--output-dir` and the eval
`--model`/`--output-dir` so the checkpoint and eval artifact remain paired.
