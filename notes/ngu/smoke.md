# NGU integration smoke — 2026-09-18

Config: [`configs/debug/ngu.toml`](../../configs/debug/ngu.toml). Model: `PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`. Binary reward is reverse-text similarity ≥0.8, implemented in [`tools/ngu/reverse_text.py`](../../tools/ngu/reverse_text.py). K=4, continuation probability .875, inclusive history age 4, batch target 16.

The native RL launcher ran inference, environment servers, orchestrator and trainer inside one H200 SLURM allocation. One GPU trained and one served inference; the single-node template reserved the node exclusively. No SWE training was launched.

## Fresh run

Job **743** completed with exit code 0 after five optimizer updates. The batches contained **16, 20, 20, 20, 16** trained traces. Seventeen cohorts trained, including five expanded cohorts: four with eight attempts and one with twelve. The metric windows recorded ten dispatched retry rounds and one give-up.

Artifact checks verified that each trained visit occupied one optimizer update, retained advantages summed to zero, positive advantages equaled `1 - successes/attempts`, trained policy age was at most four, and episode records had unique IDs. The trainer reported finite losses and gradients for all five updates. Both trainer and orchestrator wrote the final step-5 checkpoint and exited successfully.

The step-5 orchestrator checkpoint contained nine active visits, including five with committed history: 44 attempts and 28 retained payloads. Three finalized cohorts were queued. This state exercises continuation and queued-cohort recovery on resume.

## Resume

Job **744** resumed step 5, completed optimizer updates **6 and 7**, wrote the final step-7 checkpoint and exited with code 0. Its first metric window reported all nine restored visits. The resume trained eight additional cohorts (32 traces), including restored queued work.

The combined artifact check passed for **277 unique episode records**, **25 shipped cohorts** and **124 trained traces** across seven updates. The five expanded cohorts remained intact in a single update each; all trained visits passed the centering, historical positive-scale and policy-age checks. The full receipt is `NGU_VERIFICATION.json` in the run directory. Both allocations are released.

## Reproduction

```bash
uv run rl @ configs/debug/ngu.toml \
  --slurm.project-dir "$PWD" \
  --output-dir "$PWD/outputs/ngu-smoke" \
  --run.name reverse-text-ngu-checkpoint-20260918

uv run rl @ configs/debug/ngu.toml \
  --slurm.project-dir "$PWD" \
  --output-dir "$PWD/outputs/ngu-smoke" \
  --run.name reverse-text-ngu-checkpoint-20260918 \
  --resume.step 5 --max-steps 7
```

Use a fresh run name when repeating the first command. Runtime artifacts are under `outputs/ngu-smoke/reverse-text-ngu-checkpoint-20260918/`, with separate logs/configs for each attempt and an append-only trace stream.

## Local checks

`uv run --no-sync pytest tests/unit/orchestrator tests/unit/test_configs.py -q`: **295 passed, 1 skipped**. Ruff check/format and native config dry runs passed. Focused NGU tests cover historical anchoring, expiry, independent visits, give-up, incomplete rounds, token eviction, configuration validation, continuation RNG restoration, and lossless wire-episode checkpoint serialization including routed experts.

The GPU test caught a native wire-episode pickling failure. Checkpoints now serialize episode data explicitly, disabling float rounding and preserving training arrays; a regression test uses the actual wire types. This smoke validates the implementation and resume path, not an accuracy advantage over static GRPO or the six-node GLM deployment.
