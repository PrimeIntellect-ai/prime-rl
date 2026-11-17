---
name: recipes
description: Rules and playbook for the model × hardware RL recipes in `recipes/`, and the sweep harness in `benchmarks/recipes/` that fits them. Use when adding a recipe for a new model or hardware type, re-fitting a recipe after a stack change, or choosing trainer and inference parallelism for a model.
---

# Recipes

A recipe is a tested RL config for one model on one hardware type. It sets the infrastructure
(deployment, trainer parallelism, inference engine) and the model-specific settings (renderer,
parsers, precision). It sets no task: users add their environment as a second config.

```bash
uv run rl @ recipes/<model>/<hardware>.toml @ my-task.toml
```

## Rules

- **Model set.** Keep one recipe per builder per weight class, for the newest checkpoint in
  that class. When a builder releases a newer model in the same class, replace the recipe.
- **Context.** Fit every recipe to 256k (`seq_len = max_model_len = 262144`), so that it works
  for any agentic workload. Shorter context can run much faster; that is the user's trade-off.
- **Precision.** Use the checkpoint's native precision when both the trainer and vLLM support
  it on the target hardware. Otherwise use the nearest supported format and record the gap in
  `meta.toml`.
  - Trainer: `bf16`, blockwise `fp8` (SM90 and SM100), `mxfp8` (SM100 only). No NVFP4 or MXFP4.
  - H200 is SM90: no MXFP8, no NVFP4. FP8 checkpoints use blockwise FP8 on both sides.
- **One flat file per hardware.** `recipes/<model>/<hardware>.toml` holds the full config. Do
  not split it into a model base and a hardware overlay: hosted training takes one file.
- **Recipe and task never share a table.** A recipe sets `[model]`, `[deployment]`, `[slurm]`,
  `[weight_broadcast]`, `[trainer.model]` and `[inference]`. A task in `recipes/tasks/` sets
  `[orchestrator]`, the sources and `[trainer.optim]`. Then `cat recipe task > run.toml` gives a
  single file for `prime train run` that resolves the same as `@ recipe @ task`.
- **Document the launch command.** Each recipe's header holds
  `#   uv run rl @ recipes/<model>/<hardware>.toml @ recipes/tasks/<task>.toml`.
  `tests/unit/test_configs.py` parses every such stack.
- **Smallest fit.** Pick the fewest trainer nodes that fit one 262144-token sample with about
  10% memory headroom. Among configs of that size, pick the highest tok/s per GPU. For
  inference, pick the replica shape with the best output tok/s per GPU that serves the
  heavy-tailed workload without preemption.

## Fitting a recipe

Run each stage on dummy weights: loading real weights and FlashInfer autotuning can each take
longer than the whole sweep. Ask before you submit SLURM jobs; other teams have priority.

1. **Prior.** Read the model's `config.json`. Start from the nearest existing config in
   `examples/advanced/` or `configs/advanced/`. Those configs are 128k: double `cp` for 256k.
2. **Trainer fit.** Write `benchmarks/recipes/<model>/trainer.toml` (fake data, one
   262144-token sample per micro batch, `random_init`, `force_balanced_routing`) and a variants
   file, then run `benchmarks/recipes/trainer_fit.sh`. Read `summary.md`.
3. **Inference fit.** Write `inference.toml` (`load_format = "dummy"`) and a variants file, then
   run `benchmarks/recipes/inference_fit.sh`. It drives each server shape with
   `bench_inference.py` (multi-turn sessions, log-normal lengths, a tail at 256k).
4. **Smoke.** Run the recipe end to end with real weights for about 10 steps. This checks the
   weight broadcast and the real memory peaks, which dummy weights do not show.
5. **Write** `recipes/<model>/<hardware>.toml`, the measured numbers in `meta.toml`, the
   summaries in `benchmarks/recipes/<model>/results-<hardware>.md`, and a row in
   `recipes/README.md`. Check the composition with
   `uv run rl @ recipe @ recipes/tasks/swe.toml --dry-run`.

### Sweep lessons

- **Trainer cost is mostly compile.** Step 1 includes `torch.compile`; later variants reuse the
  inductor cache and take about half the wall time. Run the most likely winner first.
- **At 256k, attention dominates.** CP and offloading change throughput by less than 10%, so
  the trainer choice is mostly about memory headroom.
- **Peak memory is not monotonic in CP.** At cp = world size, DP is 1 and every rank runs every
  micro batch, which can cost more memory than cp / 2. Sweep at least cp ∈ {2, 4, 8}.
- **TP above the KV head count replicates the KV cache.** Skip those shapes.
- **vLLM DP shards MoE experts across DP ranks**, so `dp > 1` is one lockstep engine, not
  independent replicas.
- **Inference throughput has a peak in concurrency.** Below it, throughput equals the offered
  load and shapes look the same. Past it, the prefix cache thrashes and throughput falls. Raise
  concurrency until throughput falls and compare shapes at their peaks.
- **Server startup takes 7 to 10 min**, about two thirds of it Python imports from the shared
  filesystem. Keep the number of server variants small.

## Memory levers, cheapest first

Each lever costs throughput. Turn on only what the fit needs.

1. FSDP + EP shard the weights (`trainer.model.ep`).
2. CP shards activations along the sequence (`trainer.model.cp`).
3. Full activation checkpointing (`[trainer.model.ac] freq = 1`).
4. Activation offloading (`trainer.model.ac_offloading.max_inflight_activations`).
5. Optimizer offload (`trainer.model.offload = "optimizer"`, the default) or `"full"`.

## Known traps

- Nemotron-H has no CP: one 256k sample must fit on one rank.
- DeepSeek V4 supports only ring CP and needs router replay, which rules out disaggregated
  inference.
- The weight-update staging buffer needs free GPU memory on inference. DeepSeek V4 ran out of
  memory at `gpu_memory_utilization = 0.85`.
- Balanced routing understates trainer memory. Real routing is uneven; on Qwen3.6-35B-A3B the
  peak was 8–27 GiB above the sweep. Read the peak of every smoke step, not only step 1.
- The inference bench is closed-loop: below its saturation point it reports offered load, not
  capacity. The smoke run gives the real per-GPU rates. Record them and the trainer's
  `time/wait_for_batch` share, and set `num_infer_replicas` so the trainer stays busy.
- The logged MFU counts attention as non-causal. At long context it overstates real MFU by
  almost 2x.
