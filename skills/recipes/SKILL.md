---
name: recipes
description: Rules and methodology for the model × entrypoint × hardware recipes in `recipes/`. Use when adding a recipe for a new model, entrypoint or hardware type, re-fitting a recipe after a stack change, or choosing trainer and inference parallelism for a model.
---

# Recipes

A recipe is a tested config for one model, one entrypoint (`rl`, `sft`) and one hardware
type. It sets the infrastructure (deployment, trainer parallelism, inference engine) and the
model-specific settings (renderer, parsers, precision). It sets no task or data. For RL, users
add their environment as a second config:

```bash
uv run rl @ recipes/<model>/rl/<hardware>.toml @ my-task.toml
```

The fitting method below covers RL. Only RL recipes exist so far.

## Rules

- **Model set.** Keep one recipe per builder per weight class, for the newest checkpoint in
  that class. When a builder releases a newer model in the same class, replace the recipe.
- **Context.** Fit every recipe to 256k (`seq_len = max_model_len = 262144`), so that it works
  for any agentic workload. Shorter context can run much faster; that is the user's trade-off.
- **Precision.** Use the checkpoint's native precision when both the trainer and vLLM support
  it on the target hardware. Otherwise use the nearest supported format and say so in the
  recipe header.
  - Trainer: `bf16`, blockwise `fp8` (SM90 and SM100), `mxfp8` (SM100 only). No NVFP4 or MXFP4.
  - H200 is SM90: no MXFP8, no NVFP4. FP8 checkpoints use blockwise FP8 on both sides.
- **One flat file per hardware.** `recipes/<model>/<entrypoint>/<hardware>.toml` holds the
  full config. Do not split it into a model base and a hardware overlay: hosted training takes
  one file.
- **Recipe and task never share a table.** An RL recipe sets `[model]`, `[deployment]`,
  `[slurm]`, `[weight_broadcast]`, `[trainer.model]` and `[inference]`. A task in
  `recipes/rl/tasks/` sets `[orchestrator]` and the sources. Then `cat recipe task > run.toml`
  gives a single file for `prime train run` that resolves the same as `@ recipe @ task`.
- **Document the launch command.** Each RL recipe's header holds
  `#   uv run rl @ recipes/<model>/rl/<hardware>.toml @ recipes/rl/tasks/<task>.toml`.
  `tests/unit/test_configs.py` parses every such stack.
- **Explain non-default choices.** Give each setting that differs from the default a one-line
  comment with the measurement behind it.
- **Smallest fit.** Pick the fewest trainer nodes that fit one 262144-token sample with about
  10% memory headroom. Among configs of that size, pick the highest tok/s per GPU. For
  inference, pick the replica shape with the best output tok/s per GPU that serves the
  heavy-tailed workload without preemption.

## Fitting a recipe

Fit the trainer and the inference server separately on dummy weights, then run the whole
recipe once with real weights. Loading real weights and FlashInfer autotuning can each take
longer than a whole sweep. Aim for less than 30 min of GPU time per stage. Ask before you
submit SLURM jobs; other teams have priority.

### 1. Prior

Read the model's `config.json`: total and active parameters, expert count, attention type, KV
head count, and the share of full-attention layers. Start from the nearest existing config in
`examples/advanced/` or `configs/advanced/`. Those configs are 128k: double `cp` for 256k.

### 2. Trainer fit

The trainer must fit the worst case: one 262144-token sample per micro batch. Run the trainer
alone on one node with fake data, random weights and balanced routing:

```bash
torchrun --nproc-per-node 8 --standalone src/prime_rl/trainer/rl/train.py \
    --model.name <model> --model.seq-len 262144 --model.ep 8 --model.ac.freq 1 \
    --model.debug.random-init --model.debug.force-balanced-routing \
    --data.fake.batch-size 8 --data.fake.generate-samples --data.fake.doc-len 262144 \
    --max-steps 3 --no-monitors.prime --output-dir <out>/<variant> <variant overrides>
```

- Skip step 1: it includes `torch.compile`. Read tok/s and peak memory from steps 2 and 3 in
  `<out>/<variant>/metrics.jsonl`.
- Sweep `cp ∈ {1, 2, 4, 8}` first. Then turn levers off on the best one or two: activation
  offloading (`--no-model.ac-offloading`) and optimizer offload (`--model.offload none`).
- Treat a variant as fitting only if it leaves at least 10% of GPU memory free. Real routing
  adds memory on top of the balanced-routing peak (see Known traps).
- Run the most likely winner first. Later variants reuse the inductor cache and start about
  twice as fast.

### 3. Inference fit

The inference replica must hold the 256k tail and serve many shorter agentic sessions.

- **TP.** Keep `tensor_parallel_size` at or below the KV head count. Above it, vLLM
  replicates the KV cache. Use the smallest TP that holds the weights and several 256k
  contexts. vLLM logs the KV capacity in tokens at startup.
- **DP.** Fill the node with `data_parallel_size`. For MoE models, vLLM DP shards the experts
  across DP ranks, so a `tp × dp` node is one lockstep engine.
- **Compare shapes** with dummy weights (`--vllm.load-format dummy`) under a synthetic
  agentic load. Use multi-turn sessions that share a prefix, log-normal target lengths
  (median about 24k), and a few percent of sessions that reach 256k. Raise concurrency until
  output tok/s per GPU falls, and compare shapes at their peaks. Reject any shape that
  preempts requests.
- Server startup takes 7 to 10 min. Keep the number of server variants small.

### 4. Smoke run

Run the recipe with real weights and a real task for 5 to 10 steps:

```bash
uv run rl @ recipes/<model>/rl/<hardware>.toml @ recipes/rl/tasks/swe.toml --max-steps 5 \
    --orchestrator.batch-size 64 --orchestrator.eval.skip-first-step
```

The smoke run checks the weight broadcast and the real memory peaks, which dummy weights do
not show. It also gives the real throughput:

- Trainer: tok/s and peak memory of every step after step 1.
- Inference: output and prefill tok/s per GPU, prefix-cache hit rate, peak KV cache usage,
  preemptions.
- `time/wait_for_batch` on the trainer. Set `deployment.num_infer_replicas` so that the
  trainer computes most of the time. If the trainer waits for half of each step, the
  inference side needs about twice the replicas.

### 5. Write the recipe

Write `recipes/<model>/rl/<hardware>.toml` and add a row to `recipes/README.md` with the layout,
the measured trainer tok/s and the status. Check the composition:

```bash
uv run rl @ recipes/<model>/rl/<hardware>.toml @ recipes/rl/tasks/swe.toml --dry-run
```

## Sweep lessons

- **At 256k, attention dominates trainer time.** CP and offloading change throughput by less
  than 10%, so the trainer choice is mostly about memory headroom.
- **Peak memory is not monotonic in CP.** At cp = world size, DP is 1 and every rank runs every
  micro batch, which can cost more memory than cp / 2. Sweep at least cp ∈ {2, 4, 8}.
- **Inference throughput has a peak in concurrency.** Below it, throughput equals the offered
  load and all shapes look the same. Past it, the prefix cache thrashes and throughput falls.
- **A closed-loop synthetic load measures offered load, not capacity.** Use it to rank shapes.
  Take the real per-GPU rates from the smoke run.

## Memory levers, cheapest first

Each lever costs throughput. Turn on only what the fit needs.

1. FSDP + EP shard the weights (`trainer.model.ep`).
2. CP shards activations along the sequence (`trainer.model.cp`).
3. Full activation checkpointing (`[trainer.model.ac] freq = 1`).
4. Activation offloading (`trainer.model.ac_offloading.max_inflight_activations`).
5. Optimizer offload (`trainer.model.offload = "optimizer"`, the default) or `"full"`.

## Known traps

- Balanced routing understates trainer memory. Real routing is uneven; on Qwen3.6-35B-A3B the
  peak was 8–27 GiB above the sweep. Read the peak of every smoke step, not only step 1.
- The logged MFU counts attention as non-causal. At long context it overstates real MFU by
  almost 2x.
- Nemotron-H has no CP: one 256k sample must fit on one rank.
- DeepSeek V4 supports only ring CP and needs router replay, which rules out disaggregated
  inference.
- The weight-update staging buffer needs free GPU memory on inference. DeepSeek V4 ran out of
  memory at `gpu_memory_utilization = 0.85`.
