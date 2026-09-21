# GLM-5.3 NVFP4 Scale-SWE RL on B300

From the repository root:

```bash
uv sync --all-extras --all-packages
uv run rl @ examples/advanced/glm-5.3/infer/pd-nvfp4-b300.toml --dry-run
uv run rl @ examples/advanced/glm-5.3/infer/pd-nvfp4-b300.toml
```

The dry run writes resolved trainer, orchestrator, prefill/decode and environment
configs plus the Slurm script without allocating nodes. The launch requests six
exclusive eight-GPU nodes: four trainer nodes, one prefill node and one decode
node, with no job time limit. The router listens on port 8000 on the first
inference node and accepts the alias `glm53` as well as the shared model path.

Training stops after 1,000 optimizer steps. `clean=true` deletes this named
run's directory before each fresh launch, including its previous logs and
checkpoints; explicit `--resume` instead follows resume semantics. The run
directory is `outputs/glm53-nvfp4-rl/scaleswe-nvfp4-4over6`.

Trainer and orchestrator log to the shared W&B project `glm53-nvfp4-rl`, with
the run name inherited from `[run]`. The Slurm launcher sources the repository's
git-ignored `.env`; `pre_run_command` exports `WANDB_API_KEY` to its child
processes. Keep the key in `.env`, outside the TOML and generated configs.

| Trainer setting | Value |
|---|---|
| Nodes / GPUs | 4 / 32 |
| Expert parallelism | EP8, default torch dispatch with BF16 transport |
| Context parallelism | CP4, ring |
| Sequence length | 65,536 |
| Optimizer | AdamW; default LR 1e-6, weight decay 0.01 |
| CPU offload | Optimizer states; full CPU offload disabled |
| Activation checkpointing | Selective, every layer, default retained operations |
| Activation offloading | Enabled, one activation in flight |
| Expert compute | NVFP4 4over6 for the first 85% of layers; BF16 elsewhere |
| Index cache | Recompute every 4 layers, matching inference |
| Weight transfer | NCCL, BF16 weights quantized online by inference |

Both sides start from the existing local `zai-org/GLM-5.3-BF16` download at
revision `9d2398f478cab2de883137db3a36ad2c96205e24`. Trainer loading uses its
converted Prime checkpoint. `apply_to="85%"` selects the first 66 of 78 layers
(rounding down): routed experts in layers 3–65 use NVFP4, while the final 12
layers remain BF16. Dense and shared-expert modules also remain BF16. Inference
uses the same layer selection with `nvfp4_per_token` online quantization and
4over6's 448 normalization / MAE scoring. NVFP4 trainer backward uses
dequantized forward operands with BF16 GEMMs. Optimization and reduction dtypes
retain their model defaults.

| Inference setting | Prefill | Decode |
|---|---|---|
| Parallelism | DPEP8, TP1 | DPEP8, TP1 |
| All-to-all backend | FlashInfer NVLink one-sided | DeepEP low latency |
| Batched tokens | 32,768 | 320 |
| Maximum sequences per GPU | 256 | 320 |
| GPU memory utilization | 0.75 | 0.90 |
| Graphs | None | Full + piecewise, breakable |
| Mooncake store RAM per node | 2,560 GiB | 0 |
| HiSparse host RAM per GPU | Disabled | 300 GiB |

Both roles use FP8 KV, a 65,536-token context limit, index cache with frequency
4, natural expert routing, and no EPLB, DBO or MTP. Routing is sticky least
loaded with retries disabled. Each rank binds to its configured NUMA node and
UCX NIC. These mappings and host-memory capacities target this B300 cluster;
adjust them for a different machine topology.

Use the pinned HiSparse/NVFP4 wheel with the standard Python frontend and native
`NixlConnector`. Mooncake master/storage processes belong to the Slurm job, so
cancellation also releases their memory. Both roles connect to the same store
pool with separate role cache-key prefixes and 512 MiB staging buffers per
connector. The prefixes identify the BF16-source online-quantized configuration.

The training source uses Scale-SWE quarter-zero tasks selected by the instance-ID
filter, bash harness, automatic compaction, 720 sandbox creates per minute, a
six-hour rollout timeout, and no agentic judge. It retains 16 rollouts per task
and `clear_thinking=false`. The static environment pool has 20 workers with 128
concurrent slots each. Training sandboxes use `glm53-pd-train` and
`int4-syn-gen-bash` labels.

Rollout concurrency is fixed at 2,048 by setting initial, minimum and maximum
in-flight episodes to the same value. Episodes doing tool work also occupy slots,
so this does not guarantee 2,048 simultaneous inference requests.
`orchestrator.batch_size=512` counts trainer-bound traces per optimizer step,
independently of concurrency. The default `constant_trainer_batch_size=true`
prunes zero-advantage RL samples before counting toward the batch and keeps
collecting until 512 useful traces remain. Plain GRPO groups with identical
rewards contribute no useful traces.

Sampling uses temperature 1, top-p 0.95 and top-k 512, matching the prior sampling
replay run. Top-p comes from the BF16 checkpoint's generation config; 512 is the
bounded top-k used for replay (the checkpoint does not specify a top-k).
Sampling-mask capture is enabled on inference, and the trainer replays those
masks, including with CP. Inference returns processed logprobs so rollout and
trainer probabilities use the same truncated distribution. Router replay is
disabled. This config does not launch an additional eval or profiler.
