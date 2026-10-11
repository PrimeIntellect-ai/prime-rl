# Weight-broadcast benchmark: ModelExpress vs NCCL

Run on 2026-10-10/11, branch `feat/modelexpress-test` (PR #3890 rebased on main), ModelExpress client and
server at ai-dynamo/modelexpress `8512b8c`, vLLM 0.31.0, H200 nodes with InfiniBand.

Setup: `qwen3-30b-a3b.toml`: Qwen3-30B-A3B-Instruct-2507, 1 trainer node (FSDP over 8 GPUs) + 1 inference
node (TP=2, 4 DP servers = 8 receiving workers), reverse-text env, 10 steps, `max_off_policy_steps = 1`.

```bash
uv run rl @ configs/debug/weight-broadcast/qwen3-30b-a3b.toml @ configs/debug/weight-broadcast/<overlay>.toml
```

Medians over the 10 steps (trainer: steps 3-10). *Trainer* is `time/broadcast_weights`, the time the
trainer blocks on a weight update until inference has installed it. *Inference* is
`watcher/last_update_weights_time`, the time the orchestrator keeps inference paused for the update.

| Inference dtype | Transport (overlays) | Trainer (s) | Inference (s) | Mismatch KL |
|---|---|---:|---:|---|
| BF16 | NCCL (`nccl`) | 5.86 | 4.34 | 0.019-0.034 |
| BF16 | ModelExpress, full staging (`modelexpress`) | **4.57** | **2.32** | 0.016-0.036 |
| BF16 | ModelExpress, bounded 2 x 1 GiB staging (`modelexpress-bounded`) | 92.87 | 89.28 | - |
| FP8 per-block | NCCL (`nccl` + `fp8`) | 8.74 | 6.99 | 0.070-0.160 |
| FP8 per-block | ModelExpress, full staging (`modelexpress` + `fp8`) | **5.65** | **3.84** | 0.070-0.144 |

MX's per-update timing on one inference GPU (BF16, full staging): 40.6 GB received, wire transfer 1.49 s
(~27 GB/s), receive sync 0.11 s, install 0.05 s, 1.8 s end to end.

## Findings

- With full staging, ModelExpress is 1.3-1.5x faster than NCCL on the trainer side and 1.8-1.9x faster on
  the inference pause, for both BF16 and online-FP8 inference.
- Mismatch KL matches NCCL in both modes, so the installed weights are correct, including vLLM's online
  FP8 quantization after each update.
- Full staging needs a second copy of each inference GPU's weight shard. With vLLM's default
  `gpu_memory_utilization` of 0.9 the first update OOMs on Qwen3-30B; the benchmark uses 0.6.
- Bounded staging (`staging_buffer_bytes`) is ~20x slower than full staging. MX attributes ~91.6 s of the
  91.7 s update to no instrumented stage, so the time is spent in the streaming read/install pipeline.
- MX rejects bounded staging for quantized engines ("bounded streaming currently requires an unquantized
  engine"), so FP8 inference only works with full staging. Large FP8 models (GLM-5 scale) need either
  room for a full BF16 shard copy per GPU or a fixed, FP8-capable bounded mode.
- A single-node run of `tests/integration/test_reverse_text_modelexpress.py` (Qwen3-0.6B, 1+1 GPU) passes:
  reward 0.14 -> 0.71, mismatch KL 0.0018, ~0.5 s per update.

## Not covered

Multi-node trainer (GLM-4.5-Air, 2 trainer nodes), expert-parallel inference, P/D disaggregation,
resume, and the existing `nixl` transport against the bumped MX client/server.
