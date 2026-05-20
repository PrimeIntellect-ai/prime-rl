# Highlights

## 1. Fused LM head / chunking (logits + loss)

We introduced a fused lm head with selective logprobs, significantly decreasing the peak vram required for the RL loss function. This is now enabled by default and should greatly reduce the vram requirements for doing rl training.

Example on `Qwen/Qwen3-0.6B` at `16384` sequence length where we reduced the peak vram from `44.2GiB` -> `3.3 GiB`:
**With previous implementation**
```
                                                            Benchmark                                                            
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃    Step ┃             MFU              ┃           Throughput            ┃            Step Time            ┃   Peak Memory    ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│       1 │            7.85%             │             12.12K              │             21.63s              │     44.2 GiB     │
│       2 │            7.89%             │             12.19K              │             21.39s              │     44.2 GiB     │
│       3 │            7.83%             │             12.10K              │             21.99s              │     44.2 GiB     │
│         │                              │                                 │                                 │                  │
│ Overall │ 7.86% ± 0.03% [7.83%, 7.89%] │ 12.13K ± 46.45 [12.10K, 12.19K] │ 21.67s ± 0.30s [21.39s, 21.99s] │ 44.2 GiB (93.1%) │
└─────────┴──────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┴──────────────────┘
```

**With fused chunked lm head**
```
                                                           Benchmark                                                           
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃    Step ┃             MFU              ┃           Throughput            ┃            Step Time            ┃    Peak Memory   ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│       1 │            8.07%             │             12.47K              │             21.02s              │      3.3 GiB     │
│       2 │            8.10%             │             12.51K              │             20.90s              │      3.3 GiB     │
│       3 │            8.14%             │             12.56K              │             20.67s              │      3.3 GiB     │
│         │                              │                                 │                                 │                  │
│ Overall │ 8.10% ± 0.03% [8.07%, 8.14%] │ 12.51K ± 48.19 [12.47K, 12.56K] │ 20.86s ± 0.18s [20.67s, 21.02s] │   3.3 GiB (6.9%) │
└─────────┴──────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┴──────────────────┘

```

[#1525](https://github.com/PrimeIntellect-ai/prime-rl/pull/1525) — Fused LM Head implementation.
[#1544](https://github.com/PrimeIntellect-ai/prime-rl/pull/1544) — Default fused_lm_head_chunk_size=2048 for RL.
[#1545](https://github.com/PrimeIntellect-ai/prime-rl/pull/1545) — Enable loss chunking for non-custom HF models.


## 2. On Policy distillation 

Added  [on policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) : the student learns from a teacher on the student’s own rollouts, so it stays on-policy while still getting dense, step-by-step guidance. Compared to normal (off-policy) distillation, this reduces the “teacher-only states” mismatch and helps the model learn to recover from its own mistakes instead of only imitating perfect trajectories

Quickstart docs at https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/on_policy_distillation.md

[#1458](https://github.com/PrimeIntellect-ai/prime-rl/pull/1458) - Add support for On Policy distillation

## 3. Advanced multi LoRA support

LoRA are now first class citizen in prime-rl. This release, we add some preliminary features to support training multiple separate lora from different runs with the same trainer and inference deployment. We also now support training loras for MoE experts.

[#1571](https://github.com/PrimeIntellect-ai/prime-rl/pull/1571) — Update LoRA default alpha to 32.
[#1567](https://github.com/PrimeIntellect-ai/prime-rl/pull/1567) — Change LoRA alpha default to 32.
[#1526](https://github.com/PrimeIntellect-ai/prime-rl/pull/1526) — MoE LoRA support.
[#1520](https://github.com/PrimeIntellect-ai/prime-rl/pull/1520) — Retry load_lora_adapter (NFS delays).

## 4. New model support
We now natively support AFMoE!

[#1515](https://github.com/PrimeIntellect-ai/prime-rl/pull/1515) — AFMoE support

## 5. Trainer observability / metrics

Prime-rl RL trainer can now optionally expose metrics through a prometheus metrics server

[#1547](https://github.com/PrimeIntellect-ai/prime-rl/pull/1547) — Prometheus metrics server for trainer.

## 5. Refactor logging for environment 

We can now redirect the log of a given environment to a specific logging file, we also intercept verifier logger into prime-rl format

[#1594](https://github.com/PrimeIntellect-ai/prime-rl/pull/1594)
[#1561](https://github.com/PrimeIntellect-ai/prime-rl/pull/1561) 

--- 

# Breaking changes 

* Config rename: ckpt.keep → ckpt.keep_last (and new ckpt.keep_interval). Update configs that still set ckpt.keep. (2025-12-31)

* Behavior change / defaults: MultiLoRAMoE / QwenMoE now enables training expert LoRAs by default via target_modules changes. 

* Behavior change (RL defaults): RL training auto-sets model.fused_lm_head_chunk_size=2048 when unset (except impl='liger_kernel'). This can change memory/throughput characteristics vs v0.2. (2026-01-05)

* Default change: model.lora.alpha default changed 16.0 → 32.0 (impacts effective LoRA scaling if you relied on the old default). (2026-01-10)

---

## Bug fixes

[#1568](https://github.com/PrimeIntellect-ai/prime-rl/pull/1568) — Unique rollout request IDs to avoid collisions.
[#1546](https://github.com/PrimeIntellect-ai/prime-rl/pull/1546) — Detect dead worker process in collect_responses.
[#1563](https://github.com/PrimeIntellect-ai/prime-rl/pull/1563) — Fix orchestrator null-batch handling.
[#1537](https://github.com/PrimeIntellect-ai/prime-rl/pull/1537) — Fix checkpoint cleanup on resume + cancelled rollout metric.
[#1531](https://github.com/PrimeIntellect-ai/prime-rl/pull/1531) — Fix NCCL handshake.
[#1529](https://github.com/PrimeIntellect-ai/prime-rl/pull/1529) — Fix W&B integration.
[#1520](https://github.com/PrimeIntellect-ai/prime-rl/pull/1520) — Retry load_lora_adapter for NFS delays.

## misc

[#1551](https://github.com/PrimeIntellect-ai/prime-rl/pull/1551) — TrainingSample reward: adds reward to TrainingSample for logging/consumption in training pipelines.
[#1521](https://github.com/PrimeIntellect-ai/prime-rl/pull/1521) — Checkpoint retention policy: adds keep_interval to keep periodic checkpoints in addition to “last N”.
[#1536](https://github.com/PrimeIntellect-ai/prime-rl/pull/1536) — Blackwell kernels: enables grouped_mm on Blackwell GPUs.
[#1557](https://github.com/PrimeIntellect-ai/prime-rl/pull/1557) — Cumsum dtype: switches multilinear cumsum dtype to int32 (avoids wider dtype overhead).
[#1571](https://github.com/PrimeIntellect-ai/prime-rl/pull/1571) — Update LoRA layer default alpha from 16.0 to 32.0.
[#1567](https://github.com/PrimeIntellect-ai/prime-rl/pull/1567) — Change LoRA alpha default to 32.
[#1538](https://github.com/PrimeIntellect-ai/prime-rl/pull/1538) — Add step param to Monitor.log() interface.
[#1518](https://github.com/PrimeIntellect-ai/prime-rl/pull/1518) — Refactor online eval.
[#1533](https://github.com/PrimeIntellect-ai/prime-rl/pull/1533) — README updates.
[#1550](https://github.com/PrimeIntellect-ai/prime-rl/pull/1550) — Remove PR template.
[#1522](https://github.com/PrimeIntellect-ai/prime-rl/pull/1522) — Docs/changelog entry for ckpt.keep_last + ckpt.keep_interval.
[#1506](https://github.com/PrimeIntellect-ai/prime-rl/pull/1506) — Inference readiness handshake (later reverted).
[#1528](https://github.com/PrimeIntellect-ai/prime-rl/pull/1528) — Revert inference readiness handshake from #1506.
[#1516](https://github.com/PrimeIntellect-ai/prime-rl/pull/1516) — Remove step usage in W&B monitor (later reverted).
[#1530](https://github.com/PrimeIntellect-ai/prime-rl/pull/1530) — Revert W&B monitor “step” removal from #1516.
[#1580](https://github.com/PrimeIntellect-ai/prime-rl/pull/1580) —  add fa3 dependency


