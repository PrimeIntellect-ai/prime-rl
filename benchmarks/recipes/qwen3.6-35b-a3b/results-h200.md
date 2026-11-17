# Qwen3.6-35B-A3B on H200: fit results (2026-11-16, commit 167d4c3bf)

Dummy weights, one 8x H200 node per sweep.

## Trainer (trainer_fit.sh, ep 8, one 262144-token sample per micro batch, 8 micro batches per step)

| variant | status | steps | step s | tok/s | MFU % | peak GiB | headroom GiB | wall s |
|---|---|---|---|---|---|---|---|---|
| cp1 | OOM | 0 | - | - | - | - | - | 197 |
| cp2 | ok | 2 | 84.8 | 24,737 | 45.9 | 112.7 | 27.1 | 507 |
| cp2-no-acoff | ok | 2 | 81.2 | 25,895 | 48.0 | 128.4 | 11.4 | 373 |
| cp4 | ok | 2 | 86.8 | 24,182 | 44.9 | 79.4 | 60.4 | 455 |
| cp4-no-acoff | ok | 2 | 82.2 | 25,573 | 47.5 | 87.4 | 52.4 | 377 |
| cp4-optim-gpu | ok | 2 | 83.3 | 25,189 | 46.7 | 115.6 | 24.2 | 378 |
| cp8 | ok | 2 | 88.3 | 23,764 | 44.1 | 93.2 | 46.6 | 821 |

## Inference (inference_fit.sh, bench_inference.py defaults)

| variant | conc | startup s | out tok/s/GPU | engine prefill tok/s | prefix hit | KV mean | KV max | waiting | preempt | TTFT p50/p99 s | ITL p50/p99 ms | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tp1-dp8 | 256 | 581 | 542 | 151,570 | 0.88 | 0.17 | 0.18 | 0 | 0 | 0.4/26.7 | 23/216 | 0 |
| tp1-dp8 | 1024 | 581 | 330 | 123,765 | 0.42 | 0.59 | 0.73 | 124 | 0 | 36.4/218.3 | 171/885 | 0 |
| tp2-dp4 | 256 | 551 | 535 | 156,306 | 0.87 | 0.17 | 0.18 | 0 | 0 | 0.4/16.2 | 22/182 | 0 |
| tp2-dp4 | 1024 | 551 | 347 | 219,243 | 0.59 | 0.68 | 0.72 | 78 | 0 | 17.5/133.8 | 212/775 | 0 |
| tp4-dp2 | 256 | 400 | 458 | 134,948 | 0.88 | 0.31 | 0.33 | 0 | 0 | 0.5/39.4 | 26/193 | 0 |
| tp4-dp2 | 1024 | 400 | 296 | 98,285 | 0.00 | 0.58 | 0.63 | 606 | 0 | 165.9/217.7 | 149/319 | 0 |

tp1-dp8 at 1024 ran before the warmup waited for the queue to drain; it is not comparable.

## Smoke run (real weights, 2026-11-17, commit 505b59b28)

`uv run rl @ recipes/qwen3.6-35b-a3b/h200.toml @ recipes/tasks/swe.toml --max-steps 5
--orchestrator.batch-size 64 --orchestrator.eval.skip-first-step`, one trainer and one
inference node. All 5 steps passed. Reward 0.45 → 0.59, mismatch KL 0.0011–0.0013, no
truncation, no timeouts, no preemptions.

| step | tokens | fwd+bwd s | trainer tok/s | wait for batch s | step s | peak GiB |
|---|---|---|---|---|---|---|
| 1 | 1.43M | 322 | 4,430 | 1,075 | 1,412 | 114.0 |
| 2 | 1.99M | 99 | 20,081 | 339 | 446 | 95.7 |
| 3 | 1.90M | 76 | 24,941 | 157 | 242 | 95.5 |
| 4 | 1.37M | 59 | 23,207 | 84 | 151 | 95.5 |
| 5 | 1.75M | 89 | 19,606 | 255 | 353 | 95.2 |

- Step 1 includes `torch.compile`. Steps 2–5: 21.7k tok/s (2.7k per GPU), 27.4k tokens per
  sample on average.
- Real routing is uneven (EP imbalance ~1.3, max violation ~4.3). Peak memory is 8–27 GiB
  above the balanced-routing sweep.
- The trainer computes 27% of the time and waits for batches the rest. The recipe therefore
  sets 3 inference replicas; that setting was not run.
- Inference over the last 29 min: 914 output tok/s per GPU, 21.4k uncached prefill tok/s per
  GPU, 74% prefix-cache hit rate, up to 1,019 running requests, KV cache at most 73% full.
- vLLM startup with real weights: 442 s.
