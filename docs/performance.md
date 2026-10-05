# Performance

Measured SFT throughput of the prime-rl trainer against [torchtitan](https://github.com/pytorch/torchtitan) on the same hardware, with both stacks configured identically. The configs are in [`configs/benchmark/`](../configs/benchmark); the torchtitan counterpart is reproduced in that directory's README.

## Qwen3-235B-A22B, 4 nodes, seq 16k

Setup: 4 nodes x 8 B300, FSDP over 32 GPUs, expert parallelism 8, seq 16384 with one packed 16384-token sample per GPU per step (32 x 16384 tokens per step), AdamW with fp32 master weights and fp32 optimizer states, bf16 compute, fp32 gradient reduce, FA4 varlen attention, `torch.compile` with `fullgraph` per block, 20 steps with steady state taken over steps 5 to 20. Routing is forced round-robin balanced on both sides, since torchtitan trains from random init and its natural routing is not comparable to a trained router. prime-rl loads the HF checkpoint; torchtitan trains from random init.

All runs measured on 2026-10-02 back to back on the same four nodes. prime-rl at `mario/overlap-moe-kernel-integration` with #3753 and #3754; torchtitan at `9e159aed7` on a torch 2.15 nightly.

| run | config | s/step | tokens/s/GPU | MFU | peak memory |
|---|---|---|---|---|---|
| torchtitan, full AC | `qwen3_235b_a22b_primerl_bench_4n_s16k_fullac_bal` | 5.04 | 3253 | 40.7% | 144 GiB |
| prime-rl, torch dispatch, full AC | [`sft-4n-16k.toml`](../configs/benchmark/qwen3-235b-a22b/sft-4n-16k.toml) | 4.39 | 3732 | 46.6% | 141 GiB |
| prime-rl, Mega MoE, full AC | [`sft-4n-16k-mega-moe.toml`](../configs/benchmark/qwen3-235b-a22b/sft-4n-16k-mega-moe.toml) `--model.ac.mode full` | 4.10 | 3998 | 49.9% | 141 GiB |
| prime-rl, Mega MoE, selective AC | [`sft-4n-16k-mega-moe.toml`](../configs/benchmark/qwen3-235b-a22b/sft-4n-16k-mega-moe.toml) | 3.93 | 4168 | 52.0% | 203 GiB |

Relative to torchtitan: 1.15x with the torch dispatch, 1.23x and 1.28x with Mega MoE. Selective AC only fits at this size with Mega MoE, whose fused op saves one tensor per layer instead of the grouped-GEMM intermediates.

With the HF checkpoint's real routing instead of balanced routing, the torch-dispatch config runs at 6.11 s/step (31.2% MFU): routing imbalance costs about 40% at this size, on both stacks.

### How MFU is computed

Both stacks use the same formula, torchtitan's: `6 x active parameters + 6 x layers x heads x (qk_head_dim + v_head_dim) x seq` FLOPs per token, with the model's real head dim (128 for Qwen3-235B, not `hidden_size / num_heads` = 64), over the full step wall time and a 2.25 PFLOP/s bf16 peak per B300. For Qwen3-235B-A22B at seq 16384 that is 280.8 GFLOP/token.

Treat the absolute figures with care: the formula charges dense causal attention over the 16k window, which is 53% of the counted FLOPs, while packed short documents make attention block-diagonal and nearly free (0.2 to 0.3 s of each step in both stacks). MFU on executed FLOPs is roughly half of the numbers above. The comparison between stacks is unaffected.

### Where the time goes

From rank-0 profiler traces of one steady-state step (torch dispatch, balanced routing):

| | prime-rl | torchtitan |
|---|---|---|
| step | 4.46 s | 4.86 s |
| GEMM | 2.09 s | 1.91 s |
| compiled MoE dispatch/combine glue | 0.62 s | 0.71 s |
| attention | 0.30 s | 0.21 s |
| EP all-to-all | 0.99 s, on the compute stream | 1.34 s, on its own stream |
| compute-stream idle | ~0.15 s | ~1.55 s, waiting for the all-to-all |
| FSDP all-gather / reduce-scatter | 1.23 / 1.34 s, overlapped | 1.35 / 1.34 s, overlapped |

Both stacks have the expert-parallel all-to-all fully exposed; prime-rl spends less time in it. GEMM and attention are slightly slower in prime-rl. Mega MoE removes the exposed all-to-all, which is where its gain comes from.

## Qwen3-30B-A3B, 1 node, seq 16k

[`configs/benchmark/qwen3-30b-a3b/sft-1n-16k-mega-moe.toml`](../configs/benchmark/qwen3-30b-a3b/sft-1n-16k-mega-moe.toml) is the same setup scaled to one 8-GPU node. It has not been measured yet; this section will get a table once it has.

## Reproducing

```bash
# torch dispatch
uv run sft @ configs/benchmark/qwen3-235b-a22b/sft-4n-16k.toml
# Mega MoE, selective AC (needs `uv sync --extra mega-moe`)
uv run sft @ configs/benchmark/qwen3-235b-a22b/sft-4n-16k-mega-moe.toml
# Mega MoE, full AC
uv run sft @ configs/benchmark/qwen3-235b-a22b/sft-4n-16k-mega-moe.toml --model.ac.mode full
```

Per-step metrics land in `outputs/<run>/monitors/file/metrics.jsonl` (`time/step`, `perf/mfu`, `perf/throughput_per_gpu`, `perf/peak_memory`). Average `time/step` over steps 5 to 20 for the numbers above.
