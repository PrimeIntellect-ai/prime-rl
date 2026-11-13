# NCCL 2.32.3 pin: A/B harness and results

A/B of `main` at `9b4e0f50a` (torch 2.14.0+cu130 with its default NCCL 2.30.7) against `8d4adad4c` (the same code with `nvidia-nccl-cu13==2.32.3`), on 2 nodes of 8x H100 80GB SXM with 8x 400 Gb/s InfiniBand NICs per node, two GPUs and two NICs under each PCIe switch, vLLM 0.33.0. Each arm ran from its own detached checkout with its own venv; the loaded NCCL version was read at runtime (`ncclGetVersion` and the NCCL and vLLM version banners), and both arms of every cell ran serially on the same nodes.

## Results

Concurrent cross-node all-gather, bf16, 1 GiB output, bus bandwidth GB/s per GPU, 8 one-GPU-per-node groups (one per local GPU index) at once vs local GPU 0's group alone. The 16-rank collectives over both nodes (all_gather, reduce_scatter, all_to_all, all_reduce) were within 1% between versions.

| NCCL | 8 groups at once | one group alone |
|---|---|---|
| 2.30.7 | 20.0 | 33.7 |
| 2.32.3 | 33.5 | 33.7 |

Trainer (`trainer/`): Qwen3-30B-A3B, fake data with balanced routing, FSDP over 16 GPUs with `ep = 8`, full activation checkpointing, no offloading, 20 steps, median step time over steps 6 to 20, two runs per arm in the order main, fix, main, fix. Loss and grad norm agree within run-to-run noise; every rank of every run exited 5 to 9 s after the last step.

| tokens per micro-batch, micro-batches per step | main step s | fix step s | fix tok/s/GPU vs main |
|---|---|---|---|
| 16384, 2 | 4.759, 4.759 | 4.678, 4.681 | 7004, 7000 vs 6885, 6885 |
| 4096, 1 | 0.812, 0.811 | 0.666, 0.672 | 6150, 6095 vs 5043, 5053 |

RL (`rl/`): reverse-text, trainer on one node (FSDP8, `ep = 8`), four vLLM servers (TP=2) on the other, NCCL weight broadcast, 20 steps, two runs per arm. Fix and main are within run-to-run noise; every component exited cleanly.

| | main | fix |
|---|---|---|
| step s, median per run | 7.31, 7.02 | 7.71, 7.07 |
| weight broadcast s, median over both runs | 5.13 | 5.11 |
| trainer rank exit after last step, s | 5.3 to 6.8 | 5.0 to 7.0 |

A mixed run (trainer on NCCL 2.32.3, vLLM on 2.30.7, same code) fails in the weight-broadcast `ncclCommInitRank` on the inference side with `Message truncated : received 168,144 bytes instead of 144,144`: every process in one communicator must load the same NCCL version.

vLLM serving (`vllm/`): Qwen3-30B-A3B-Instruct-2507 on one node via the `inference` entrypoint, random 1024-token prompts with 512 output tokens, 128 concurrent requests, 1024 measured requests after warmup, one run per arm. Every run completed all requests and shut down cleanly on Ctrl-C. vLLM force-kills its workers at shutdown, so this does not exercise NCCL communicator finalize.

| layout | main output tok/s | fix output tok/s | main TPOT p50 ms | fix TPOT p50 ms |
|---|---|---|---|---|
| TP=8 | 12496 | 12545 | 9.36 | 9.47 |
| DP=8 with expert parallel | 11749 | 11615 | 9.80 | 9.98 |
| TP=8 with expert parallel | 12265 | 12365 | 9.44 | 9.60 |

`destroy_process_group` hang (pytorch/pytorch#201536): see `teardown_hang_repro.py` and `teardown_hang_repro_results.txt`, added in the commit before this one.

## Running

Each entry script takes its checkouts, output directory and, where it launches across nodes, its SLURM job and nodes from the arguments and environment variables in its usage line. The trainer and RL scripts start their steps in an existing allocation with `srun --overlap`; run `vllm/run_cell.sh` itself on a node of one.

- `trainer/run_one.sh <main|fix> <run name> <master port> [config]`, then `trainer/compare.py <runs dir> <run names...>`.
- `rl/run_arm.sh <trainer checkout> <inference checkout> <run name>`, then `rl/summarize.py`.
- `vllm/run_cell.sh` with a layout config (`tp8.toml`, `dp8ep.toml`, `tp8ep.toml`), then `vllm/summarize.sh`.
