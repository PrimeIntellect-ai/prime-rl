# dist-moe-sft-ab

Traced SFT runs (4 steps, rank 0 analyzed unless noted) on node primeintellect-tur-r4n4 (job 2761), prime-rl worktree prime-rl-feat-dist-moe at acf7e6354. Configs in benchmarks/scripts/dist_moe/ (base + dispatch overlay), launched via ~/tmp/dist-moe-sft/launch.sh (commands in ~/tmp/dist-moe-sft/commands.sh). Timing claims come from the untraced runs in ~/tmp/dist-moe-sft/ (summaries/*.csv).

- traces/tr-qwen-torch, traces/tr-qwen-distmoe: Qwen3-30B-A3B, fake data, balanced routing, ep 8, 32k tokens/GPU, 2 micro batches/step, AC full, compile default.
- traces/tr-glm-torch, traces/tr-glm-distmoe: GLM-4.5-Air 8 layers, INTELLECT-3-SFT-10K math, ep 8, 16k tokens/GPU (if present).
- derived/qwen-{torch,distmoe}-top.{txt,csv}: top_kernels over the last step (2 forward annotations).
- derived/qwen-accounting.{txt,csv}: step_accounting, all 8 ranks, last 2 traced steps, with untraced medians.
- derived/qwen-host-ops.txt: host_ops, rank 0.
- derived/qwen-kernel-diff.txt: kernel_diff of the top_kernels CSVs.
- figures/qwen-layer-forward.png: layer 10 forward timeline, both arms, main stream, plus per-category delta.
- figures/qwen-accounting.png: step accounting delta bars.
- No deepep traces: the deepep arm fails at step 1 on this torch 2.14 build (see ~/tmp/dist-moe-sft/NOTES.md).
- derived/glm-accounting.{txt,csv}, derived/glm-host-ops.txt: same analyses for GLM.
- figures/glm-layer5-forward.png: GLM layer 5 forward timeline (representative). figures/glm-layer-forward.png (layer 3) is dominated by one 90 ms all-to-all wait in the torch arm on rank 0 (a straggler), not representative.
