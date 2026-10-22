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
- traces/tr-fg-{qwen,glm}-{torch,distmoe}: same as the tr-* runs but with compile.fullgraph = true (D/compile-fullgraph.toml), node primeintellect-tur-r1n6 (job 2802), same code.
- derived/fg-qwen-{torch,distmoe}-top.{txt,csv}, fg-qwen-kernel-diff.txt, fg-{qwen,glm}-accounting.{txt,csv}, fg-{qwen,glm}-host-ops.txt: same analyses on the fullgraph traces, untraced medians from ~/tmp/dist-moe-sft/summaries/fg-*-ab.csv (Qwen torch 5.960, dist_moe 5.057; GLM 1.458, 1.124).
- Fullgraph Qwen: torch traced window 6001 ms vs 8756 ms with fullgraph off. The gain is all torch-arm compute: elementwise/copy 2156 -> 415 ms and token permute/gather/scatter 1795 -> 537 ms; exposed all-to-all is unchanged at about 1224 ms. dist_moe window 5068 vs 5108 ms. Remaining dist_moe - torch delta of -933 ms: exposed all-to-all -1225, permute -491, elementwise -189, grouped GEMM -590 offset by dist_moe fused kernels +1033, wgrad +160, triton glue +143, symm-mem barrier +130, attention +143 (flash bwd slower in the dist_moe arm).
- Fullgraph GLM: torch window 1498 ms (1.89 s untraced before), dist_moe 1142 ms; delta -356 ms is mostly exposed all-to-all -398 and permute -73, offset by symm-mem barrier +78 and dist_moe kernels.
- No fullgraph timeline figures (timeline_figure.py was being changed concurrently).
