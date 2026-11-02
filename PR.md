This PR adds `model.dsv4_cudnn_sink_grad`, which takes DeepSeek V4's sparse-attention sink gradient from what the `cudnn_flashmla` backward already computes instead of rerunning the TileLang preprocess pass, saving about 1.55 ms per layer per 32768-token micro batch on H200 with no new nondeterminism.

#4078 folded the sink into the LSE to keep TileLang's op contract, not because cuDNN lacked sinks, so with the flag the forward keeps FlashMLA's sink-free LSE and cuDNN gets the real sinks. On SM90 the change folds the sink into cuDNN's LSE, since cuDNN's own `d_sink` is slower there and issues nondeterministic fp32 atomics, and on SM100 it uses cuDNN's native `d_sink`, exercised by the B300 campaign.

The numbers come from the DSv4 stack A/B campaign on the 6-layer perf proxy (a DeepSeek V4 Flash checkpoint, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), run on H200 with FP8 blockwise and on B300 with MXFP8, each paired against its own main arm on the same node. The step-time metric is `time/forward_backward`, the table shows medians over steps 4 to 20, and a negative delta is faster. B300 repeat floors were 0.04 to 0.12 percent.

| run | median time/forward_backward s, H200 | median time/forward_backward s, B300 | delta vs main, H200 | delta vs main, B300 | tok/s per GPU (B300) | peak GiB (B300) |
|---|---:|---:|---:|---:|---:|---:|
| main | 14.631 | 8.227 | - | - | 29937 | 109.17 |
| l1 (this change) | 14.540 | 8.212 | -0.62% | -0.18% | 30055 | 108.55 |

Numerics are verified unchanged up to the expected level: the forward is bitwise unchanged and dQ, dKV and dsink match fp64 references at the same error as the old path, with dsink bitwise and deterministic.
