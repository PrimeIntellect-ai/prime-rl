# EPLB weight reload check

Two-node Qwen3.5-35B-A3B RL on Hendrycks-Math: eight trainer GPUs and eight
inference GPUs. The debug algorithm retains every group, and zero learning rate
isolates reload errors from policy updates. Inference uses BF16 checkpoint weights
with online FP8 per-block quantization, TP8/EP8, and the Triton MoE kernel.

The trainer places the experts at every weight update, with eight redundant experts. Policy weights travel from
the trainer through NCCL:

```bash
uv run rl @ configs/debug/eplb/rl.json --run.name eplb-math
```

The null harness runs in Prime sandboxes to enforce the math task's network policy;
authenticate with `prime login` or `PRIME_API_KEY`. The JSON config disables the
optional reference judge with `null`, so rewards use local math verification. It also disables
image/video inputs and leaves the small GDN projections and shared experts in
BF16 because their TP8 dimensions are smaller than FP8 block sizes.

For an EPLB-disabled control on a separate pair of nodes:

```bash
uv run rl @ configs/debug/eplb/rl.json --run.name eplb-math-control \
  --no-inference.vllm.enable-eplb
```

Compare trainer `mismatch_kl/all/mean` across repeated broadcasts. Reload logs
report how many physical expert slots differ from the initial placement. A run
with unchanged placement does not exercise the reload remapping.

To exercise optimizer updates and CUDA graphs as well, add
`--trainer.optim.lr 1e-6 --no-inference.vllm.enforce-eager` to either command.
