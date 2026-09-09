# Recorded numerical evidence

These JSON files preserve the completed reproduction's compact audit outputs.
They do not contain model weights or the full rollout traces.

- `completion-audit.json`: all twelve completed runs, token counts, update
  evidence, scheduler states, and hashes of their final `zero-audit.json` files.
- `kl-metrics.json`: observed minimum and maximum of each run's logged KL,
  stable K3, raw bit-mismatch, and absolute-error metrics. Every recorded value
  is exactly zero in the twelve aligned runs.
- `learning-sampling-audit.json`: actual sampling settings and one-call coverage
  for all 3,360 consumed training traces across the six learning arms.
- Per-run `zero-audit.json`: comparisons of raw FP32 selected-token logprobs,
  generation/shipment/trainer policy joins, step coverage, and router probes.
- Per-run `prefill-replay.json`: live frozen trainer/prefill/decode comparisons.
- `evaluation-policy-spans.json`: all five GLM GSM8K evaluations, with complete
  policy spans and finish reasons.
- Operator probes: dense forward/backward checks, FP32 heads, MoE routing,
  GLM partial RoPE and updated UVA weights, attention layouts, pinned allocations,
  and three real CPU-offloaded Adam updates with reclaimed gradient pages.

The nonzero dense baseline and failed MoE candidates are retained separately.
The unchanged-policy math candidate correctly fails the learning gate even
though its logprobs agree.

The audits were computed against full-precision traces in the original
`outputs/mismatch/<run>` directories. To independently recompute raw-bit and
policy joins, retain or regenerate those traces and use the tools documented
in the parent README. These summary files alone cannot reconstruct the traces.

The scope is the sampled tokens and tested configurations. Learning arms run
20 steps at LR 1e-6; frozen checks run three steps at LR zero. The frozen long
checks are heavily truncated numerical stress tests. No convergence or held-out
accuracy improvement is claimed.
