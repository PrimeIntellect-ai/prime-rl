# GLM-4.5-Air blockwise FP8 inference investigation

## Full trainer-compute follow-up, September 10 05:35 UTC

The later four-node attempts did not establish a real optimizer step:

- Job 1241 stopped at a missing `.prime-v1` conversion-cache marker in the
  user-local HF cache. The prepared config now explicitly uses `/home/hf-cache`.
- Job 1244 reached the initial FP8 broadcast after converting/loading weights.
- Job 1247 completed startup v0 synchronization at 02:25:11 UTC, but all 40
  finished reverse-text rollouts had zero reward. GRPO assigned zero advantages;
  constant-size batching pruned every sample and the zero-output guard fired
  at 02:25:56. The one-hour failure-cleanup grace period kept the allocation
  alive until 03:26. This is not evidence of a GPU deadlock or nonfinite FP8.

Those trainers had `quantization=None` and MoE `compute=bf16`: FP8 export is not
FP8 training. The prepared `configs/debug/fp8-air-4n-micro.toml` now enables both
trainer linear FP8 and routed-expert DeepGEMM FP8, plus vLLM blockwise FP8 and
trainer-side FP8 NCCL transfer. It uses two optimizer steps, batch four, two
trainer nodes and two one-node inference replicas (TP1/DP8/EP), a one-hour
walltime, immediate failure cleanup, no W&B or sandboxes, and the upstream
`debug` algorithm with advantage one. The three files from upstream commit
`aa3a6522b` (#3521) were applied byte-exactly without creating a commit.

Enabling actual trainer FP8 revealed another source-level correctness bug:
`Float8BlockwiseLinear` kept the Q/K/V biases but never added them in forward.
The isolated fix adds bias, pads both logical dimensions to 128 for GEMM,
preserves gradient shapes through padding/slicing, and honors constructor dtype.
It no longer silently skips GLM Air's 10,944-wide dense MLP projections.

Verification so far: ruff and `git diff --check` pass; three CPU Air conversion
tests pass; four CPU/meta forward/backward structural cases pass, including
input, weight and bias gradients. Three focused GPU numerical cases are ready
in `tests/unit/train/models/test_fp8_utils.py` but **not run yet**. Config dry-run
succeeded under `/home/prod/outputs/fp8-air-4n-micro/fp8-air-4n-compute-debug-20260910`.
No new job was submitted: Garrett's higher-priority pending allocation needs
the remaining nodes. Production jobs and shared runtimes were not modified.

No code or config changes in this investigation have been committed or pushed.

## Follow-up: all quantizable projections in FP8

The mixed-precision workaround below is no longer the only successful option.
The isolated `PRIME_RL_FP8_PAD_RAGGED=1` patch pads input tails with zeros instead
of leaving any dense/shared down projection in BF16. Weight loader metadata keeps
the original dimensions; kernel selection and runtime matmuls use padded inputs.
This retains 128x128 FP8 weights and dynamic per-128 activation quantization.

Reference recipe inspected: `/home/prod/research-prod/rlm5/prod/prod.toml` uses
`quantize_in_weight_transfer = true` and DeepGEMM. GLM Air lacked a
`convert_layer_to_vllm_kernel` implementation. The new Air converter reuses the
GLM-5 block quantizer/MoE layout, adds fused MHA Q/K/V and Q/K norms, and preserves
the padded input shape on the wire. Existing GLM-5 conversion behavior is unchanged.

| Job | Test | Result |
| --- | --- | --- |
| 1238 | TP8 + EP, initial all-FP8 padding implementation, BF16 checkpoint layerwise reload | 16 generations, 3,495 returned log-probabilities, zero NaN/Inf |
| 1239 | TP1, DeepGEMM, trainer-side FP8 conversion + actual NCCL transfer | All 47 state-dict groups received; 16 generations, 3,490 returned log-probabilities, zero NaN/Inf |
| 1240 | TP1, DeepGEMM, CUDA graphs, prefix cache disabled, two FP8 NCCL updates | Both 47-group transfers completed; 24 generations, 5,425 returned log-probabilities, zero NaN/Inf |

Runtime audit: 184 FP8 linear methods, 45 FP8 MoE methods; no unquantized linear
methods. Standard embedding/head and router/norm precision is not changed.
At TP8, 46 down projections are padded and remain `torch.float8_e4m3fn` before
and after checkpoint reload. At TP1, only the dense MLP down projection needs
padding: 10,944 -> 11,008 input columns. TP1 model memory: 100.93 GiB on one H200.

The NCCL test uses inference on GPU0 and a lightweight checkpoint-weight sender
on GPU1 of the same node. It uses PrimeRL's existing broadcast protocol and
kernel receiver, with the Air transfer converter, without constructing a trainer
or running an optimizer. No production jobs/configs/credentials are changed.
Initial reload probes used repeated prompts with prefix caching enabled; job
1240 explicitly disables caching to require fresh prefill after each update.
Job 1240 captured five mixed prefill/decode graphs and four full decode graphs;
capture took eight seconds and 0.12 GiB. All three batches completed basic
arithmetic and exact-text prompts. This is eight prompt types repeated across
three batches, not a task-quality evaluation. The sender reloads the original
checkpoint using trainer-side FP8 quantization, not optimizer-updated weights.
The latest implementation's padded-shape kernel-selection change is validated
by the TP1 tests; job 1238 used the earlier TP8 padding variant.

All follow-up diagnostic jobs have finished; node008 is released. Production
ablations remain running and untouched.

Three CPU conversion tests pass (including a byte-exact comparison against the
GLM-5 quantizer on unpadded values and zero checks on the padded tail).

Reproduce the no-TP transfer test:

```bash
cd /home/prod/fp8-blockwise-investigation
PRIME_RL_FP8_PAD_RAGGED=1 sbatch tools/fp8_inference_probe.sbatch \
  --tp 1 --backend deep_gemm --audit --nccl-reload --rounds 3 --cuda-graphs
```

Kernel-format transfer with TP>1, other backend layouts/GPUs, production-length
contexts, and a full RL optimizer step are not established by these tests.

## Earlier experiments

One-node, inference-only diagnostics on `qpn01gpu008` (8 H200 GPUs),
2026-09-09/10 UTC. No trainer, sandbox requests, W&B runs, or production changes.

Environment: isolated `/home/prod/fp8-blockwise-investigation`, vLLM 0.28.0,
torch 2.13.0+cu130, FlashInfer 0.6.16.post3. Original BF16 model snapshot:
`a24ceef6ce4f3536971efe9b778bdaa1bab18daa`.

## Confirmed failure

Job 1233, TP8 + EP, online `fp8_per_block` for both linear and MoE layers,
fails during engine profiling:

```text
AssertionError: the last dimension of `x` 1368 must be divisible by `group_size` 128
```

Trace: vLLM `input_quant_fp8.py:110` -> `quantization/utils/fp8_utils.py:563`.
The dense MLP has intermediate size 10,944; TP8 shards it into 1,368 values.
Its row-parallel down projection therefore violates the activation quantizer's
128-element group requirement. Shared-expert down projections are also ragged
at TP8 (1,408 / 8 = 176). Under EP, each routed expert retains its full aligned
1,408-wide intermediate dimension.

This is a reproducible startup failure in the tested environment, not yet a
demonstration of the cause of earlier production NaNs.

## Follow-up experiments

- Job 1234: TP8 + EP, `quantization="online"`,
  `quantization_config={"moe": "fp8_per_block"}`. Linear layers stay BF16.
  **Passed** at 23:51 UTC: 8 requests, 1,740 returned log-probabilities checked,
  zero NaN/Inf values. Answers include `4`, `Paris.`, `156`, and `hello world`.
  Two verbose answers reached the 128-token limit. Model loaded at 13.4 GiB/GPU;
  FlashInfer/CUTLASS was the selected MoE backend. First-time compilation and
  initialization dominated the 1,141.7-second wall time; this is not a throughput
  benchmark.
- Job 1235: dependent on successful completion of 1234, same node sequentially.
  `fp8_per_block` with dense/shared-expert `down_proj` excluded from quantization.
  Engine initialization passed (12.76 GiB/GPU), but the added precision audit
  failed before generation because callable RPC serialization is disabled by
  default. This is a probe failure, not an observed quantization failure.
- Job 1236: same narrow exclusions, rerun with
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` **only inside this trusted offline probe**
  to permit the local `apply_model` inspection callback. No HTTP server is exposed.
  Engine startup passed, but the second RPC serialization boundary rejected the
  `__main__` inspection function (`PicklingError`). No generation was attempted.
- Job 1237: identical narrow precision exclusions, with the optional inspection
  callback and serialization override removed. Four rounds of the same eight
  prompts (32 requests, not 32 independent tasks). **Passed**: all 32 requests
  generated, 7,220 returned log-probabilities checked, zero NaN/Inf values.
  Answers include correct arithmetic, Paris, hello world, and coherent prose.
  Model memory: 12.76 GiB/GPU. Probe wall time: 210.7 seconds, including startup.

## Tested workaround

Use TP8 + EP and online `fp8_per_block`, excluding the dense MLP and shared-expert
down projections from quantization. In the tested model this means
`model.layers.0.mlp.down_proj` and
`model.layers.{1..45}.mlp.shared_experts.down_proj`. The probe also lists unused
counterpart names in each layer; these have no matching module.

The resolved vLLM configuration retains 128x128 FP8 weight blocks for both linear
and MoE layer kinds and applies the explicit ignore list. Startup logs selected
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` and `CutlassFp8BlockScaledMMKernel`
for FP8 linear layers and
`FLASHINFER_CUTLASS` for FP8 MoE. The attempted per-module dtype audit did not
complete; do not present it as independent runtime verification.

This is mixed precision, not all-layer FP8. An upstream general solution needs
compatible handling of ragged input dimensions (for example padding or an
appropriate fallback), not removal of the quantizer's divisibility assertion.
The expert-only FP8 variant is a broader fallback with its own successful smoke
test. Neither result proves the cause of historical training-time NaNs.

Reproduce the narrow workaround on one node:

```bash
cd /home/prod/fp8-blockwise-investigation
sbatch tools/fp8_inference_probe.sbatch --tp 8 --skip-down-proj --rounds 4
```

All diagnostic jobs have finished and released their node. The four production
ablations were not modified.

Both use eager inference, context limit 4,096 and eight concurrent sequences.
Success here would not validate production context lengths, CUDA graphs,
trainer weight transfer, or task-level quantization quality.

Logs: `/home/outputs/int4-debug/fp8-blockwise/job_<jobid>.log`.
Reproduction entrypoints: `fp8_inference_probe.py` and
`fp8_inference_probe.sbatch` in this directory.
