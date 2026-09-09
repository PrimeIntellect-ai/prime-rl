# Train–inference mismatch experiments

Reproduction work for <https://kiddyboots216.github.io/mismatch/> in Prime-RL's
trainer and vLLM. The base configs establish measured baselines; opt-in alignment
overlays share forward arithmetic. The article uses XoRL and modified SGLang
kernels, so every Prime-RL arm requires its own exact-zero audit.

Run stages sequentially, using at most **two nodes total: one trainer and one
inference node**, each with eight H200s. Run reverse text, then single-turn GSM8K,
for Qwen3-8B, Qwen3-30B-A3B, and finally GLM-4.5-Air. Each task starts from the
named pretrained checkpoint; math does not resume the reverse-text checkpoint.

Compact completed-run audits and operator-probe results are included in
[`evidence/`](evidence/README.md). Full rollout traces and cluster logs remain
under the original run output directories.

## Launch

Initialize submodules and install the environment packages first:

```bash
git submodule update --init --recursive
uv sync --all-extras --package prime-rl --package reverse-text --package gsm8k
```

The cluster overlay selects partition `all`. Use a cluster-specific overlay on
another installation. Run from the repo root:

```bash
# Frozen weights first: three batches, optimizer LR zero.
uv run rl @ configs/experiments/mismatch/reverse-text.toml @ configs/experiments/mismatch/frozen.toml @ configs/experiments/mismatch/h200-slurm.toml --output-dir outputs/mismatch --run.name qwen8b-reverse-frozen

# Dense Qwen learning baseline.
uv run rl @ configs/experiments/mismatch/reverse-text.toml @ configs/experiments/mismatch/h200-slurm.toml --output-dir outputs/mismatch --run.name qwen8b-reverse

# Single-turn math, with a separate GSM8K test split.
uv run rl @ configs/experiments/mismatch/reverse-text.toml @ configs/experiments/mismatch/math.toml @ configs/experiments/mismatch/h200-slurm.toml --output-dir outputs/mismatch --run.name qwen8b-math
```

For subsequent models insert `@ configs/experiments/mismatch/qwen30b.toml` or
`@ configs/experiments/mismatch/glm45air.toml` after `reverse-text.toml`; use a
fresh run name. The math overlay replaces the train source. Append
`@ configs/experiments/mismatch/full-offload.toml` if optimizer-state offloading
and activation offloading do not provide enough memory. Full offload changes
the gradient rounding path and disables gradient clipping; hold the offload
mode fixed across paired comparisons and record it. Neither overlay changes
`optimization_dtype` or `reduce_dtype`.

Append `--dry-run --no-dashboard` to validate and generate SLURM scripts without
submitting. Do not launch stages concurrently or increase node counts to fit a
model. The twenty-step runs are diagnostics, not evidence of converged learning.

For the frozen long-input arm, first generate the deterministic local taskset
from the repository root:

```bash
uv run python tools/make_long_reverse_dataset.py outputs/mismatch/long-reverse
```

Compose `long-reverse.toml` and `frozen.toml` after the model's aligned overlays.
The dataset path is relative to the repository root. On distributed launchers
with a different working directory, override `orchestrator.train.source.0.env.taskset.dataset_name`
with the absolute path to this shared directory. GLM uses batch size8 to keep
one packed microbatch per trainer rank.

After the frozen baseline, append
`@ configs/experiments/mismatch/vllm-batch-invariant.toml` in a separate run to
measure vLLM's serving-side batch-invariance ablation. Keep the model, task,
sampling and optimizer settings identical. This arm still requires the exact
logprob checks; it does not supply matching trainer kernels.
The overlay also applies vLLM's NCCL settings to both nodes so the weight
broadcast communicator uses consistent protocols. These communication settings
also affect trainer collectives; frozen-weight measurements isolate the forward
comparison from optimizer updates.

For the MoE attention layouts, run
`uv run python tools/probe_mismatch_ops.py <output.json> --moe-layouts`
on one Hopper GPU. This checks 32-query/4-KV and 96-query/8-KV heads at lengths
257,1,024,8,192, comparing prefill with decode and paged decode and checking
gradients against an FP32 reference. These operator results do not establish
full-model GLM fit or equality.

## Measurement

Sampling uses temperature 1 and the full vocabulary. Training thinking is disabled
through the renderer; math evaluation disables it explicitly through chat-template
kwargs because evaluation uses the chat API. Each environment permits one model
turn, and `max_off_policy_steps = 0` bounds
staleness. Prefix caching is disabled. The loss config uses IPO's importance
ratio with a probability-difference threshold of 1 and zero KL penalty, giving
the unclipped importance-sampling policy gradient for finite valid logprobs.

Audit actual trained trace versions after each run:

```bash
uv run python tools/audit_mismatch_policy.py outputs/mismatch/<run_name>
```

For a completed zero-mismatch arm, run
`uv run python tools/audit_mismatch_zero.py outputs/mismatch/<run_name>`.
This independently compares the raw FP32 trace bits, requires every configured
step and full-precision trace storage, includes the policy-version audit, and
reports actual long-position coverage. Its `verified` result covers that run's
sampled tokens; it does not certify untested models, lengths, or optimizer modes.
Aligned MoE diagnostics also compare one router parameter's FP32 bits before and
after each optimizer step. The audit reports those changed steps separately;
this supplies evidence of parameter updates when full CPU offload disables the
gradient-norm metric. A zero change count for one router does not establish that
every model parameter was unchanged. For positive-learning-rate runs with probe
records, the audit additionally requires at least one subsequent scored policy
after an observed router update. Frozen runs are exempt from this requirement.

If GSM8K groups supply no advantage signal, `math-harder.toml` composed after
`math.toml` selects Hendrycks MATH training and MATH-500 evaluation. Keep it as a separately
named experiment and report its results separately. It retains single-turn
sampling and disables `math-env`'s reference-judge fallback for deterministic
math verification. Validate dataset availability before submitting this arm;
a successful config dry run does not load the dataset. This overlay uses Docker
because these tasksets require network isolation. Check Docker availability on
the orchestrator node before submitting it. Where isolation backends are absent,
`math-uniform.toml` instead retains GSM8K and samples across its full training set
using the existing seeded pool sampler with equal weights. It changes task
selection, not sampling temperature or numerical settings.

This requires generation start/end and the trainer's incoming version to equal
the shipped batch step minus one. A missing version record fails the audit.
Keep this requirement during learning as well as frozen-weight measurements.
The live synchronized shipment check aborts on a mismatched policy span.
The frozen overlay retains zero-advantage samples so numerical coverage includes
uniform-reward groups. Keep that filtering choice fixed within paired runs;
learning runs use the default pruning unless explicitly overridden.

While the same frozen run's inference server is live and startup broadcast has
completed, replay a small set of trained sequences through inference prefill:

```bash
uv run python tools/replay_mismatch_prefill.py outputs/mismatch/<run_name> http://<inference-host>:8000 4
```

The server must hold that run's unchanged weights. The tool scores the exact
recorded token IDs and compares trainer scoring, serving prefill, and recorded
decode logprobs; it rejects nonzero learning rates. For isolated dense operator
comparisons on one idle GPU, use
`uv run python tools/probe_mismatch_ops.py /path/to/operator-probe.json`.
These probes localize disagreement; operator agreement alone does not certify
the full model.

`dense-alignment.toml`, composed after `vllm-batch-invariant.toml`, is an
experimental shared forward path for dense Qwen3. It uses eager trainer and
serving execution, common fixed-K matrix multiplication, RMSNorm, CPU-generated
RoPE tables and CUDA rotation, SwiGLU, vLLM FA4, and full-vocabulary log-softmax.
The full-vocabulary head uses additional memory.
The overlay explicitly disables `inference.enable_fp32_lm_head`: the shared
projection emits BF16 logits on both sides, then computes FP32 log-softmax.
The default inference FP32-output projection bypasses this shared linear path.
This candidate is restricted to Hopper with 128-dimensional attention heads;
it overrides vLLM 0.28's automatic FA4-to-FA2 batch-invariance fallback.
Validate GPU gradients,
prefill/decode equality, and full-model sampled logprob bits before interpreting
this arm as aligned; the overlay itself provides no zero-mismatch guarantee.

Append `fp32-head.toml` to retain the head's FP32 accumulator output on both
sides. This matches the article's head precision while retaining a full logits
tensor for now. Its standalone GPU check is
`uv run python tools/probe_mismatch_ops.py /path/to/head.json --fp32-head-only`.
The shared sampler reduction covers both vLLM's original sampler and the V2
model runner's selected-token and prompt-logprob paths.

For Qwen3-30B-A3B, compose `moe-alignment.toml` after `qwen30b.toml` and both
dense overlays. This experimental arm uses EP1/TP1, FP32 routing with explicit
IEEE dot products, shared expert GEMMs/SwiGLU, and FP32 expert summation in
ascending expert-ID order. It preserves learned FP32 router weights during
weight transfer; downcasting these weights breaks agreement after updates.
It is deliberately eager and can be slow. Its new
Python-only `mismatch_router` kernel lives in the `deps/prime-kernels` checkout;
the experiment runtime includes that checkout on `PYTHONPATH`. The installed
kernel wheel does not include this unpublished operator. Run
`uv run python tools/probe_mismatch_moe.py /path/to/moe-probe.json` before a
full-model run. The probe covers batch shapes, gradients, and empty experts;
it does not establish full-model or GLM alignment.

The tested router change is supplied as an applyable patch because publishing
to the kernel repository was unavailable. Before running the MoE arms, apply it
to a clean pinned submodule and expose its source registry to both trainer and
serving processes:

```bash
git -C deps/prime-kernels apply --unidiff-zero --check ../../configs/experiments/mismatch/patches/0001-ieee-fp32-router.patch
git -C deps/prime-kernels apply --unidiff-zero ../../configs/experiments/mismatch/patches/0001-ieee-fp32-router.patch
export PYTHONPATH="$PWD/deps/prime-kernels${PYTHONPATH:+:$PYTHONPATH}"
```

The experiment uses the submodule source; it does not publish a kernel wheel
or change the installed release pin. The patch is based on kernel commit
`3fb83eb`; it must land in prime-kernels and become a submodule/release update
before treating this draft as a normal installed-runtime integration. Do not
reapply it to a checkout where the kernel already exists.

The staged `glm-alignment.toml` variant uses the sigmoid router and selection
bias, partial rotary embeddings, and a separately added shared expert. It
requests 150 GiB of CPU weight offload per TP1 serving replica. It also enables
exact-size pinned allocations above1MiB with
`pinned_max_round_threshold_mb:1`. The tested PyTorch2.13 allocator otherwise
rounds GLM's2.75GiB and1.375GiB expert tensors to4GiB and2GiB blocks, so the
logical offload budget understates host RAM. Validate this control with
`uv run python tools/probe_mismatch_pinned_memory.py <output.json>` under the
serving allocator environment before loading all eight replicas. It changes
allocation sizes, not weight dtypes or values. Serving uses an 8,192-token
context cap and `gpu_memory_utilization=0.75` to leave GPU headroom during
decode and weight updates. On these H200s, the measured KV cache is 53.83 GiB
per replica, with about 34.8 GiB free during warmup. Monitor both GPU and host
memory through actual training; checkpoint loading alone does not prove fit.
The variant also enables
full CPU optimizer offload: state-only offloading still
materializes Adam states on the GPU during updates. Full offload disables
gradient clipping through the existing validator. Its native Linux backend
releases consumed gradient pages to reduce resident host RAM. Do not append the
boolean `full-offload.toml` overlay after this table: it would reset those options.
Unfinished accumulation still needs all gradient pages, so the large GLM run
must fit one packed microbatch per rank; use batch size 8 for its long math arm.
Estimate checkpoint bytes from safetensors headers, since GLM's index metadata
underreports them. The operator
probe also checks GLM routing, preservation of unrotated RoPE channels and
their gradients, and exact GEMM results with UVA views of CPU-resident weights.
It copies updated FP32 router and BF16 expert weights through those UVA views
and checks both the stored bits and subsequent forward results.
The frozen short-reverse GLM run passed full-model memory, raw-logprob, and live
prefill checks on this cluster. The twenty-step reverse and single-turn GSM8K
learning runs passed the complete exact-zero and updated-policy audits. The
three-step frozen long-input run also passed, covering12,288 sampled tokens at
positions593 or later and sequences up to4,137 tokens. Its live prefill replay
matched trainer and decode exactly. See `RESULTS.md` for counts and limitations;
these short diagnostics do not establish converged learning or arbitrary-length
numerical equivalence.

`inference-swiglu.toml` is a separate dense-forward ablation: it calls vLLM's
SwiGLU kernel from the trainer and uses the eager two-stage PyTorch derivative.
It introduces graph breaks and requires `trainer.model.compile.fullgraph=false`.
Use the GPU operator probe to verify forward bits and both input gradients before
launching this arm. It does not align attention, RoPE, RMSNorm, or the LM head.

`trainer.model.debug.mismatch_diagnostics = true` adds sampled-token metrics:

| Metric prefix (suffix `/all/mean` or `/all/max`) | Meaning |
| --- | --- |
| `logprob_abs_error` | Absolute trainer minus sampling logprob |
| `logprob_bit_mismatch` | Different FP32 bits or a non-finite pair; mean is a fraction |
| `logprob_nonfinite` | Fraction of pairs with a non-finite value |
| `mismatch_k3_stable` | `expm1(delta) - delta`, evaluated in FP64 then stored in FP32 |

These supplement `mismatch_kl/all/mean`; they do not change the loss. The
existing FP32 K3 expression can round to zero for different logprobs. An exact
match requires zero bit mismatches, zero non-finite pairs, and nonempty token
coverage. Trace and trainer-annotation floats are retained at full precision
by setting `float_decimals = "None"` on both component file monitors.

Read metrics under `<run_dir>/monitors/file/metrics.jsonl`, plus the resolved
configs, logs and trace annotations. Compare reward, held-out math accuracy,
token coverage, truncation, staleness and throughput alongside mismatch.

## Alignment ladder

1. Measure frozen-weight reverse-text scoring, with repeated batches.
2. Compare the same token sequences across batch shapes and sequence lengths,
   including positions beyond 593 and math-length continuations.
3. Separate trainer versus inference prefill from inference prefill versus
   cached decode using identical weights and selected tokens. Live trainer
   metrics alone combine these effects; they do not isolate the first bad layer.
4. Localize divergent operations, then align forward arithmetic and verify
   backward correctness. Audit RoPE tables/rounding, RMSNorm, GEMMs, SwiGLU,
   attention reductions, and the vocabulary reduction. For MoEs also inspect
   router decisions, expert GEMMs, and expert-output reductions.
5. Repeat during optimizer updates. Only label an arm zero mismatch after every
   sampled token passes the exact check at the same weight version. Compare
   matched learning runs over multiple seeds before claiming a reward benefit.

Synchronized rollouts, full-precision trace export, and vLLM batch invariance
alone do not establish cross-engine bitwise agreement.
