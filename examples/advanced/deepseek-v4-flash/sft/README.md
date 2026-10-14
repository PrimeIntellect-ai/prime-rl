# DeepSeek V4 Flash FP8 QAT: decisions and validation

FP8 QAT uses FP8 forward GEMMs and BF16 backward GEMMs. Dense projections and routed experts are independent options; FP8 KV simulation is a third option. RN means round to nearest, and SR means stochastic rounding.

This change is stacked on [#4067](https://github.com/PrimeIntellect-ai/prime-rl/pull/4067), which provides the shared FP8 SR infrastructure and stateless SGD support. It selectively adapts the grouped projection and KV ideas from [#3627](https://github.com/PrimeIntellect-ai/prime-rl/pull/3627) at `d35a408f78e207555d418dd660c60b9ece621761`.

## Decisions

- **Keep a distinct QAT mode.** Ordinary DeepGEMM FP8 also quantizes backward GEMMs. QAT requests BF16 backward and follows the audited vLLM forward conventions. SR alone can use ordinary FP8 with `stochastic_rounding=true`.
- **Share the SR implementation.** QAT calls the same `FP8RoundingState`, parameter-identity setup, step setter, and E4M3 rounding helper as ordinary FP8+SR. Dense and expert QAT implementations inherit the shared interfaces. There is no separate QAT random-number implementation.
- **Reuse the forward-rounded weight in backward.** For a dense projection, `dX = dY @ decoded(W_fp8)` and `dW = dY.T @ X_bf16`; grouped projections use the corresponding grouped BF16 GEMMs. The weight decisions match forward exactly. Activations saved for weight gradients retain BF16 precision. This is an explicit straight-through surrogate, not the literal derivative of rounding. KV fake quantization has an identity straight-through gradient.
- **Follow vLLM for QAT numerical conventions.** The comparison target is pinned vLLM **0.31.0**, including its two-stage weight cast. Ordinary FP8 keeps its established scale conventions. The protected `optimization_dtype` and `reduce_dtype` settings retain their FP32 defaults.
- **Use SGD to validate full depth on two nodes.** EP16, momentum-free SGD, sequence 512, and smaller NCCL buffers reuse the successful configuration from #4067. These checks do not establish AdamW behavior or a quality benefit.

## BIG NOTICE — TRAINER / vLLM QUANTIZATION CONVENTIONS DIFFER

**Sharing an SR mask does not imply identical final FP8 weights when scales or requantization stages differ. Stock vLLM 0.31.0 uses RN in the audited online weight cast; enabling trainer SR does not enable serving SR. Quantizer parity also does not establish whole-model logit parity.**

| Convention | Difference | QAT decision and evidence |
| --- | --- | --- |
| Weight floor | Ordinary trainer FP8 floors **scale** at `1e-4`; vLLM floors **amax** at `1e-4` before dividing by 448. Minimum scales differ by 448x. | Use vLLM's amax floor and operation order. H200 payload/scales match actual vLLM references for normal, tiny, odd, and strided inputs. |
| Weight requantization | vLLM online-casts first, then DeepGEMM requantizes decoded weights to UE8M0. The audited vLLM default enables UE8M0 on Hopper and Blackwell; ordinary trainer FP8 selects it on SM100+. | Reproduce both stages. SR applies to the first stage; the following deterministic requantization means the final resident weight is not guaranteed to be an unbiased SR estimate. |
| Negative zero | E4M3 `+0` is byte `0x00`; `-0` is `0x80`. Numeric equality hides the difference. | Preserve the original sign bit, including negative tiny values rounding to zero. Compare FP8 bytes and decoded BF16 bit patterns. |
| Stochastic draws | #3627's optional serving extension uses rank-local RNG and a nearest/other-neighbor draw. Shared FP8+SR uses stateless Philox and a lower/upper-magnitude draw. The distributions agree, but identical uniforms can select different values. | Use #4067's exact seed, salt, counter, and rounding helper. A serving SR extension must adopt the same contract; matching only a user seed is insufficient. |
| Generic activation floor | vLLM's production CUDA UE8M0 path floors **scale** at `1e-10`; the trainer/fused output-projection convention floors **amax** at `1e-10`. Minimum power-of-two scales are `2^-33` and `2^-42`. | Dense/expert QAT uses the CUDA scale floor. Grouped output projection follows its fused kernel's amax floor. Tiny-input byte checks pass on H200. |
| Grouped output projection | Ordinary dense replacement does not preserve `o_a_proj`'s grouped semantics. vLLM performs inverse RoPE in FP32 and casts directly to FP8. | Preserve groups and fuse inverse RoPE with FP8 activation casting, avoiding an intervening BF16 round. Actual fused-vLLM byte/scales checks pass. |
| Compact scale encoding | Hopper uses FP32 containers for power-of-two scale values; Blackwell uses packed UE8M0 exponent bytes, with distinct strides and padding. | Keep numerical scales separate from their storage format; let DeepGEMM transform them at the GEMM boundary. **B300 execution and packing remain unvalidated**; SM103 compilation passes. |
| KV layout and floors | DSV4 serving stores 448 E4M3 channels, 64 BF16 RoPE channels, and seven exponent scales in eight metadata/padding bytes: 576 data bytes plus eight metadata bytes per token. SWA floors amax; compressed KV floors scale, both at `1e-4`. | Preserve both floor rules. Fake quantization returns decoded BF16 values, not a physical paged-cache byte buffer. H200 checks use actual vLLM SWA/compressor writers, compiled STE, and CSA/HCA integration. |
| Sparse indexer | Ordinary trainer amax floor is `1e-10`; the audited serving query/key floor is `1e-4`. | Use `1e-4` when QAT is enabled. Actual fused-vLLM query bytes/scales match on H200. |
| Remaining attention boundaries | vLLM fuses compressor normalization/rotation for unquantized RoPE channels; the trainer retains a BF16 intermediate. Pooling/reduction implementations also differ. | Retain the trainable compressor/autograd pipeline. End-to-end serving logit parity is not claimed. |

### Exact shared SR contract

The random decision belongs to a logical weight element and optimizer step. Rank, microbatch, kernel launch order, and ambient CUDA RNG consumption do not enter the key.

1. Define `hash63(s)` as the little-endian integer from `BLAKE2b(s.encode(), digest_size=8)`, masked with `(1 << 63) - 1`.
2. The parameter salt is `hash63(canonical_parameter_name)`. Use the first deduplicated `named_parameters()` name after MoE runtime setup and before checkpoint/compile/FSDP wrapping. Tied parameters share the salt; EP retains the names.
3. The step seed is `hash63(f"{base_seed}:{optimizer_step}")`; the Philox stream is `step_seed XOR salt`.
4. The counter is `((global_expert * N) + output_row) * K + input_column`, using 64-bit coordinates. Dense matrices use expert zero. Expert weights use logical `[E,N,K]` coordinates; EP supplies the global expert offset. FSDP casts after gathering full quantization blocks.
5. Use Philox4x32-10 with counter words `(lo, hi, 0, 0)`, stream words `(lo, hi)`, and pinned Triton's `tl.rand` float mapping. Select the upper magnitude neighbor when `uniform < (magnitude - lower) / (upper - lower)`; otherwise select the lower neighbor. Restore the original sign bit.
6. All accumulation microsteps and checkpoint replays within a step reuse the realization. Advancing the step replaces the seed tensor; saved forward seeds remain immutable for backward reconstruction. Seeds/salts are tensor attributes rather than buffers, avoiding DTensor conversion.

Gathered FSDP weights, DP/CP replicas, tied parameters, and whole-expert EP shards follow this contract. A future layout that casts partial row/column shards must supply global coordinates and full logical 128x128 block scales. Independently scaling partial blocks would violate it.

## Configuration

For the validated two-node SR setup:

```bash
uv run sft \
  @ examples/advanced/deepseek-v4-flash/sft/h200/base.toml \
  @ examples/advanced/deepseek-v4-flash/sft/h200/two-node-sgd.toml \
  @ examples/advanced/deepseek-v4-flash/sft/fp8-qat.toml \
  @ examples/advanced/deepseek-v4-flash/sft/stochastic-rounding.toml \
  --run.name qat-sr-two-node --slurm
```

- Omit `stochastic-rounding.toml` for RN.
- For CP2, add `@ examples/advanced/deepseek-v4-flash/sft/h200/two-node-sgd-cp2.toml`. This sets CP2 and global batch 8, retaining one microstep per optimizer step.
- `fp8-qat.toml` enables dense QAT, expert QAT, and KV simulation. These settings can be selected independently using `model.quantization.qat`, `model.moe.compute.qat`, and `model.dsv4_kv_precision`.
- The four-node `h200/base.toml` alone uses full CPU AdamW offload. **The successful results below require the two-node SGD overlay**; they do not validate that base-only AdamW run.
- For exact checkpoint reproduction, override both `--model.name` and `--tokenizer.name` with a local snapshot directory at the revision below. The validation runs used that pinned cached snapshot. Infrastructure-specific cache paths and credentials are not part of these portable examples.
- In RL, model settings live under `trainer.model`. This report validates SFT; it does not claim an end-to-end RL or serving run.

## Full-depth H200 results

Collected on **2026-10-13** at runtime commit **`f1eee78b725b3dafc2281144b8169be780806c64`**. The later merge of #4067 preserves `src/`, `packages/`, `tests/`, dependencies, and lockfiles byte-for-byte from that tested tree; it adds the parent's documentation/config ancestry. The report commit adds documentation only.

- Full **43-layer** `PrimeIntellect/DeepSeek-V4-Flash-Base-bf16`, checkpoint revision `f6844322b0844389ca71de948dadb74dd9bfa4a6`.
- `PrimeIntellect/INTELLECT-4-SFT`, revision `22e1ce974e3c5b20ad229674ea1712172f77c1f3`, subset `i3-math--null--low`, data seed 42.
- **Two nodes / 16 H200 GPUs**, EP16, sequence length 512, SGD learning rate `1e-5`, FP32 optimization/reduction defaults, compiled layers, full activation checkpointing, activation offloading with five inflight activations, and FP8 KV simulation. SR seed `20261012`.
- `model.offload="optimizer"`; `NCCL_BUFFSIZE=1048576`, `NCCL_MAX_CTAS=4`.
- Slurm **4082**, all phases **COMPLETED / exit 0**, total elapsed **15m57s**. Phases ran sequentially on the same two nodes, in SR/CP1, RN/CP1, SR/CP2 order.
- **All 15 optimizer steps had finite losses and gradient norms and zero NaN-loss counts.**

| Mode | DP / CP / EP | Global batch | Losses, steps 1–5 | Peak reserved GiB | W&B |
| --- | --- | ---: | --- | ---: | --- |
| SR | 16 / 1 / 16 | 16 | 0.93577, 1.04991, 1.03535, 1.06745, 1.01928 | 138.172 | [w0bptcdu](https://wandb.ai/primeintellect/fp8-qat-sft/runs/w0bptcdu) |
| RN | 16 / 1 / 16 | 16 | 0.93796, 1.04780, 1.03550, 1.06919, 1.01822 | 138.162 | [g3n42bfv](https://wandb.ai/primeintellect/fp8-qat-sft/runs/g3n42bfv) |
| SR | 8 / 2 / 16 | 8 | 0.94656, 0.93943, 1.07837, 0.98706, 1.07499 | 137.996 | [9qsgw68k](https://wandb.ai/primeintellect/fp8-qat-sft/runs/9qsgw68k) |

| Mode | Step times in seconds, steps 1–5 | Last-two-step median |
| --- | --- | ---: |
| SR / CP1 | 130.543, 18.668, 13.794, 10.858, 11.619 | 11.239 |
| RN / CP1 | 41.211, 11.698, 11.440, 10.829, 10.233 | 10.531 |
| SR / CP2 | 41.651, 12.187, 12.785, 11.190, 25.572 | 18.381 |

CP1 SR was **6.7% slower than RN** using the last two samples. This is preliminary: two samples, sequential runs, first-step compilation, and allocator retries do not establish a stable overhead estimate. CP2 changes global batch, and its final step was an outlier. Five steps demonstrate functionality, not convergence or a quality advantage.

**Memory is tight.** The reported peak is rank 0's `torch.cuda.max_memory_reserved()`, not a maximum across all ranks or all device allocations. Device capacity was 139.80 GiB. Recovered allocator mapping failures occurred; external allocations such as NCCL consume part of the apparent remaining capacity. These successful short runs do not establish that larger batches or sequences fit.

Exact per-step losses, times, gradient norms, and W&B links are in [validation-results.json](validation-results.json). Original run folders are under `outputs/sft/`:

- `qat-shared-sr-full-ep16-cp1-sgd`
- `qat-shared-rn-full-ep16-cp1-sgd`
- `qat-shared-sr-full-ep16-cp2-sgd`

The local `outputs/validation/slurm/shared-sr-suite-manifest.json` records config/launcher hashes. Original logs and manifests remain local run artifacts rather than repository files.

### Earlier failed attempts

Full CPU optimizer offload did not produce a successful full-depth step: job 3930 logged CPU optimizer timeouts and its two nodes became cordoned/unresponsive; retry 3949 exited with SIGSEGV before a completed step. Neither establishes OOM as the root cause. Retry [v4aqx517](https://wandb.ai/primeintellect/fp8-qat-sft/runs/v4aqx517) has no completed-step metrics. The dependent four-node AdamW suite was canceled. The successful SGD configuration does not resolve those failures or validate AdamW/full offload.

## Correctness checks

At the tested runtime revision:

| Check | Result | Coverage |
| --- | --- | --- |
| H200 GPU suite, one allocated node | **101 passed** | QAT and ordinary FP8+SR, actual vLLM cast references, independent Philox reference, negative zero, matching cross-mode masks at matching scales, wide counters, eager/compiled BF16 gradients, empty experts, tied weights, checkpoint replay, inverse RoPE. |
| Two-rank NCCL suite, same node | **3 passed per rank** | FSDP, CP token splitting, and EP shard/whole-weight consistency, including input gradients. |
| Config tests | **245 passed** | Existing configuration behavior and SR/SGD options. |
| MoE regression tests | **30 passed** | Expert runtime and model conversion checks. |
| SM103 compile checks | **9 variants passed** | Weight cast/requantization, KV reference, inverse RoPE. **Compilation only; no B300 hardware execution.** |

The three distributed cases are skipped by the ordinary single-process command and run separately:

```bash
uv run pytest tests/unit/train/models/test_fp8_qat.py \
  tests/unit/train/models/test_fp8_rounding.py \
  tests/unit/train/models/test_fp8_utils.py -q

uv run torchrun --standalone --nproc-per-node=2 -m pytest \
  tests/unit/train/models/test_fp8_qat.py -q -k distributed

uv run pytest tests/unit/test_configs.py -q
uv run pytest tests/unit/train/models/test_moe.py \
  tests/unit/train/models/test_moe_conversions.py -q
```

Earlier H200 validation also checked the actual vLLM SWA/compressor cache writers, indexer query quantization, compiled KV STE, and CSA/HCA attention integration. The KV/indexer implementation is unchanged in the tested runtime revision. One- and four-layer SFT runs were useful smoke checks; the table above reports full depth.

## Operator measurements

H200 forward-plus-backward CUDA milliseconds at the tested runtime revision. Each case used three warmups followed by five timed repetitions with CUDA events and synchronization. Inputs and weights were BF16; every mode used the same shapes and tensors within its comparison.

| Mode | Dense M1024/K8192/N4096 | Experts E32/M4096/K4096/N2048 |
| --- | ---: | ---: |
| BF16 | 0.279 | 0.596 |
| Ordinary FP8 RN | 0.813 | 2.832 |
| Ordinary FP8 SR | 0.940 | 4.851 |
| QAT RN | 0.558 | 1.455 |
| QAT SR | 0.652 | 4.183 |

These are short operator measurements, not end-to-end training throughput. BF16 was fastest at these token counts. QAT's BF16 backward was faster than ordinary FP8 backward at these shapes, while SR added substantial casting cost, especially for experts. This evidence supports keeping QAT for the requested backward precision and serving conventions; it does not establish a general speed or quality advantage. Ordinary FP8+SR remains sufficient when SR is the only requested change.

## Remaining validation

- **B300 runtime is unvalidated:** no B300/GB300 resource was available for the requested execution checks. SM103 compilation is not a substitute.
- **PTX KV implementation is pending:** retain the Triton reference and begin the PTX alternative only after the requested validation gate, including B300, is satisfied. Require exact parity and performance measurements.
- Longer, optimizer-matched training is required before claiming a quality benefit. These five-step SGD results do not establish AdamW stability or resolve full CPU offload failures.
