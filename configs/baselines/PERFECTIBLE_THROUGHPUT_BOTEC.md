# Perfectible-subset throughput BOTEC

Parametric back-of-the-envelope for sizing `max_concurrency`, `max_num_seqs`, and
related knobs on the 4 perfectible-subset baselines configs.

Last calibrated: 2026-05-13, against cached model configs and in-flight
perfectible-subset runs on Isambard alloc 4574749 (4×GH200/node × 7 inference
nodes after head exclusion;
see `isambard_srun_pty_anti_pattern.md` for why head is excluded).

---

## 1. Architecture inputs

All numbers verified from `config.json` of each cached HF snapshot, not vibed.

| Symbol | Quantity | gemma-4-E4B-it | OLMo-3-7B-Instruct-DPO | rnj-1-instruct | gemma-4-26B-A4B-it |
|---|---|---|---|---|---|
| L | Layers | 42 | 32 | 32 | 30 |
|   | (SWA / global breakdown) | 35 / 7 | 24 / 8 | Gemma3-style `sliding_window=32768`, effectively full at 16k | 25 / 5 |
|   | (SWA window) | 512 | 4096 | 32768 | 1024 |
| H | hidden_size | 2560 | 4096 | 4096 | 2816 |
| Hq | num_attention_heads | 8 | 32 | 32 | 16 |
| Hkv | num_kv_heads | 2 (GQA-4) | 32 (**MHA**) | 8 (GQA-4) | 8 SWA / 2 global |
| d | head_dim | 256 / 512 (heterog.) | 128 | 128 | 256 / 512 (heterog.) |
| KV-share | num_kv_shared_layers | **18 SWA layers share KV** | none | none | none |
| Wbf | bf16 weight memory | 14.894 GiB full checkpoint / 14.014 GiB language | 13.594 GiB | 30.959 GiB | 48.067 GiB total / **16.163 GiB per EP=4 rank** / 8.187 GiB active top-8 |
| KV/seq @ L=2000 (bf16) | KV-cache bytes for one sequence after 2000 generated tokens | 71.7 MiB | 1000.0 MiB | 250.0 MiB | 239.1 MiB |
| KV/seq @ L=2000 (FP8) | Same with `kv_cache_dtype="fp8"` | — (blocked, see below) | 0.50 GiB | 0.122 GiB | — (blocked) |
| FP8 KV available? | | unvalidated locally | ✓ | ✓ | unvalidated locally |
| Notable | | multimodal checkpoint; language stack is ~14.014 GiB | MHA, no GQA → massive KV | Gemma3-arch, 32768 window is effectively full at current max len | 128 experts top_k=8, moe_intermediate_size=704 |

**Gemma4 FP8 KV is not a proven-safe knob yet.** Earlier notes called it
"blocked" due to head_dim=512, but local vLLM FlashAttention code reports
head_dim≤512 and FP8 support on FA3/Hopper paths. Treat any Gemma4 FP8-KV claim
as empirical until a local smoke validates or falsifies it.

---

## 2. Hardware inputs (per GH200 node)

| Symbol | Quantity | Value |
|---|---|---|
| BW | HBM bandwidth | 4.0 TB/s |
| C | BF16 compute peak (sustained) | ~700 TFLOPS (≈ 70% of 990 TFLOPS marketing peak) |
| M | HBM capacity | 96 GiB |
| η | Observed vLLM efficiency (38% from run 1) | 0.30–0.45 of mem-BW peak |
| N_GPU | GPUs in service (after exclude-local) | 28 (= 7 nodes × 4 GPUs) |

---

## 3. Parametric BOTEC equations

### 3.1 Decode throughput per GPU (memory-bandwidth bound)

For batch size B per GPU, dense models:
$$\text{tok/s}_{\text{peak}}(B) = \frac{B \cdot \text{BW}}{W_{bf} + B \cdot k}$$
where `k` is the effective KV-cache bytes read per decode token per sequence.
`k` depends on (a) avg sequence length cached, (b) KV dtype, (c) layer heterogeneity:

For a model with `L_swa` SWA layers (window `w`), `L_glb` global layers, `H_kv` KV heads,
head_dim `d` (or `d_swa`, `d_glb` if heterogeneous), `dtype_bytes` per element:

$$k = \underbrace{L_{swa} \cdot \min(L_{seq}, w) \cdot 2 H_{kv,swa} \cdot d_{swa} \cdot dtype}_{\text{SWA layers}} + \underbrace{L_{glb} \cdot L_{seq} \cdot 2 H_{kv,glb} \cdot d_{glb} \cdot dtype}_{\text{global layers}}$$

At realistic vLLM efficiency: `tok/s_observed(B) ≈ η · tok/s_peak(B)`.

### 3.2 Effective batch per GPU — IMPORTANT distinction

`max_num_seqs` is a **ceiling**, not a target. The actual concurrent sequences vLLM
processes at steady state are determined by the slowest of:

1. **Client feed**: `max_concurrency / N_GPU` (verifiers harness cap, divided across GPUs)
2. **Engine ceiling**: `max_num_seqs` (per-engine slot count)
3. **KV memory**: how many seqs fit under `gpu_memory_utilization × M − W_bf`

Steady-state actual:
$$B_{\text{actual}} = \min\left(\frac{\text{max\_concurrency}}{N_{GPU}},\ \text{max\_num\_seqs}\right)$$

In our 256 → 1024 jumps, we've moved B_actual from ~9 → ~36 per GPU. We never approach
max_num_seqs=192 unless max_concurrency exceeds ~5376.

**Why OLMo3 didn't OOM at bf16-KV-equivalent budget**: even though `192 seqs × 1.0 GiB
bf16 = 192 GiB > 96 GiB GH200`, the actual concurrent count was ~9, KV used was ~9 GiB.
Setting max_num_seqs=192 was safe because the client never fed enough to saturate the
ceiling. The "edge of KV budget" wording is only relevant *at saturation* (`max_concurrency
≥ max_num_seqs × N_GPU`).

### 3.3 KV memory budget

A vLLM engine can host at most:
$$B_{\text{max,KV}} = \frac{M \cdot \text{gpu\_memory\_utilization} - W_{bf}}{K_{\text{per seq @ max\_model\_len}}}$$

For our perfectible runs, `max_model_len = 16384`. Each sequence's KV is reserved up to that
context regardless of actual length (scheduler_reserve_full_isl=true).

### 3.4 End-to-end wall time

For a run of `R` rollouts each generating `T̄` output tokens at sustained per-GPU rate `r(B)`:
$$\text{wall\_sec} \approx \frac{R \cdot \bar{T}}{N_{GPU} \cdot r(B_{\text{effective}})}$$

### 3.5 Cost of doubling max_concurrency

For batch sizes well below the memory-BW crossover `B* ≈ W_{bf} / k`, scaling is near-linear:
$$\text{speedup}(B \to 2B) \approx \frac{r(2B)}{r(B)} = \frac{2B \cdot (W_{bf} + B \cdot k)}{B \cdot (W_{bf} + 2B \cdot k)} \to 2 \text{ as } B \to 0$$

Above `B*`, the read-cost of KV per forward dominates and scaling tapers. For dense ~7-8B models
on GH200 with FP8 KV, `B*` is roughly 100-200 per GPU.

---

## 4. Per-model calibration (use these as starting points)

Inputs assumed: 7 inference nodes × 4 GPUs = 28 GPUs. `max_model_len = 16384`. FP8 KV where available.
Output token median ≈ 2000 (measured in run 1; expect similar for the other math models).

### 4.1 gemma-4-E4B-it

- `W_bf = 14.014 GiB` for the language stack, `k` at L=2000 ≈ 71.7 MiB
  (mostly global layers, SWA windows capped at 512)
- Crossover batch B*: `14014 / 71.7 ≈ 200 per GPU`
- At B=9 (max_concurrency=256, current): r ≈ 2.5K tok/s × η ≈ 950 tok/s/GPU (matches measured 960)
- At B=36 (max_concurrency=1024): r ≈ 9K × η ≈ 3.5K tok/s/GPU (~3.7×)
- At B=192 (max_concurrency=5376 + max_num_seqs=192 cap): r ≈ 27K × η ≈ 10K tok/s/GPU

KV budget: 91 GiB free / 71.7 MiB ≈ 1300 seqs at L≈2000 → never KV-limited at max_num_seqs=192.

**Recommended**: `max_concurrency=4096` is the next serious re-run point
(`B≈146` on 28 GPUs, below B*≈200); keep `max_num_seqs=192`.

### 4.2 OLMo-3-7B-Instruct-DPO (MHA)

- `W_bf = 13.59 GiB`, `k` at L=2000 with FP8: 0.5 GiB
- This model is KV-cache-tight AT SATURATION: 0.5 GiB × 192 seqs = 96 GiB just for KV
- Crossover batch B*: `13594 / 500 ≈ 27 per GPU` with FP8 KV and `≈14` with bf16 KV — much lower because MHA's huge KV
- At B=9 (max_concurrency=256): r similar to E4B due to similar weight size
- **Saturation analysis**: filling max_num_seqs=192 requires max_concurrency ≥ ~5376. Under
  our 1024 setting, B_actual ≈ 36/GPU, KV used ≈ 18 GiB — completely safe. The "MHA is
  the KV outlier" caution only matters if we want to push max_concurrency past ~2500.

**Recommended**: keep `max_concurrency=1024`, `max_num_seqs=192` for FP8-KV
offline evals. For bf16-KV RLVR, `1024` is already far past the weight/KV knee
(`B≈36` vs B*≈14), so increases should be justified by measured total tok/s,
not the crossover BOTEC.

### 4.3 rnj-1-instruct (Gemma3-arch, effectively full-context GQA-4)

- `W_bf = 30.959 GiB`, `k` at L=2000 with FP8: 0.122 GiB
- KV budget: 91 - 30.959 = 60.0 GiB free / 0.122 GiB = 491 seqs theoretical max at L≈2000
- Crossover batch B*: `30959 / 122 ≈ 254 per GPU`
- At B=36 (max_concurrency=1024): linear scaling regime, r ≈ 4× current

**Recommended**: `max_concurrency=3072` is conservative; if router metrics show
no queue/pathological preemption, `4096` is still below the corrected B*
(`B≈146` on 28 GPUs). Keep `max_num_seqs=192` unless a live memory trace proves
we can raise it.

### 4.4 gemma-4-26B-A4B-it (MoE, EP=4)

MoE rewrites the scaling math. Per forward, decode reads:
- Replicated weights (attention + shared FFN + embedding + router): ~5 GiB per rank
- Active expert weights (8 of 128 routed): only experts local to the rank; the rest fetched via
  all-to-all from other EP ranks. Active expert memory ≈ 1-2 GiB per token per rank.
- KV cache (bf16; FP8 not locally validated): 239 MiB/seq at L=2000

The all-to-all expert dispatch latency dominates above ~B=30-50 per GPU. Linear scaling caps earlier
than dense.

- `W_bf` per-rank loaded: ~16.163 GiB under EP=4; active top-8 weights are
  ~8.187 GiB before communication effects.
- KV budget: 91 - 16.163 = 74.8 GiB / 0.239 GiB = ~313 seqs theoretical
- Dense-memory crossover using loaded EP-rank weight: `16.163 / 0.239 ≈ 69`
  per GPU, but all-to-all comm can cap practical batch earlier.

**Recommended**: `max_concurrency=1536` is close to the corrected dense-memory
knee (`B≈55` on 28 GPUs vs B*≈69) while leaving room for MoE all-to-all
overhead. Keep `max_num_seqs=192`; enable `ep_weight_filter=true` if available
to reduce load time.

---

## 5. The other knobs (vLLM 0.20.1)

Ordered roughly by impact on offline batch inference, dense+MoE on Hopper:

### High impact

- **`max_concurrency` (client side)**: the primary lever — controls effective batch per GPU.
  Sized per the BOTEC above. Currently 1024 for dense, 768 for MoE.
- **`max_num_seqs`**: per-engine concurrency cap inside vLLM. Sized by KV budget per § 3.3.
- **`kv_cache_dtype = "fp8"`**: 2× KV capacity when supported. Blocked on Gemma 4 (head_dim=512).
- **`enable_prefix_caching = true`**: 40 rollouts/problem share input → 39 cache hits/problem,
  saves ~5.85M prefill tokens across 1000 problems.
- **`performance_mode = "throughput"`**: doubles default seqs/tokens budgets, enables FlashInfer
  autotune. Always on for these runs.
- **`enable_expert_parallel = true`** (MoE only): shards 128 experts across the 4 DP ranks per
  node. Without this, experts replicate and 26B MoE won't fit per GH200.
- **`enable_ep_weight_filter = true`** (MoE only): skip loading non-local expert weights at init.
  Saves ~10-15 min on cold model load with EP=4.

### Medium impact

- **`max_num_batched_tokens = 65536`**: prefill+decode budget per step. With ~150 input tokens
  and decode-heavy steady state, 65536 is plenty. Larger values hurt latency more than throughput
  for our workload.
- **`stream_interval = 10`**: reduces SSE callback overhead. Could push to 50+ for offline batch.
- **`gpu_memory_utilization = 0.95`**: already at ceiling on GH200.
- **`enforce_eager = false`** (CUDA graphs on): ~10-20% decode speedup. Always keep off.
- **`compilation_config.fast_moe_cold_start`** (MoE only): auto-on usually.

### Low/no impact for our workload

- **`disable_cascade_attn`**: would help if input prompts > 256 tokens with shared prefix.
  Our inputs are ~143 tokens median; skip.
- **`enable_eplb`** (Expert Parallel Load Balancing): runtime expert reorganization. Overhead not
  justified for one-shot batch inference. Skip.
- **`speculative_config`**: would need a public draft model (none for Gemma 4 or rnj-1). Skip.
- **`block_size`**: 16 default is fine for our sequence lengths (~1650 tokens ≈ 103 blocks).

### MoE backend choice (`moe_backend`)

10 options exist in vLLM 0.20.1:
`auto | triton | deep_gemm | deep_gemm_mega_moe | cutlass | flashinfer_trtllm | flashinfer_cutlass | flashinfer_cutedsl | marlin | aiter`

For Gemma-4-26B-A4B-it on Hopper (bf16 unquantized, EP=4):
- `deep_gemm` family ❌ requires FP8 quantization
- `marlin` ❌ INT4-only
- `aiter` ❌ AMD ROCm
- `cutlass` / `flashinfer_cutlass` / `flashinfer_trtllm` / `triton` — all applicable

Leaving `moe_backend = "auto"` lets vLLM choose. If decode throughput on 26B-A4B falls noticeably
short of BOTEC for its batch size, try forcing `cutlass` or `flashinfer_cutlass` explicitly and
A/B against auto.

---

## 6. Using this doc for a new model

To size knobs for a new model on the same hardware:

1. **Extract architecture inputs** (L, H, Hq, Hkv, d, window, layer pattern) from its `config.json`.
   Pull from `$HF_HUB_CACHE/models--<org>--<name>/snapshots/*/config.json`.

2. **Compute W_bf** (bf16 weight memory in GiB) from `model.safetensors.index.json` `total_size`
   field, divided by 2 (since safetensors total_size is in fp32-equivalent bytes if the model is
   stored bf16 ... actually it's the raw byte count; check by listing the snapshot files).

3. **Compute `k`** (KV bytes per token per decoded sequence position) using § 3.1's piecewise
   formula. Decide which KV dtype: FP8 only if head_dim ≤ 256 AND model isn't Gemma 4.

4. **Compute crossover B*** = `W_bf / k`. This tells you where memory-bandwidth scaling caps.

5. **Compute KV budget B_max,KV** per § 3.3 with your `gpu_memory_utilization` and `max_model_len`.
   This caps `max_num_seqs`.

6. **Set `max_num_seqs`** = min(your KV budget headroom, ~256, model-specific cap from prior runs).
   For MoE: typically don't bump past 192 because per-rank weight memory dominates.

7. **Set `max_concurrency`** ≈ `B* × N_GPU`. For a 28-GPU cluster and B*≈100, that's ~2800. We use
   1024 as a safer first-step that should already give 3-4× over the prime-rl default of 256.

8. **Measure on a 50×5 smoke** before the full run. If observed `r(B)` is within 30-40% of BOTEC's
   prediction, the run is healthy. If it's wildly off, look at:
   - Per-node load balance (any node lagging? router routing OK?)
   - Aborted-rollout count (any backend dropping requests?)
   - vLLM logs for preemption/swap events (signal that KV is over-committed)

---

## 7. Calibration checkpoints

| Date | Run | Config | Observed | Predicted | Notes |
|---|---|---|---|---|---|
| 2026-05-13 | gemma-4-E4B-it 7-node smoke | max_conc=256, max_num_seqs=192 | 311s wall, 960 tok/s/GPU at B=9 | BOTEC: 2.5K tok/s @ B=9 × η=0.38 → 950 | ✓ within 1% |
| 2026-05-13 | gemma-4-E4B-it 7-node FULL (in flight) | same | 806 rollouts/min, B≈9 | same | ✓ same regime |
| TBD | rnj-1-instruct FULL | max_conc=1024, max_num_seqs=256 | TBD | BOTEC: ~3.5K tok/s @ B=36 × η=0.38 → ~1.3K (per GPU) | watch for 3-4× over current pace |
| TBD | gemma-4-26B-A4B-it FULL | max_conc=768 | TBD | smaller speedup; expect ~2× |

When you re-run, append a row here. If the predictions stay within ±30% of observation, the
parametric model is good. If they diverge, the model needs amendment (different η, different
crossover behavior, different all-to-all cost for MoE).

---

## 8. References

- `fp8_kv_gemma4_blocked.md` — older empirical failure note; do not treat as
  proof that current local vLLM cannot run Gemma4 FP8 KV
- `isambard_srun_pty_anti_pattern.md` — why head node is excluded from inference
- `per_model_eos_audit.md` — Llama-3 dual-EOS idiom and generation_config="auto" vs "vllm"
- `primerl_perf_knobs_gotchas.md` — earlier knob landmines (compile, bf16-reduce, impl=custom)
- vLLM 0.20.1 source: `.venv/lib/python3.12/site-packages/vllm/`
  - `config/{scheduler,cache,vllm,compilation,kernel,model,attention,parallel}.py` — engine args
  - `model_executor/layers/fused_moe/` — MoE backend implementations
