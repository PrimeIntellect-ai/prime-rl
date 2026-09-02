# What Megatron-LM already has for DeepSeek V4

Surveyed against Megatron-LM `dev` at `/home/garrett/github/NVIDIA/Megatron-LM`, HEAD `24fc94d27`
(verified). Every path below was confirmed to exist. The DS V4 stack lives under
`megatron/core/transformer/experimental_attention_variant/`.

## The structural fact that shapes everything else

Megatron has **two** sparse-attention families and they do not share a kernel backend.

- **DSA** (`dsa.py`, DeepSeek-3.2 style) dispatches through `dsa_kernels.py` over
  `none` / `tilelang` / `cudnn`.
- **DS V4 hybrid CSA/HCA** (`csa.py`, `deepseek_v4_hybrid_attention.py`) does **not** use that
  dispatch. It reads only the boolean `use_fused_dsa_kernels` and then calls cuDNN-Frontend and
  FlashMLA directly. `transformer_config.py:1911-1915` rejects TileLang outright:

  > `dsv4_hybrid does not support dsa_kernel_backend='tilelang'; use 'cudnn' for fused CSA kernels or 'none' for the PyTorch fallback.`

So for DS V4 the choice in Megatron is **cuDNN-Frontend `develop` plus FlashMLA `nv_dev`, or
eager PyTorch**. There is no TileLang CSA path to lift.

## Component inventory

| Component | What it is | Runs on sm90? | Portability |
|---|---|---|---|
| `csa.py` (3056 lines) | `Compressor`, `CSAIndexer`, `CompressedSparseAttention`, with SBHD / THD / THD+CP paths | eager path yes | **Hard** as classes, **Easy** for the helpers |
| `deepseek_v4_hybrid_attention.py` | `DSv4HybridSelfAttention(Attention)`, per-layer YaRN-vs-plain rope split | yes | **Hard**, and redundant |
| `absorbed_mla.py` | Absorbed MLA for the DSA family | yes | **Hard**, and off the DS V4 path |
| `dsa_kernels.py` | Backend-neutral dispatch shim, "return `None` means fall back to eager" | n/a | **Easy** pattern, irrelevant code |
| `csa_utils/fused_sparse_attention.py` (2452 lines) | FlashMLA forward + cuDNN DSA backward, three integration paths | yes, with an sm90 carve-out | **Hard**: both kernel providers unobtainable |
| `csa_utils/fused_compressor.py` | Dispatch shim onto `cudnn.csa.compressor` | **no, Blackwell only** | do not port |
| `csa_utils/csa_teacher_lse.py` (493 lines) | Three Triton kernels for the indexer KL loss teacher denominator | **yes** | **Easy**, zero Megatron imports |
| `fusions/fused_mla_yarn_rope_apply.py` (1283 lines) | Six autotuned Triton RoPE kernels, in-place, with an `inverse=True` mode | **yes** | **Moderate**: interleaving mismatch |
| `fusions/fused_mhc_kernels.py` (3129 lines) | `fused_sinkhorn`, `fused_h_aggregate`, `fused_h_post_bda` (Triton); `fused_proj_rms_compute_h` (cuTile only) | Triton ops yes, cuTile ops effectively no | **Easy** to **Moderate** |

## sm90 status, and how it was determined

`fused_compressor.py` is hard-gated to Blackwell by an **equality** test:

```python
_SUPPORTED_COMPUTE_CAPABILITY = (10, 0)              # fused_compressor.py:74
supported = torch.cuda.get_device_capability(index) == _SUPPORTED_COMPUTE_CAPABILITY   # :146
```

On H200 it returns `False` before the frontend is probed, and `Compressor._forward_thd` silently
stays eager. Its 62-line module docstring is still worth reading: it is a precise numerics
contract for what a prime-rl-native fused compressor would have to compute.

The cuTile kernels are gated by an external compiler probe, not an arch list:
`_cutile_supports_current_device` (`fused_mhc_kernels.py:168-207`) shells out to
`tileiras --gpu-name sm_{major}{minor}` and fails closed when the binary is absent. Because
`fused_proj_rms_compute_h` has **no Triton variant**, the largest mHC op falls back to
`@torch.compile` on any normal H200 install.

The Triton files carry no capability checks at all: `csa_teacher_lse.py`,
`fused_mla_yarn_rope_apply.py`, and the Triton half of `fused_mhc_kernels.py` all JIT for the
running device. `fused_sparse_attention.py` explicitly handles sm90
(`_get_topk_alignment` pads top-k to 128 on sm90 versus 64 on sm100+), and
`transformer_config.py:1919-1923` asserts `sm >= 9`, with one carve-out at `:1924-1936`: on sm90,
ratio-4 indexer plus **dense** indexer loss is rejected because "the cuDNN Frontend SM90 dense DSA
kernels are not reliable for this path". Sparse indexer loss on sm90 is allowed.

## The two hard constraints

**1. The cuDNN backend's dependencies are not on PyPI.** Confirmed at the import sites:

```python
from cudnn import DSA                    # dsa_cudnn_kernels.py:933, fused_sparse_attention.py:188
from cudnn.csa import compressor         # fused_compressor.py:113
from flash_mla import flash_mla_sparse_fwd   # fused_sparse_attention.py:101
```

Megatron's own `pyproject.toml` marks both as unpackaged:

```toml
no_pypi_wheels = ["flash_mla", ...]
[tool.uv.sources]
flash_mla = [{ git = "https://github.com/deepseek-ai/FlashMLA", rev = "nv_dev" }]
nvidia-cudnn-frontend = { git = "https://github.com/NVIDIA/cudnn-frontend.git", rev = "0a14b71" }
```

The GB200 recipe Dockerfile builds both from source branches (`--branch nv_dev`,
`--branch develop`). A third git-only dependency, `fast_hadamard_transform`, is required by
`CSAIndexer`'s `rotate_activation`. prime-rl's `DeepseekV4Indexer` does **not** Hadamard rotate,
but this is not a checkpoint-basis mismatch: Megatron rotates *activations*, not weights, and
nothing per-channel sits between the rotation and the score matmul, so the orthogonality of `H`
leaves the score mathematically invariant. The rotation reads as a low-precision numerics measure,
which makes the real question whether any indexer matmul is quantized. That was not checked in
either codebase.

**2. CP16 is a prerequisite for 64k, not an optimization.** The recipe is at
`examples/moe_recipes/deepseek_v4_flash/gb200/mxfp8_THD64K_128GPU_TP1PP2EP64CP16.yaml`. Diffing it
against the 4K sibling recipe isolates the change set: `seq_length 4096 -> 65536`,
`context_parallel_size 1 -> 16`, host offload settings, and `cp_partition_mode: contiguous`.
Everything else is identical, **including `max_seqlen_per_dp_cp_rank: 4096` and
`recompute_modules: [mla_up_proj]`**. Both recipes hold 4096 tokens per rank; 64k is reached
purely by `16 * 4096`.

`megatron/training/arguments.py:1786-1803` makes this an assert, not a preference:

```python
assert total_cp_ranks * args.max_seqlen_per_dp_cp_rank >= args.seq_length
```

Setting CP back to 1 at `seq_length 65536` trips it. Megatron did not choose to raise the per-rank
budget even on 192 GB GB200s with mxfp8 weights and full optimizer-state offload.

prime-rl hard-rejects CP for DS V4 at `modeling_deepseek_v4.py:241-245`, and that rejection is
*correct as written*: the sliding-window mask is dense and local, built from post-shard
boundaries, so global boundaries cannot address it. Megatron's answer is a **hidden-only
left-boundary exchange** (`cp_utils.exchange_cp_boundary_hidden`,
`cp_utils.prepare_cp_compressor_input`): each rank receives the `window_size` rows preceding its
shard and projects `boundary_kv` locally. That is the design reference if CP is ever revisited.

Also note the recipe as a whole is Blackwell-only for other reasons: `mxfp8` needs sm100 tensor
cores, `NVTE_CUDA_ARCHS="100a;103a"`, and FlashMLA built with `FLASH_MLA_DISABLE_SM90=1`.

## What prime-rl already has that overlaps

| Megatron piece | prime-rl equivalent | Verdict |
|---|---|---|
| `ops/tilelang_sparse_mla_{fwd,bwd}.py` | `models/kernels/sparse_mla_{fwd,bwd}.py` | **Same upstream fork** (tile-ai/tilelang). prime-rl's is arguably ahead: `T.dynamic(...)` shapes so one compiled kernel serves all packed lengths, wrapped as a `torch.library.custom_op` with `register_fake`. |
| `ops/indexer.py`, `ops/tilelang_indexer_*` | `models/kernels/fp8_indexer.py` | Functionally redundant; different implementations (TileLang vs Triton+fp8), same job. |
| `absorbed_mla.py` absorption math | `glm_moe_dsa/sparse_mla_attention.py:167-206` | Redundant. Megatron adds checkpoint-layout handling prime-rl lacks. |
| `Compressor` / `CSAIndexer` / `CompressedSparseAttention` | `DeepseekV4Compressor` / `Indexer` / `CSACompressor` / `HCACompressor` | Redundant. prime-rl's `PackedContext` + `CompressionLayout` is a cleaner encoding of the same per-document bookkeeping. |
| `hyper_connection.py` native ops | `DeepseekV4HyperConnection` / `HyperHead` | Redundant, and numerically identical (Sinkhorn verified line by line). |

## Config mapping

Field-by-field, prime-rl to Megatron. Flagged rows are where semantics differ, not just names.

| prime-rl | value | Megatron | note |
|---|---|---|---|
| `sliding_window` | 128 | `csa_window_size` | same, both inclusive of the query |
| `index_topk` | 512 | `dsa_indexer_topk` | same, both clamp and use `-1` for surplus |
| `index_n_heads` / `index_head_dim` | 64 / 128 | `dsa_indexer_n_heads` / `dsa_indexer_head_dim` | same |
| `o_groups` / `o_lora_rank` | 8 / 1024 | `o_groups` / `o_lora_rank` | same |
| `head_dim` | 512 | `v_head_dim` (+ `kv_channels`) | **same value, different framing**; Megatron has no unified head_dim |
| `hc_mult` | 4 | `num_residual_streams` | same meaning, different name |
| `hc_sinkhorn_iters` | 20 | `mhc_sinkhorn_iterations` | **same, including the off-by-one**: softmax, one column normalization, then `iters-1` row/column pairs |
| `compress_rates` + `layer_types` | dict + per-layer list | `csa_compress_ratios` | **structural difference**: Megatron is one flat array of length `num_layers + mtp_num_layers` (44); prime-rl's is length 43 with no MTP tail |
| `hc_eps` | 1e-6 | (none) | **granularity differs**: Megatron hardcodes three separate epsilons; prime-rl uses one for all three roles |
| `compress_rope_theta` | 160000 | `csa_compress_rotary_base` = 40000 + `rotary_scaling_factor` = 4 | **do not naively map**; Megatron's is a YaRN base with a scaling factor, prime-rl's default is a plain base |
| (none) | | `dsa_indexer_loss_coeff` = 1e-2, `dsa_indexer_use_sparse_loss` = true | **no prime-rl equivalent**; see below |
| (none) | | `dsa_indexer_rotate_activation` = true | **no prime-rl equivalent**; Megatron Hadamard-rotates indexer `q` and its compressor output |

## The gap that is not about performance

Megatron trains the Lightning Indexer with a KL divergence between the indexer's score
distribution and the attention distribution (`dsa_indexer_loss.py`, `FusedDSAIndexerLoss`, the
teacher-LSE kernels in `csa_teacher_lse.py`, coefficient `1e-2` in the recipe). **prime-rl has no
such loss**, and as a direct consequence its indexer receives no gradient at all. This was
confirmed by measurement, not inference: see `README.md`.

## Portability shortlist

1. **THD sparse-attention index algebra and the eager gather-based reference.**
   `csa.py:626` (`unfused_compressed_sparse_attn`), plus `get_window_topk_idxs_thd` (`:225`),
   `get_compress_topk_idxs_thd` (`:262`), `build_cu_seqlens_kv_full` (`:330`), `cat_per_segment`
   (`:352`), and `build_flat_topk_idxs` (`fused_sparse_attention.py:642`).
   *Why first:* it is the only thing that removes prime-rl's dense mask and dense logits, which is
   what caps context length today, and it produces exactly the flat-global index layout a sparse
   kernel consumes. **Moderate**: pure torch, but prime-rl's per-document entry coordinates have
   to be re-expressed against Megatron's per-segment `cu_seqlens`.
2. **Attention-sink support in prime-rl's own TileLang `sparse_mla` kernel.** prime-rl already
   owns the kernel and already ships TileLang as a required dependency. The changes are a sink
   term in the online-softmax denominator and relaxing `assert topk % block_I == 0`
   (`sparse_mla_fwd.py:44`). Note `128 + 512 = 640 = 10 * 64` already aligns for CSA.
   **Moderate**: kernel modification, not a port.
3. **`fused_mla_yarn_rope_apply.py`.** One Megatron import, sm90-clean, and its `inverse=True`
   mode maps onto prime-rl's conjugate de-rotation. **Moderate**: prime-rl uses adjacent-pair
   interleaving, which the kernel refuses at `:433`, and half-width `cos`/`sin`.
4. **`csa_teacher_lse.py` plus the indexer KL loss.** Closes a correctness gap, not a speed gap.
   **Easy** for the 493 lines of standalone Triton, **Moderate** for the loss plumbing.
5. **`fused_sinkhorn` from `fused_mhc_kernels.py`.** Smallest absolute win, cheapest correct one:
   semantics are verified byte-identical, so it is a drop-in with an equivalence test. **Easy**.

**Explicitly not recommended:** `fused_compressor.py` (Blackwell-only and unobtainable),
`fused_sparse_attention.py` as a whole (both kernel providers are git-branch-only), `absorbed_mla.py`
(off the DS V4 path), `ops/tilelang_sparse_mla_*` and `ops/indexer.py` (prime-rl's forks are equal
or better), the `csa.py` module classes (welded to `Attention` / `build_module` /
`ProcessGroupCollection`), and `_forward_thd_cp` plus `csa_utils/cp_*` (blocked on the CP
rejection, though they are the design reference if that changes).
