"""DeepSeek V4 with an NVFP4 compressed KV cache: ``[inference.vllm] kv_cache_dtype = "nvfp4_ds_mla"``.

vLLM 0.33 stores DeepSeek V4's KV only as fp8 records (584 B per token). With ``nvfp4_ds_mla`` the
compressed KV of the compress-ratio-4 layers, which holds nearly all of the per-token KV, is stored
as 320 B NVFP4 records instead. The sliding-window, ratio-128 and indexer caches keep their fp8
records, so the rest of the model runs vLLM's ``fp8_ds_mla`` path unchanged.

- Write: the compressor stores NVFP4 records (Triton, ``dsv4_nvfp4_kv_kernels``).
- Prefill: the compressed history is dequantized to bf16 for the sparse prefill kernel, as before.
- Decode: FlashMLA reads only fp8 records, so each query's selected records are converted into a
  per-step fp8 cache that FlashMLA reads as its extra cache.
- Memory: the shared KV block is as wide as the widest cache group. Sliding-window and compressor
  state groups wider than the smaller compressed group are split, so the block shrinks with it.
"""

import math
from collections import defaultdict
from dataclasses import replace

import torch
from vllm.utils.math_utils import cdiv

from prime_rl.inference.vllm import dsv4_nvfp4_kv_kernels as kernels

NVFP4_KV_CACHE_DTYPE = "nvfp4_ds_mla"
# A state group at most this much wider than the widest full-attention group stays whole: splitting
# it costs every request its window's blocks again for a few percent of block width.
_STATE_GROUP_SLACK = 1.05

_nvfp4_compressed_kv = False


def patch_deepseek_v4_nvfp4_kv_cache():
    _patch_kv_cache_groups()
    _patch_deepseek_v4()


def _patch_deepseek_v4():
    from vllm.models.deepseek_v4 import attention, compressor
    from vllm.models.deepseek_v4.nvidia import flashmla

    if getattr(attention._resolve_dsv4_kv_cache_dtype, "_prime_rl_nvfp4", False):
        return
    resolve = attention._resolve_dsv4_kv_cache_dtype

    def _resolve_dsv4_kv_cache_dtype(use_fp8_ds_mla_layout, kv_cache_dtype, cache_config):
        global _nvfp4_compressed_kv
        if kv_cache_dtype == NVFP4_KV_CACHE_DTYPE:
            if not use_fp8_ds_mla_layout:
                raise ValueError(f"{NVFP4_KV_CACHE_DTYPE} needs DeepSeek V4's FlashMLA attention backend (SM100).")
            _nvfp4_compressed_kv = True
            # Every cache but the compressed one keeps the fp8 records; this writes fp8_ds_mla back.
            kv_cache_dtype = "fp8"
        return resolve(use_fp8_ds_mla_layout, kv_cache_dtype, cache_config)

    _resolve_dsv4_kv_cache_dtype._prime_rl_nvfp4 = True
    attention._resolve_dsv4_kv_cache_dtype = _resolve_dsv4_kv_cache_dtype

    get_kv_cache_spec = attention.DeepseekV4Attention.get_kv_cache_spec

    def get_kv_cache_spec_nvfp4(self, vllm_config):
        spec = get_kv_cache_spec(self, vllm_config)
        if not _nvfp4_compressed_kv or self.compress_ratio != 4:
            return spec
        return replace(
            spec,
            state_content_bytes=kernels.NVFP4_RECORD_BYTES,
            page_size_padded=None,
            # Whole records of both kinds in the shared block stride.
            block_stride_alignment=math.lcm(spec.alignment, kernels.NVFP4_RECORD_BYTES),
        )

    attention.DeepseekV4Attention.get_kv_cache_spec = get_kv_cache_spec_nvfp4

    compress_norm_rope_store = compressor.DeepseekCompressor._compress_norm_rope_store

    def _compress_norm_rope_store(
        self, state_metadata, state_cache, state_width, positions, rotary_emb, attn_metadata, pdl_kwargs
    ):
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        if not _is_nvfp4_cache(kv_cache):
            return compress_norm_rope_store(
                self, state_metadata, state_cache, state_width, positions, rotary_emb, attn_metadata, pdl_kwargs
            )
        kernels.compress_norm_rope_store_nvfp4(
            state_cache=state_cache,
            num_actual=state_metadata.slot_mapping.shape[0],
            token_to_req_indices=state_metadata.token_to_req_indices,
            positions=positions,
            slot_mapping=state_metadata.slot_mapping,
            block_table=state_metadata.block_table,
            block_size=state_metadata.block_size,
            state_width=state_width,
            cos_sin_cache=rotary_emb.cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=attn_metadata[self.k_cache_prefix],
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
        )

    compressor.DeepseekCompressor._compress_norm_rope_store = _compress_norm_rope_store

    gather = flashmla.dequantize_and_gather_k_cache

    def dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, **kwargs):
        if not _is_nvfp4_cache(k_cache):
            return gather(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, **kwargs)
        assert gather_lens is None, "the compressed history is always gathered whole"
        kernels.gather_nvfp4_k_cache(out, k_cache, seq_lens, block_table, block_size, offset)

    flashmla.dequantize_and_gather_k_cache = dequantize_and_gather_k_cache

    flash_mla_with_kvcache = flashmla.flash_mla_with_kvcache

    def flash_mla_with_kvcache_nvfp4(
        *args, extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None, **kwargs
    ):
        if extra_k_cache is not None and _is_nvfp4_cache(extra_k_cache):
            num_tokens, _, top_k = extra_indices_in_kvcache.shape
            indices = extra_indices_in_kvcache.reshape(num_tokens, top_k)
            if extra_topk_length is None:
                extra_topk_length = torch.full((num_tokens,), top_k, dtype=torch.int32, device=indices.device)
            pages, page_stride = kernels.fp8_record_pages(num_tokens * top_k)
            # Allocated per call: inside a captured decode graph this comes from the graph's pool.
            workspace = torch.empty(pages * page_stride, dtype=torch.uint8, device=indices.device)
            fp8_indices = torch.empty_like(indices)
            extra_k_cache = kernels.nvfp4_to_fp8_records(
                indices,
                extra_topk_length,
                extra_k_cache.squeeze(2),
                extra_k_cache.shape[1],
                workspace,
                fp8_indices,
            )
            extra_indices_in_kvcache = fp8_indices.view(num_tokens, 1, top_k)
        return flash_mla_with_kvcache(
            *args,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            **kwargs,
        )

    flashmla.flash_mla_with_kvcache = flash_mla_with_kvcache_nvfp4


def _is_nvfp4_cache(kv_cache: torch.Tensor) -> bool:
    # NVFP4 compressed caches are the only DeepSeek V4 caches with 320 B records.
    return kv_cache.dim() >= 2 and kv_cache.shape[-1] == kernels.NVFP4_RECORD_BYTES


def _patch_kv_cache_groups():
    """Split sliding-window and compressor-state groups that are wider than the compressed KV group.

    Every block of the block-outermost pool is as wide as the widest group. With fp8 records the
    compressed group (C4 + C128 + indexer KV) is the widest; with NVFP4 records it is not, and the
    state groups would keep the block, and thus the per-token cost, nearly as wide as before.
    """
    from vllm.v1.core import kv_cache_planning
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    original = kv_cache_planning._get_packed_kv_cache_groups
    if getattr(original, "_prime_rl_nvfp4", False):
        return

    def layer_specs(group):
        return dict(group.kv_cache_spec.kv_cache_specs)

    def group_bytes(group):
        return sum(spec.page_size_bytes for spec in layer_specs(group).values())

    def is_window(group):
        return isinstance(group.kv_cache_spec.first_spec, SlidingWindowSpec)

    def _get_packed_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = original(vllm_config, kv_cache_spec)
        has_nvfp4 = any(
            isinstance(spec, MLAAttentionSpec) and spec.state_content_bytes == kernels.NVFP4_RECORD_BYTES
            for spec in kv_cache_spec.values()
        )
        if not groups or not has_nvfp4 or not all(isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) for g in groups):
            return groups
        limit = max(group_bytes(group) for group in groups if not is_window(group))

        # Re-merge the window groups vLLM split by its repeat pattern, then split each bucket into the
        # fewest groups that fit under the compressed group, in the place of the bucket's first group.
        buckets: list[dict] = []
        bucket_of = {}
        for i, group in enumerate(groups):
            if not is_window(group):
                continue
            for b, bucket in enumerate(buckets):
                if UniformTypeKVCacheSpecs.is_uniform_type({**bucket, **layer_specs(group)}):
                    bucket.update(layer_specs(group))
                    bucket_of[i] = b
                    break
            else:
                bucket_of[i] = len(buckets)
                buckets.append(layer_specs(group))

        def split(bucket):
            page_size_layers = defaultdict(list)
            for name, spec in bucket.items():
                page_size_layers[spec.page_size_bytes].append(name)
            # vLLM packs window buckets as 1:1 repeats of their page sizes.
            repeats = list(zip(*page_size_layers.values()))
            assert sum(map(len, repeats)) == len(bucket), "window bucket is not a 1:1 page-size pattern"
            repeat_bytes = sum(page_size_layers)
            if repeat_bytes * len(repeats) <= _STATE_GROUP_SLACK * limit:
                num_groups = 1
            else:
                num_groups = cdiv(len(repeats), max(limit // repeat_bytes, 1))
            for i in range(num_groups):
                names = [name for repeat in repeats[i::num_groups] for name in repeat]
                yield KVCacheGroupSpec(names, UniformTypeKVCacheSpecs.from_specs({n: bucket[n] for n in names}))

        regrouped = []
        emitted = set()
        for i, group in enumerate(groups):
            if i not in bucket_of:
                regrouped.append(group)
            elif bucket_of[i] not in emitted:
                emitted.add(bucket_of[i])
                regrouped.extend(split(buckets[bucket_of[i]]))
        return regrouped

    _get_packed_kv_cache_groups._prime_rl_nvfp4 = True
    kv_cache_planning._get_packed_kv_cache_groups = _get_packed_kv_cache_groups
