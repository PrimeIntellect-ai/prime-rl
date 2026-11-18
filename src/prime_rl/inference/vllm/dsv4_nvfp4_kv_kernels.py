"""Triton kernels for DeepSeek V4's compressed KV cache stored as NVFP4.

Record (320 B per compressed token, token-major within a page):
  [0, 224)    448 NoPE values as e2m1, two per byte (low nibble = even element)
  [224, 252)  28 e4m3 scales, one per 16 NoPE values
  [252, 256)  zero padding
  [256, 320)  64 RoPE values as e4m3, unscaled

FlashMLA's sparse decode reads only fp8 records, so decode converts the selected records into
DeepSeek V4's 584 B record (448 fp8 NoPE + 64 bf16 RoPE, then a page's ue8m0 scale rows).
"""

import torch
import triton
import triton.language as tl

NVFP4_RECORD_BYTES = 320
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
NVFP4_GROUP = 16
NOPE_BYTES = NOPE_DIM // 2
SCALE_OFFSET = NOPE_BYTES
ROPE_OFFSET = 256

# DeepSeek V4 fp8 record, as FlashMLA reads it.
FP8_TOKEN_BYTES = 576  # 448 fp8 NoPE + 64 bf16 RoPE
FP8_SCALE_BYTES = 8  # 7 ue8m0 scales + 1 pad
FP8_RECORD_BYTES = FP8_TOKEN_BYTES + FP8_SCALE_BYTES
FP8_QUANT_GROUP = 64
FLASHMLA_PAGE_ALIGNMENT = 576


@triton.jit
def _e2m1_code(x):
    # Magnitude code of |x| (already divided by its scale) on the e2m1 grid, round half to even.
    a = tl.abs(x)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return code | ((x < 0).to(tl.int32) << 3)


@triton.jit
def _e2m1_value(nibble):
    exponent = (nibble >> 1) & 3
    mantissa = (nibble & 1).to(tl.float32)
    magnitude = tl.where(exponent == 0, mantissa * 0.5, tl.exp2((exponent - 1).to(tl.float32)) * (1.0 + 0.5 * mantissa))
    return tl.where((nibble & 8) != 0, -magnitude, magnitude)


@triton.jit
def _store_nvfp4_record(record_ptr, x, NOPE: tl.constexpr, GROUP: tl.constexpr, ROPE_OFF: tl.constexpr):
    """Quantize one 512-wide fp32 row (NoPE first, then RoPE) into a 320 B record."""
    WIDTH: tl.constexpr = 512
    groups = tl.reshape(x, (WIDTH // GROUP, GROUP))
    amax = tl.max(tl.abs(groups), axis=1)
    scale = tl.clamp(amax / 6.0, 0.001953125, 448.0).to(tl.float8e4nv)
    scale_f = scale.to(tl.float32)
    codes = _e2m1_code(groups / scale_f[:, None])
    codes = tl.reshape(codes, (WIDTH // 2, 2))
    lo, hi = tl.split(codes)
    packed = (lo | (hi << 4)).to(tl.uint8)
    byte = tl.arange(0, WIDTH // 2)
    tl.store(record_ptr + byte, packed, mask=byte < NOPE // 2)
    group = tl.arange(0, WIDTH // GROUP)
    tl.store(record_ptr + NOPE // 2 + group, scale.to(tl.uint8, bitcast=True), mask=group < NOPE // GROUP)
    pad = tl.arange(0, 4)
    tl.store(record_ptr + NOPE // 2 + NOPE // GROUP + pad, tl.zeros((4,), dtype=tl.uint8))
    col = tl.arange(0, WIDTH)
    rope = tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(record_ptr + ROPE_OFF + (col - NOPE), rope, mask=col >= NOPE)


@triton.jit
def _load_nvfp4_record(record_ptr, NOPE: tl.constexpr, GROUP: tl.constexpr, ROPE_OFF: tl.constexpr):
    """Dequantize a 320 B record into a 512-wide fp32 row."""
    WIDTH: tl.constexpr = 512
    col = tl.arange(0, WIDTH)
    is_nope = col < NOPE
    byte = tl.load(record_ptr + col // 2, mask=is_nope, other=0).to(tl.int32)
    nibble = tl.where((col % 2) == 0, byte & 15, byte >> 4)
    scale = tl.load(record_ptr + NOPE // 2 + col // GROUP, mask=is_nope, other=0)
    scale_f = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    rope = tl.load(record_ptr + ROPE_OFF + (col - NOPE), mask=~is_nope, other=0)
    rope_f = rope.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return tl.where(is_nope, _e2m1_value(nibble) * scale_f, rope_f)


@triton.jit
def _compress_norm_rope_store_nvfp4_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    KV_BLOCK_STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    GROUP: tl.constexpr,
    ROPE_OFF: tl.constexpr,
):
    """vLLM's head-512 compress -> RMSNorm -> GPT-J RoPE kernel, storing an NVFP4 record."""
    token_idx = tl.program_id(0)
    if tl.load(slot_mapping_ptr + token_idx) < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    block_numbers = tl.load(block_table_ptr + req_idx * block_table_stride + pos // block_size, mask=mask_pos, other=0)
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + (pos % block_size) * state_cache_stride1
        + head_offset
    )
    col = tl.arange(0, HEAD_SIZE)
    row_mask = mask_pos[:, None] & (col[None, :] < HEAD_SIZE)
    score = tl.load(row_base[:, None] + STATE_WIDTH + col[None, :], mask=row_mask, other=float("-inf"))
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + col[None, :], mask=row_mask, other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + col)
    rrms = tl.rsqrt(tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    NOPE: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    even, odd = tl.split(tl.reshape(normed, (HEAD_SIZE // 2, 2)))
    rope_pair = tl.arange(0, HEAD_SIZE // 2) - NOPE // 2
    is_rope_pair = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cache_base = cos_sin_cache_ptr + (position // COMPRESS_RATIO) * COMPRESS_RATIO * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    roped = tl.interleave(even * cos_v - odd * sin_v, odd * cos_v + even * sin_v)
    # NoPE is quantized from its bf16 rounding, as vLLM's fp8 writer does.
    row = tl.where(col < NOPE, normed.to(tl.bfloat16).to(tl.float32), roped)

    record_ptr = (
        k_cache_ptr
        + (kv_slot_idx // kv_cache_block_size).to(tl.int64) * KV_BLOCK_STRIDE
        + (kv_slot_idx % kv_cache_block_size) * RECORD_BYTES
    )
    _store_nvfp4_record(record_ptr, row, NOPE, GROUP, ROPE_OFF)


def compress_norm_rope_store_nvfp4(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    **_,
) -> None:
    assert head_dim == HEAD_DIM and rope_head_dim == ROPE_DIM
    _compress_norm_rope_store_nvfp4_kernel[(num_actual,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        HEAD_SIZE=head_dim,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        RECORD_BYTES=NVFP4_RECORD_BYTES,
        GROUP=NVFP4_GROUP,
        ROPE_OFF=ROPE_OFFSET,
        num_warps=4,
    )


@triton.jit
def _gather_nvfp4_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    offset,
    cache_block_size,
    KV_BLOCK_STRIDE: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    NOPE: tl.constexpr,
    GROUP: tl.constexpr,
    ROPE_OFF: tl.constexpr,
):
    req = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + req)
    col = tl.arange(0, 512)
    for i in range(tl.program_id(1), seq_len, tl.num_programs(1)):
        block = tl.load(block_table_ptr + req * block_table_stride + i // cache_block_size)
        record_ptr = k_cache_ptr + block.to(tl.int64) * KV_BLOCK_STRIDE + (i % cache_block_size) * RECORD_BYTES
        row = _load_nvfp4_record(record_ptr, NOPE, GROUP, ROPE_OFF)
        tl.store(out_ptr + req * out_stride0 + (offset + i) * out_stride1 + col, row.to(tl.bfloat16))


def gather_nvfp4_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Dequantize every compressed token of each request into ``out[req, offset + i]`` (bf16)."""
    assert out.stride(2) == 1 and out.shape[2] == HEAD_DIM
    _gather_nvfp4_k_cache_kernel[(seq_lens.shape[0], 128)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        block_table.stride(0),
        offset,
        block_size,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        RECORD_BYTES=NVFP4_RECORD_BYTES,
        NOPE=NOPE_DIM,
        GROUP=NVFP4_GROUP,
        ROPE_OFF=ROPE_OFFSET,
        num_warps=4,
    )


@triton.jit
def _nvfp4_to_fp8_records_kernel(
    indices_ptr,
    lens_ptr,
    k_cache_ptr,
    out_cache_ptr,
    out_indices_ptr,
    top_k,
    cache_block_size,
    KV_BLOCK_STRIDE: tl.constexpr,
    OUT_PAGE_TOKENS: tl.constexpr,
    OUT_PAGE_STRIDE: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    NOPE: tl.constexpr,
    GROUP: tl.constexpr,
    ROPE_OFF: tl.constexpr,
    FP8_TOKEN: tl.constexpr,
    FP8_SCALE: tl.constexpr,
    FP8_GROUP: tl.constexpr,
):
    token = tl.program_id(0)
    k = tl.program_id(1)
    slot = tl.load(indices_ptr + token * top_k + k)
    j = token * top_k + k
    if (k >= tl.load(lens_ptr + token)) | (slot < 0):
        tl.store(out_indices_ptr + j, -1)
        return
    record_ptr = k_cache_ptr + (slot // cache_block_size).to(tl.int64) * KV_BLOCK_STRIDE
    record_ptr += (slot % cache_block_size) * RECORD_BYTES
    row = _load_nvfp4_record(record_ptr, NOPE, GROUP, ROPE_OFF)

    page_ptr = out_cache_ptr + (j // OUT_PAGE_TOKENS).to(tl.int64) * OUT_PAGE_STRIDE
    pos = j % OUT_PAGE_TOKENS
    data_ptr = page_ptr + pos * FP8_TOKEN
    scale_ptr = page_ptr + OUT_PAGE_TOKENS * FP8_TOKEN + pos * FP8_SCALE

    WIDTH: tl.constexpr = 512
    groups = tl.reshape(row, (WIDTH // FP8_GROUP, FP8_GROUP))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1e-4)
    exponent = tl.ceil(tl.log2(amax / 448.0))
    fp8 = tl.clamp(groups * tl.exp2(-exponent)[:, None], -448.0, 448.0).to(tl.float8e4nv)
    fp8 = tl.reshape(fp8.to(tl.uint8, bitcast=True), (WIDTH,))
    col = tl.arange(0, WIDTH)
    tl.store(data_ptr + col, fp8, mask=col < NOPE)
    group = tl.arange(0, WIDTH // FP8_GROUP)
    encoded = tl.where(group < NOPE // FP8_GROUP, tl.clamp(exponent + 127.0, 0.0, 255.0), 0.0)
    tl.store(scale_ptr + group, encoded.to(tl.uint8))
    rope_ptr = (data_ptr + NOPE).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + (col - NOPE), row.to(tl.bfloat16), mask=col >= NOPE)
    tl.store(out_indices_ptr + j, j)


def fp8_record_pages(num_records: int) -> tuple[int, int]:
    """(pages, page stride in bytes) of a DeepSeek V4 fp8 cache holding ``num_records`` 64-token pages."""
    page_tokens = 64
    stride = triton.cdiv(page_tokens * FP8_RECORD_BYTES, FLASHMLA_PAGE_ALIGNMENT) * FLASHMLA_PAGE_ALIGNMENT
    return triton.cdiv(num_records, page_tokens), stride


def nvfp4_to_fp8_records(
    indices: torch.Tensor,
    lens: torch.Tensor,
    k_cache: torch.Tensor,
    block_size: int,
    workspace: torch.Tensor,
    out_indices: torch.Tensor,
) -> torch.Tensor:
    """Copy each query's selected NVFP4 records into a DeepSeek V4 fp8 cache, slot ``q * top_k + k``.

    ``indices`` holds global cache slots ``[num_tokens, top_k]``; entries at or past ``lens`` are
    skipped and come out as -1 in ``out_indices``. Returns the fp8 cache view for FlashMLA's
    ``extra_k_cache``: ``[pages, 64, 1, 584]`` uint8 over ``workspace``.
    """
    num_tokens, top_k = indices.shape
    pages, page_stride = fp8_record_pages(num_tokens * top_k)
    assert workspace.numel() >= pages * page_stride
    out_cache = workspace.as_strided(
        (pages, 64, 1, FP8_RECORD_BYTES), (page_stride, FP8_RECORD_BYTES, FP8_RECORD_BYTES, 1)
    )
    if num_tokens:
        _nvfp4_to_fp8_records_kernel[(num_tokens, top_k)](
            indices,
            lens,
            k_cache,
            workspace,
            out_indices,
            top_k,
            block_size,
            KV_BLOCK_STRIDE=k_cache.stride(0),
            OUT_PAGE_TOKENS=64,
            OUT_PAGE_STRIDE=page_stride,
            RECORD_BYTES=NVFP4_RECORD_BYTES,
            NOPE=NOPE_DIM,
            GROUP=NVFP4_GROUP,
            ROPE_OFF=ROPE_OFFSET,
            FP8_TOKEN=FP8_TOKEN_BYTES,
            FP8_SCALE=FP8_SCALE_BYTES,
            FP8_GROUP=FP8_QUANT_GROUP,
            num_warps=4,
        )
    return out_cache
