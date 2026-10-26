"""FP8 indexer for DSA sparse token selection: DeepGEMM scoring + top-k.

Scoring formula: I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

FP8 quantization vendored from vLLM (Apache 2.0).
"""

import functools
import importlib.util

import torch
import triton
import triton.language as tl

from prime_rl.utils.logger import get_logger

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10
# 256 fp32 logits = the 1 KiB row-stride alignment DeepSelect requires.
KEY_ALIGNMENT = 256
LOGITS_CHUNK_BYTES = 1 << 30
MIN_CHUNK_ROWS = 512
DEEP_SELECT_MAX_TOPK = 4096


@functools.cache
def _deep_select_supported(device: torch.device) -> bool:
    """DeepSelect is built for sm_90a, sm_100a and sm_103a only."""
    capability = torch.cuda.get_device_capability(device)
    supported = importlib.util.find_spec("deep_select") is not None and capability in ((9, 0), (10, 0), (10, 3))
    get_logger().info(f"DSA indexer top-k uses {'DeepSelect' if supported else 'torch.topk'}")
    return supported


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, use_ue8m0: bool = True, eps: float = FP8_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1

    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[-1],
        x.stride(-2),
        eps,
        FP8_MIN,
        FP8_MAX,
        use_ue8m0=use_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s


def select_topk(logits: torch.Tensor, ke: torch.Tensor, topk: int, sentinel: int) -> torch.Tensor:
    """Indices of the `topk` largest logits in each row, in no particular order.

    Logits at and past a row's end `ke` must be -inf. Slots that can only hold an -inf logit are either
    `sentinel` or an index outside the row's range, so the caller masks by range. Uses DeepSelect where it
    is built for the GPU, `torch.topk` elsewhere; the two agree except on which of several tied logits
    fills the last slots.
    """
    n_rows, n_cols = logits.shape
    actual_topk = min(topk, n_cols)
    if 0 < actual_topk <= DEEP_SELECT_MAX_TOPK and _deep_select_supported(logits.device):
        import deep_select

        _, indices = deep_select.topk(
            logits,
            actual_topk,
            end=ke,
            indices_type=torch.int32,
            idx_oob_fill_value=sentinel,
            return_value=False,
            abort_when_nan_found=False,
        )
    else:
        _, indices = torch.topk(logits, actual_topk, dim=-1)
    if actual_topk < topk:
        padding = torch.full((n_rows, topk - actual_topk), sentinel, dtype=indices.dtype, device=logits.device)
        indices = torch.cat([indices, padding], dim=-1)
    return indices


@torch.library.custom_op("prime_rl::fp8_indexer", mutates_args=())
def fp8_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    weight_scale: float = 1.0,
    amax_floor: float = FP8_EPS,
) -> torch.Tensor:
    """FP8 indexer: UE8M0 quantization + DeepGEMM `fp8_mqa_logits` scoring + topk.

    Args:
        q: [S_q, H, D] bf16 query vectors per head
        k: [S_k, D] bf16 key vectors (shared across heads)
        w: [S_q, H] bf16 per-head weights
        ks: [S_q] int32 sequence start per token (in K's coordinate system)
        ke: [S_q] int32 causal end per token (in K's coordinate system, = global position + 1)
        topk: number of top indices to return
        weight_scale: constant scaling factor for weights
        amax_floor: lower bound on each query head's and key's absolute maximum before its UE8M0 scale is taken

    Returns:
        [S_q, topk] int32 selected token indices per query (sentinel = S_k)
    """
    # NOTE: We don't use weight scale in this kernel as it produces higher KL mismatch for some reason
    # This is not a problem result-wise, as it is a constant multiplier
    S_q, H, D = q.shape
    S_k = k.shape[0]
    device = q.device

    q_flat = q.reshape(S_q * H, D).contiguous()
    q_fp8, q_scales = per_token_group_quant_fp8(q_flat, group_size=D, use_ue8m0=True, eps=amax_floor)
    S_k_aligned = triton.cdiv(max(S_k, 1), KEY_ALIGNMENT) * KEY_ALIGNMENT
    k = torch.nn.functional.pad(k, (0, 0, 0, S_k_aligned - S_k))
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True, eps=amax_floor)

    import deep_gemm

    q_fp8 = q_fp8.view(S_q, H, D)
    k_scales = k_scales.view(S_k_aligned)

    # Fold q_scale into weights
    q_scales = q_scales.view(S_q, H)
    w = w * q_scales

    indices = torch.empty((S_q, topk), dtype=torch.int32, device=device)
    chunk_rows = max(MIN_CHUNK_ROWS, LOGITS_CHUNK_BYTES // (4 * S_k_aligned))
    for start in range(0, S_q, chunk_rows):
        end = min(start + chunk_rows, S_q)
        logits = deep_gemm.fp8_mqa_logits(
            q_fp8[start:end], (k_fp8, k_scales), w[start:end], ks[start:end], ke[start:end], clean_logits=True
        )
        chunk_indices = select_topk(logits, ke[start:end], topk, S_k)
        del logits
        # Replace cross-sequence indices with sentinel S_k (maps to zero-valued KV)
        out_of_range = (chunk_indices < ks[start:end, None]) | (chunk_indices >= ke[start:end, None])
        indices[start:end] = chunk_indices.masked_fill_(out_of_range, S_k)

    return indices


@fp8_indexer.register_fake
def _fp8_indexer_fake(q, k, w, ks, ke, topk, weight_scale=1.0, amax_floor=FP8_EPS):
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)
