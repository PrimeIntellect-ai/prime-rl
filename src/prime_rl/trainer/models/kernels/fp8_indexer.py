"""FP8 indexer for DSA sparse token selection: DeepGEMM scoring + top-k.

Scoring formula: I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

FP8 quantization vendored from vLLM (Apache 2.0). With `mxfp4`, the queries and keys are rounded to vLLM's MXFP4
indexer cache instead (`mxfp4_indexer`) and their decoded values, which have two significant bits, are scored in fp32,
where every product is exact.
"""

import functools
import importlib.util

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from prime_rl.trainer.models.kernels.mxfp4_indexer import mxfp4_round
from prime_rl.utils.logger import get_logger

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10
# 256 fp32 logits = the 1 KiB row-stride alignment DeepSelect requires.
KEY_ALIGNMENT = 256
LOGITS_CHUNK_BYTES = 1 << 30
MIN_CHUNK_ROWS = 512
DEEP_SELECT_MAX_TOPK = 4096
# Four packed UE8M0 bytes encoding 1.0.
UNIT_UE8M0_X4 = 0x7F7F7F7F
# Per-row cost: scored keys + TOPK_COST_PER_KEY * (top-k scan of the scored keys + -inf fill of the row), fitted on H200.
TOPK_COST_PER_KEY = 0.13


@functools.cache
def _deep_select_supported(device: torch.device) -> bool:
    """DeepSelect is built for sm_90a, sm_100a and sm_103a only."""
    if device.type != "cuda":
        return False
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
    cp_group_name: str | None = None,
    mxfp4: bool = False,
) -> torch.Tensor:
    """FP8 indexer: UE8M0 quantization + DeepGEMM `fp8_fp4_mqa_logits` scoring + topk.

    Args:
        q: [S_q, H, D] bf16 query vectors per head
        k: [S_k, D] bf16 key vectors (shared across heads)
        w: [S_q, H] bf16 per-head weights
        ks: [S_q] int32 sequence start per token (in K's coordinate system)
        ke: [S_q] int32 causal end per token (in K's coordinate system, = global position + 1)
        topk: number of top indices to return
        weight_scale: constant scaling factor for weights
        amax_floor: lower bound on each query head's and key's absolute maximum before its UE8M0 scale is taken
        cp_group_name: contiguous CP group; rank r and rank P-1-r split their causal scoring work, same output
        mxfp4: score `q` and `k` as vLLM's MXFP4 indexer cache stores them (`amax_floor` unused); `w` should then
            already be the weights that indexer scores with (`mxfp4_indexer_weights`)

    Returns:
        [S_q, topk] int32 selected token indices per query (sentinel = S_k)
    """
    # NOTE: We don't use weight scale in this kernel as it produces higher KL mismatch for some reason
    # This is not a problem result-wise, as it is a constant multiplier
    S_q, H, D = q.shape
    S_k = k.shape[0]

    S_k_aligned = triton.cdiv(max(S_k, 1), KEY_ALIGNMENT) * KEY_ALIGNMENT
    k = torch.nn.functional.pad(k, (0, 0, 0, S_k_aligned - S_k))
    if mxfp4:
        # Decoded values (exact in fp32); the block scales live in them, so no key scales and no query scale in w
        q_fp8 = mxfp4_round(q.float())
        k_fp8 = mxfp4_round(k.float())
        k_scales = None
        w = w.float()
        # Scored against every key, per head: keep the (rows, H, S_k) scores near LOGITS_CHUNK_BYTES
        chunk_rows = max(1, LOGITS_CHUNK_BYTES // (4 * H * S_k_aligned))
    else:
        q_flat = q.reshape(S_q * H, D).contiguous()
        q_fp8, q_scales = per_token_group_quant_fp8(q_flat, group_size=D, use_ue8m0=True, eps=amax_floor)
        k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True, eps=amax_floor)

        q_fp8 = q_fp8.view(S_q, H, D)
        k_scales = k_scales.view(S_k_aligned)

        # Fold q_scale into weights
        q_scales = q_scales.view(S_q, H)
        w = w * q_scales
        chunk_rows = max(MIN_CHUNK_ROWS, LOGITS_CHUNK_BYTES // (4 * S_k_aligned))

    indices = torch.empty((S_q, topk), dtype=torch.int32, device=q.device)
    if cp_group_name is None:
        _select_topk(q_fp8, k_fp8, k_scales, w, ks, ke, S_k, chunk_rows, indices)
        return indices

    group = dist.distributed_c10d._resolve_process_group(cp_group_name)
    rank, world_size = group.rank(), group.size()
    partner = world_size - 1 - rank
    row_costs = _row_costs(ks, ke, S_k_aligned)
    chunk_costs = torch.stack([chunk.sum() for chunk in row_costs.split(chunk_rows)])
    all_chunk_costs = chunk_costs.new_empty(world_size, chunk_costs.numel())
    dist.all_gather_into_tensor(all_chunk_costs, chunk_costs, group=group)
    all_chunk_costs = all_chunk_costs.tolist()
    send_chunks, recv_chunks = cp_balance_plan(all_chunk_costs[rank], all_chunk_costs[partner])
    n_chunks = len(chunk_costs)
    keep = min(S_q, (n_chunks - send_chunks) * chunk_rows)
    recv_rows = S_q - min(S_q, (n_chunks - recv_chunks) * chunk_rows)
    peer = dist.get_global_rank(group, partner)

    if send_chunks:
        sent = [q_fp8[keep:].view(torch.uint8), w[keep:].contiguous(), ks[keep:], ke[keep:]]
        works = dist.batch_isend_irecv([dist.P2POp(dist.isend, t, peer, group) for t in sent])
    if recv_chunks:
        received = [
            torch.empty(recv_rows, H, D * q_fp8.element_size(), dtype=torch.uint8, device=q.device),
            w.new_empty(recv_rows, H),
            ks.new_empty(recv_rows),
            ke.new_empty(recv_rows),
        ]
        works = dist.batch_isend_irecv([dist.P2POp(dist.irecv, t, peer, group) for t in received])

    _select_topk(q_fp8[:keep], k_fp8, k_scales, w[:keep], ks[:keep], ke[:keep], S_k, chunk_rows, indices[:keep])

    if send_chunks:
        for work in works:
            work.wait()
        dist.irecv(indices[keep:], peer, group).wait()
    if recv_chunks:
        for work in works:
            work.wait()
        q_recv, w_recv, ks_recv, ke_recv = received
        partner_indices = torch.empty((recv_rows, topk), dtype=torch.int32, device=q.device)
        _select_topk(
            q_recv.view(q_fp8.dtype),
            k_fp8,
            k_scales,
            w_recv,
            ks_recv,
            ke_recv,
            S_k,
            chunk_rows,
            partner_indices,
        )
        dist.isend(partner_indices, peer, group).wait()

    return indices


def _row_costs(ks: torch.Tensor, ke: torch.Tensor, S_k_aligned: int) -> torch.Tensor:
    scored = (ke - ks).clamp_min(0).to(torch.float64)
    return scored + TOPK_COST_PER_KEY * (scored + S_k_aligned)


def cp_balance_plan(own_chunk_costs: list[int], partner_chunk_costs: list[int]) -> tuple[int, int]:
    """Returns how many trailing query chunks this rank sends to its partner and receives from it."""
    own, partner = sum(own_chunk_costs), sum(partner_chunk_costs)
    if own < partner:
        return 0, _trailing_chunks_to_send(partner_chunk_costs, own)
    return _trailing_chunks_to_send(own_chunk_costs, partner), 0


def _trailing_chunks_to_send(chunk_costs: list[int], partner_cost: int) -> int:
    total = sum(chunk_costs)
    best_chunks, best_cost = 0, total
    sent = 0
    for n in range(1, len(chunk_costs) + 1):
        sent += chunk_costs[-n]
        cost = max(total - sent, partner_cost + sent)
        if cost < best_cost:
            best_chunks, best_cost = n, cost
    return best_chunks


def _select_topk(q_fp8, k_fp8, k_scales, w, ks, ke, S_k, chunk_rows, indices):
    if k_scales is None:
        _select_topk_decoded(q_fp8, k_fp8, w, ks, ke, S_k, chunk_rows, indices)
        return
    from vllm.third_party import deep_gemm

    q_scales = None
    if torch.cuda.get_device_capability(q_fp8.device)[0] == 10:
        # SM100 takes MX inputs only: each key's UE8M0 scale repeated over its four 32-wide blocks, unit query
        # scales (they are folded into w) and bf16 weights, as vLLM's `mqa_logits_mx_inputs` builds them. The
        # cast is exact for bf16 head weights and rounds fp32 ones (GLM5-Next) the way vLLM does.
        k_scales = (k_scales.view(torch.int32) >> 23).to(torch.uint8)[:, None].expand(-1, 4).contiguous()
        k_scales = k_scales.view(torch.int32).squeeze(-1)
        q_scales = torch.full(q_fp8.shape[:-1], UNIT_UE8M0_X4, dtype=torch.int32, device=q_fp8.device)
        w = w.to(torch.bfloat16)
    lengths = (ke - ks).clamp_min(0)
    for start in range(0, q_fp8.shape[0], chunk_rows):
        end = min(start + chunk_rows, q_fp8.shape[0])
        # Row i holds the logits of keys [ks[i], ke[i]) from column 0; columns past ke[i] - ks[i] are not cleaned.
        logits = deep_gemm.fp8_fp4_mqa_logits(
            (q_fp8[start:end], None if q_scales is None else q_scales[start:end]),
            (k_fp8, k_scales),
            w[start:end],
            ks[start:end],
            ke[start:end],
            k_fp8.shape[0],
        ).float()
        row_lengths = lengths[start:end]
        columns = torch.arange(logits.shape[1], device=logits.device)
        logits.masked_fill_(columns >= row_lengths[:, None], float("-inf"))
        chunk_indices = select_topk(logits, row_lengths, indices.shape[1], S_k) + ks[start:end, None]
        del logits
        # Replace cross-sequence indices with sentinel S_k (maps to zero-valued KV)
        out_of_range = (chunk_indices < ks[start:end, None]) | (chunk_indices >= ke[start:end, None])
        indices[start:end] = chunk_indices.masked_fill_(out_of_range, S_k)


def _select_topk_decoded(q, k, w, ks, ke, S_k, chunk_rows, indices):
    """`_select_topk` on decoded fp32 queries `[S_q, H, D]` and keys `[S_k_aligned, D]`: each query's scores against
    every key, with the keys outside `[ks, ke)` at -inf."""
    columns = torch.arange(k.shape[0], device=k.device)
    for start in range(0, q.shape[0], chunk_rows):
        end = min(start + chunk_rows, q.shape[0])
        scores = torch.relu(torch.matmul(q[start:end], k.T))  # (rows, H, S_k_aligned)
        logits = torch.einsum("rhk,rh->rk", scores, w[start:end])
        del scores
        row_ks, row_ke = ks[start:end, None], ke[start:end, None]
        logits.masked_fill_((columns < row_ks) | (columns >= row_ke), float("-inf"))
        chunk_indices = select_topk(logits, ke[start:end], indices.shape[1], S_k)
        del logits
        out_of_range = (chunk_indices < row_ks) | (chunk_indices >= row_ke)
        indices[start:end] = chunk_indices.masked_fill_(out_of_range, S_k)


@fp8_indexer.register_fake
def _fp8_indexer_fake(q, k, w, ks, ke, topk, weight_scale=1.0, amax_floor=FP8_EPS, cp_group_name=None, mxfp4=False):
    return q.new_empty((q.shape[0], topk), dtype=torch.int32)
