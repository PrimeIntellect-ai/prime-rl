import torch
import triton
import triton.language as tl

_DTYPE_MAP = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _fused_dispatch_gemm_kernel(
    routed_input_ptr,
    weight_ptr,
    expert_out_ptr,
    hidden_peer_ptrs,
    flag_peer_ptrs,
    local_hidden_ptr,
    local_flag_ptr,
    recv_tile_to_local_expert_ptr,
    recv_tile_valid_ptr,
    tile_peer_rank_ptr,
    tile_local_row_start_ptr,
    tile_peer_row_start_ptr,
    tile_valid_rows_ptr,
    tile_flag_index_ptr,
    n_dispatch_tiles,
    n_recv_tiles,
    HIDDEN_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    N_COMM_CTAS: tl.constexpr,
    N_COMPUTE_CTAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < N_COMM_CTAS:
        hcol = tl.arange(0, HIDDEN_DIM)
        i = pid
        while i < n_dispatch_tiles:
            dest_rank = tl.load(tile_peer_rank_ptr + i)
            if dest_rank >= 0:
                local_start = tl.load(tile_local_row_start_ptr + i).to(tl.int64)
                peer_start = tl.load(tile_peer_row_start_ptr + i).to(tl.int64)
                valid = tl.load(tile_valid_rows_ptr + i)
                flag_idx = tl.load(tile_flag_index_ptr + i)

                rows = tl.arange(0, BLOCK_M)
                row_mask = rows < valid
                vals = tl.load(
                    routed_input_ptr + (local_start + rows)[:, None] * HIDDEN_DIM + hcol[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                dest_base = tl.load(hidden_peer_ptrs + dest_rank).to(tl.pointer_type(DTYPE))
                tl.store(
                    dest_base + (peer_start + rows)[:, None] * HIDDEN_DIM + hcol[None, :],
                    vals,
                    mask=row_mask[:, None],
                )

                tl.debug_barrier()
                flag_base = tl.load(flag_peer_ptrs + dest_rank).to(tl.pointer_type(tl.int32))
                tl.atomic_xchg(flag_base + flag_idx, 1)

            i += N_COMM_CTAS
    else:
        cid = pid - N_COMM_CTAS
        hcol = tl.arange(0, HIDDEN_DIM)
        ocol = tl.arange(0, OUT_DIM)

        t = cid
        while t < n_recv_tiles:
            is_real = tl.load(recv_tile_valid_ptr + t)
            if is_real != 0:
                while tl.atomic_add(local_flag_ptr + t, 0) == 0:
                    pass

                local_expert = tl.load(recv_tile_to_local_expert_ptr + t)
                w_base = weight_ptr + local_expert.to(tl.int64) * HIDDEN_DIM * OUT_DIM
                weight = tl.load(w_base + hcol[:, None] * OUT_DIM + ocol[None, :])

                rows = t * BLOCK_M + tl.arange(0, BLOCK_M)
                hidden = tl.load(local_hidden_ptr + rows[:, None] * HIDDEN_DIM + hcol[None, :])
                result = tl.dot(hidden, weight)
                tl.store(expert_out_ptr + rows[:, None] * OUT_DIM + ocol[None, :], result.to(DTYPE))

            t += N_COMPUTE_CTAS


def fused_dispatch_gemm(
    routed_input: torch.Tensor,
    weight: torch.Tensor,
    expert_out: torch.Tensor,
    *,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    local_hidden: torch.Tensor,
    local_flag: torch.Tensor,
    recv_tile_to_local_expert: torch.Tensor,
    recv_tile_valid: torch.Tensor,
    dispatch_tiles,
    n_recv_tiles: int,
    n_comm_ctas: int,
    n_compute_ctas: int,
    block_m: int,
) -> None:
    hidden_dim = routed_input.shape[1]
    out_dim = weight.shape[-1]
    dtype = _DTYPE_MAP[routed_input.dtype]
    n_dispatch_tiles = dispatch_tiles.peer_rank.numel()

    grid = (n_comm_ctas + n_compute_ctas,)
    _fused_dispatch_gemm_kernel[grid](
        routed_input,
        weight,
        expert_out,
        hidden_peer_ptrs,
        flag_peer_ptrs,
        local_hidden,
        local_flag,
        recv_tile_to_local_expert,
        recv_tile_valid,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.peer_row_start,
        dispatch_tiles.valid_rows,
        dispatch_tiles.flag_index,
        n_dispatch_tiles,
        n_recv_tiles,
        HIDDEN_DIM=hidden_dim,
        OUT_DIM=out_dim,
        N_COMM_CTAS=n_comm_ctas,
        N_COMPUTE_CTAS=n_compute_ctas,
        BLOCK_M=block_m,
        DTYPE=dtype,
        num_warps=4,
    )


@triton.jit
def _dispatch_scatter_wait_kernel(
    routed_input_ptr,
    hidden_peer_ptrs,
    flag_peer_ptrs,
    local_flag_ptr,
    recv_tile_valid_ptr,
    tile_peer_rank_ptr,
    tile_local_row_start_ptr,
    tile_peer_row_start_ptr,
    tile_valid_rows_ptr,
    tile_flag_index_ptr,
    n_dispatch_tiles,
    n_recv_tiles,
    HIDDEN_DIM: tl.constexpr,
    N_COMM_CTAS: tl.constexpr,
    N_WAIT_CTAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Same producer (symmetric-memory scatter) role as `_fused_dispatch_gemm_kernel`, but the
    consumer role only spin-waits on arrival flags instead of running a GEMM -- for use ahead of
    an external, whole-buffer compute kernel (e.g. `flash_moe`) that needs every row of
    `local_hidden` to have landed before it can be called, since those writes arrive from peers'
    streams and are otherwise invisible to this rank's own stream ordering.
    """
    pid = tl.program_id(0)

    if pid < N_COMM_CTAS:
        hcol = tl.arange(0, HIDDEN_DIM)
        i = pid
        while i < n_dispatch_tiles:
            dest_rank = tl.load(tile_peer_rank_ptr + i)
            if dest_rank >= 0:
                local_start = tl.load(tile_local_row_start_ptr + i).to(tl.int64)
                peer_start = tl.load(tile_peer_row_start_ptr + i).to(tl.int64)
                valid = tl.load(tile_valid_rows_ptr + i)
                flag_idx = tl.load(tile_flag_index_ptr + i)

                rows = tl.arange(0, BLOCK_M)
                row_mask = rows < valid
                vals = tl.load(
                    routed_input_ptr + (local_start + rows)[:, None] * HIDDEN_DIM + hcol[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                dest_base = tl.load(hidden_peer_ptrs + dest_rank).to(tl.pointer_type(DTYPE))
                tl.store(
                    dest_base + (peer_start + rows)[:, None] * HIDDEN_DIM + hcol[None, :],
                    vals,
                    mask=row_mask[:, None],
                )

                tl.debug_barrier()
                flag_base = tl.load(flag_peer_ptrs + dest_rank).to(tl.pointer_type(tl.int32))
                tl.atomic_xchg(flag_base + flag_idx, 1)

            i += N_COMM_CTAS
    else:
        cid = pid - N_COMM_CTAS
        t = cid
        while t < n_recv_tiles:
            is_real = tl.load(recv_tile_valid_ptr + t)
            if is_real != 0:
                while tl.atomic_add(local_flag_ptr + t, 0) == 0:
                    pass
            t += N_WAIT_CTAS


def dispatch_scatter_wait(
    routed_input: torch.Tensor,
    *,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    local_flag: torch.Tensor,
    recv_tile_valid: torch.Tensor,
    dispatch_tiles,
    n_recv_tiles: int,
    n_comm_ctas: int,
    n_wait_ctas: int,
    block_m: int,
) -> None:
    hidden_dim = routed_input.shape[1]
    dtype = _DTYPE_MAP[routed_input.dtype]
    n_dispatch_tiles = dispatch_tiles.peer_rank.numel()

    grid = (n_comm_ctas + n_wait_ctas,)
    _dispatch_scatter_wait_kernel[grid](
        routed_input,
        hidden_peer_ptrs,
        flag_peer_ptrs,
        local_flag,
        recv_tile_valid,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.peer_row_start,
        dispatch_tiles.valid_rows,
        dispatch_tiles.flag_index,
        n_dispatch_tiles,
        n_recv_tiles,
        HIDDEN_DIM=hidden_dim,
        N_COMM_CTAS=n_comm_ctas,
        N_WAIT_CTAS=n_wait_ctas,
        BLOCK_M=block_m,
        DTYPE=dtype,
        num_warps=4,
    )


@triton.jit
def _fused_combine_reduce_kernel(
    expert_out_ptr,
    weighted_routed_out_ptr,
    routed_scores_ptr,
    hidden_peer_ptrs,
    flag_peer_ptrs,
    local_hidden_ptr,
    local_flag_ptr,
    tile_peer_rank_ptr,
    tile_local_row_start_ptr,
    tile_peer_row_start_ptr,
    tile_valid_rows_ptr,
    tile_flag_index_ptr,
    n_combine_tiles,
    dispatch_peer_rank_ptr,
    dispatch_local_row_start_ptr,
    dispatch_tile_ordinal_ptr,
    dispatch_valid_rows_ptr,
    n_dispatch_tiles,
    OUT_DIM: tl.constexpr,
    N_COMM_CTAS: tl.constexpr,
    N_COMPUTE_CTAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid < N_COMM_CTAS:
        ocol = tl.arange(0, OUT_DIM)
        i = pid
        while i < n_combine_tiles:
            dest_rank = tl.load(tile_peer_rank_ptr + i)
            if dest_rank >= 0:
                local_start = tl.load(tile_local_row_start_ptr + i).to(tl.int64)
                peer_start = tl.load(tile_peer_row_start_ptr + i).to(tl.int64)
                valid = tl.load(tile_valid_rows_ptr + i)
                flag_idx = tl.load(tile_flag_index_ptr + i)

                rows = tl.arange(0, BLOCK_M)
                row_mask = rows < valid
                vals = tl.load(
                    expert_out_ptr + (local_start + rows)[:, None] * OUT_DIM + ocol[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                dest_base = tl.load(hidden_peer_ptrs + dest_rank).to(tl.pointer_type(DTYPE))
                tl.store(
                    dest_base + (peer_start + rows)[:, None] * OUT_DIM + ocol[None, :],
                    vals,
                    mask=row_mask[:, None],
                )

                tl.debug_barrier()
                flag_base = tl.load(flag_peer_ptrs + dest_rank).to(tl.pointer_type(tl.int32))
                tl.atomic_xchg(flag_base + flag_idx, 1)

            i += N_COMM_CTAS
    else:
        cid = pid - N_COMM_CTAS
        ocol = tl.arange(0, OUT_DIM)

        i = cid
        while i < n_dispatch_tiles:
            dest_rank_check = tl.load(dispatch_peer_rank_ptr + i)
            if dest_rank_check >= 0:
                local_start = tl.load(dispatch_local_row_start_ptr + i).to(tl.int64)
                ordinal = tl.load(dispatch_tile_ordinal_ptr + i).to(tl.int64)
                valid = tl.load(dispatch_valid_rows_ptr + i)
                flag_idx = ordinal

                while tl.atomic_add(local_flag_ptr + flag_idx, 0) == 0:
                    pass

                rows = tl.arange(0, BLOCK_M)
                row_mask = rows < valid
                combine_rows = ordinal * BLOCK_M + rows
                vals = tl.load(
                    local_hidden_ptr + combine_rows[:, None] * OUT_DIM + ocol[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                ).to(tl.float32)

                routed_rows = local_start + rows
                scores = tl.load(routed_scores_ptr + routed_rows, mask=row_mask, other=0.0).to(tl.float32)
                vals = vals * scores[:, None]

                tl.store(
                    weighted_routed_out_ptr + routed_rows[:, None] * OUT_DIM + ocol[None, :],
                    vals.to(DTYPE),
                    mask=row_mask[:, None],
                )

            i += N_COMPUTE_CTAS


def fused_combine_reduce(
    expert_out: torch.Tensor,
    weighted_routed_out: torch.Tensor,
    routed_scores: torch.Tensor,
    *,
    hidden_peer_ptrs: torch.Tensor,
    flag_peer_ptrs: torch.Tensor,
    local_hidden: torch.Tensor,
    local_flag: torch.Tensor,
    combine_tiles,
    dispatch_tiles,
    n_comm_ctas: int,
    n_compute_ctas: int,
    block_m: int,
) -> None:
    out_dim = expert_out.shape[1]
    dtype = _DTYPE_MAP[expert_out.dtype]
    n_combine_tiles = combine_tiles.peer_rank.numel()
    n_dispatch_tiles = dispatch_tiles.peer_rank.numel()

    grid = (n_comm_ctas + n_compute_ctas,)
    _fused_combine_reduce_kernel[grid](
        expert_out,
        weighted_routed_out,
        routed_scores,
        hidden_peer_ptrs,
        flag_peer_ptrs,
        local_hidden,
        local_flag,
        combine_tiles.peer_rank,
        combine_tiles.local_row_start,
        combine_tiles.peer_row_start,
        combine_tiles.valid_rows,
        combine_tiles.flag_index,
        n_combine_tiles,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.own_tile_ordinal,
        dispatch_tiles.valid_rows,
        n_dispatch_tiles,
        OUT_DIM=out_dim,
        N_COMM_CTAS=n_comm_ctas,
        N_COMPUTE_CTAS=n_compute_ctas,
        BLOCK_M=block_m,
        DTYPE=dtype,
        num_warps=4,
    )
