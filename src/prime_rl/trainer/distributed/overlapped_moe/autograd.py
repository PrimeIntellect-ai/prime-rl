import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.overlapped_moe.buffers import OverlappedMoEBuffers, init_overlapped_moe_buffers
from prime_rl.trainer.distributed.overlapped_moe.metadata import (
    TileList,
    build_schedule,
    compute_backward_aux,
)
from prime_rl.trainer.models.layers.activations import Activation, Silu


def init_overlapped_moe_grad_buffers(
    group: dist.ProcessGroup,
    *,
    hidden_dim: int,
    dispatch_capacity: int,
    combine_capacity: int,
    block_m: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[OverlappedMoEBuffers, OverlappedMoEBuffers]:
    grad_recv = init_overlapped_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=dispatch_capacity,
        combine_capacity=combine_capacity,
        block_m=block_m,
        dtype=dtype,
        device=device,
    )
    grad_combine = init_overlapped_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=dispatch_capacity,
        combine_capacity=combine_capacity,
        block_m=block_m,
        dtype=dtype,
        device=device,
    )
    return grad_recv, grad_combine


def _run_fused_kernel_dispatch_and_ffn(
    overlap_kernels,
    bufs: OverlappedMoEBuffers,
    schedule,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj: torch.Tensor | None,
    *,
    block_m: int,
    n_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_producer_blocks = max(1, n_blocks // 8)
    n_consumer_blocks = max(1, n_blocks - n_producer_blocks)
    is_gated = gate_proj is not None
    hidden_dim = bufs.hidden_dim
    intermediate_dim = up_proj.shape[1]
    device = bufs.dispatch_hidden.local.device

    expert_out = torch.empty(schedule.recv_capacity, hidden_dim, dtype=torch.bfloat16, device=device)
    act_scratch = torch.empty(n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device)
    gate_scratch = (
        torch.empty(n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device)
        if is_gated
        else None
    )
    block_start_clock = torch.empty(n_producer_blocks + n_consumer_blocks, dtype=torch.int64, device=device)
    block_end_clock = torch.empty_like(block_start_clock)

    dispatch_tiles = schedule.dispatch_tiles
    overlap_kernels.fused_dispatch_ffn(
        schedule.routed_input,
        bufs.dispatch_hidden.peer_ptrs,
        bufs.dispatch_flags.peer_ptrs,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.peer_row_start,
        dispatch_tiles.valid_rows,
        dispatch_tiles.flag_index,
        bufs.dispatch_hidden.local,
        bufs.dispatch_flags.local,
        schedule.recv_tile_valid,
        schedule.recv_tile_to_local_expert,
        gate_proj,
        up_proj,
        down_proj,
        expert_out,
        act_scratch,
        gate_scratch,
        block_start_clock,
        block_end_clock,
        block_m,
        n_producer_blocks,
        n_consumer_blocks,
    )
    hidden_shadow = bufs.dispatch_hidden.local[: schedule.recv_capacity].clone()

    return hidden_shadow, expert_out


class OverlappedMoELayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        gate_proj: torch.Tensor | None,
        activation: type[Activation],
        bufs: OverlappedMoEBuffers,
        grad_recv: OverlappedMoEBuffers,
        grad_combine: OverlappedMoEBuffers,
        group: dist.ProcessGroup,
        num_experts: int,
        top_k: int,
        block_m: int,
        n_blocks: int,
    ) -> torch.Tensor:
        import prime_kernels

        overlap_kernels = prime_kernels.load("fine_grained_compute_comm_overlap")

        hidden_dim = x.shape[1]
        n_local_tokens = x.shape[0]
        max_recv_tiles = bufs.dispatch_capacity // block_m
        schedule = build_schedule(
            x,
            top_scores,
            selected_experts_indices,
            num_experts=num_experts,
            top_k=top_k,
            group=group,
            block_m=block_m,
            max_recv_tiles=max_recv_tiles,
        )
        if schedule.combine_capacity > bufs.combine_capacity:
            raise RuntimeError(
                f"this rank's own routed tokens need combine_capacity={schedule.combine_capacity} "
                f"> buffer combine_capacity={bufs.combine_capacity}"
            )

        dispatch_tiles = schedule.dispatch_tiles
        combine_tiles = schedule.combine_tiles
        n_local_routed = schedule.routed_input.shape[0]

        bufs.reset()
        bufs.barrier()

        is_gated = gate_proj is not None
        if activation is not Silu:
            raise NotImplementedError(
                "fine_grained_compute_comm_overlap.fused_dispatch_ffn only supports Silu (gated or ungated) -- "
                "other activations aren't implemented."
            )
        hidden_shadow, expert_out = _run_fused_kernel_dispatch_and_ffn(
            overlap_kernels,
            bufs,
            schedule,
            up_proj,
            down_proj,
            gate_proj if is_gated else None,
            block_m=block_m,
            n_blocks=n_blocks,
        )

        with torch.no_grad():
            overlap_kernels.scatter_tiles(
                expert_out.detach(),
                bufs.combine_hidden.peer_ptrs,
                bufs.combine_flags.peer_ptrs,
                combine_tiles.peer_rank,
                combine_tiles.local_row_start,
                combine_tiles.peer_row_start,
                combine_tiles.valid_rows,
                combine_tiles.flag_index,
                n_blocks,
            )
            weighted_routed_out = torch.empty(n_local_routed, hidden_dim, dtype=x.dtype, device=x.device)
            overlap_kernels.wait_and_reduce(
                bufs.combine_hidden.local,
                bufs.combine_flags.local,
                schedule.routed_scores.float(),
                weighted_routed_out,
                dispatch_tiles.peer_rank,
                dispatch_tiles.local_row_start,
                dispatch_tiles.own_tile_ordinal,
                dispatch_tiles.valid_rows,
                block_m,
                n_blocks,
            )
            combine_hidden_snapshot = bufs.combine_hidden.local[: schedule.combine_capacity].clone()
            output = weighted_routed_out[schedule.token_row_map].sum(dim=1)
        bufs.barrier()

        ctx.save_for_backward(
            up_proj,
            down_proj,
            gate_proj if is_gated else up_proj,
            schedule.routed_scores.float(),
            combine_hidden_snapshot,
            schedule.token_row_map,
            selected_experts_indices,
            dispatch_tiles.peer_rank,
            dispatch_tiles.local_row_start,
            dispatch_tiles.peer_row_start,
            dispatch_tiles.valid_rows,
            dispatch_tiles.flag_index,
            dispatch_tiles.own_tile_ordinal,
            combine_tiles.peer_rank,
            combine_tiles.local_row_start,
            combine_tiles.peer_row_start,
            combine_tiles.valid_rows,
            combine_tiles.flag_index,
            schedule.recv_tile_valid,
            schedule.recv_tile_to_local_expert,
            hidden_shadow,
        )
        ctx.is_gated = is_gated
        ctx.grad_recv = grad_recv
        ctx.grad_combine = grad_combine
        ctx.block_m = block_m
        ctx.n_blocks = n_blocks
        ctx.hidden_dim = hidden_dim
        ctx.top_k = top_k
        ctx.n_local_tokens = n_local_tokens
        ctx.n_local_routed = n_local_routed
        ctx.recv_capacity = schedule.recv_capacity
        ctx.combine_capacity = schedule.combine_capacity
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import prime_kernels

        overlap_kernels = prime_kernels.load("fine_grained_compute_comm_overlap")

        (
            up_proj,
            down_proj,
            gate_proj_or_placeholder,
            routed_scores,
            combine_hidden_snapshot,
            token_row_map,
            selected_experts_indices,
            d_peer_rank,
            d_local_row_start,
            d_peer_row_start,
            d_valid_rows,
            d_flag_index,
            d_own_tile_ordinal,
            c_peer_rank,
            c_local_row_start,
            c_peer_row_start,
            c_valid_rows,
            c_flag_index,
            recv_tile_valid,
            recv_tile_to_local_expert,
            hidden_shadow,
        ) = ctx.saved_tensors
        is_gated = ctx.is_gated
        gate_proj = gate_proj_or_placeholder if is_gated else None

        dispatch_tiles = TileList(
            peer_rank=d_peer_rank,
            local_row_start=d_local_row_start,
            peer_row_start=d_peer_row_start,
            valid_rows=d_valid_rows,
            flag_index=d_flag_index,
            own_tile_ordinal=d_own_tile_ordinal,
        )

        grad_recv = ctx.grad_recv
        grad_combine = ctx.grad_combine
        block_m = ctx.block_m
        n_blocks = ctx.n_blocks
        hidden_dim = ctx.hidden_dim
        top_k = ctx.top_k
        n_local_tokens = ctx.n_local_tokens
        n_local_routed = ctx.n_local_routed
        recv_capacity = ctx.recv_capacity
        combine_capacity = ctx.combine_capacity
        device = grad_output.device
        dtype = grad_output.dtype

        dispatch_row_to_combine_pos, combine_tile_valid = compute_backward_aux(
            dispatch_tiles,
            n_local_routed=n_local_routed,
            block_m=block_m,
            device=device,
        )
        grad_weighted_routed_out = torch.zeros(n_local_routed, hidden_dim, dtype=dtype, device=device)
        grad_weighted_routed_out[token_row_map.reshape(-1).long()] = (
            grad_output.unsqueeze(1).expand(n_local_tokens, top_k, hidden_dim).reshape(-1, hidden_dim)
        )
        grad_combine_hidden = torch.zeros(combine_capacity, hidden_dim, dtype=dtype, device=device)
        grad_combine_hidden[dispatch_row_to_combine_pos] = (
            grad_weighted_routed_out.float() * routed_scores.unsqueeze(1)
        ).to(dtype)
        combine_hidden_at_r = combine_hidden_snapshot[dispatch_row_to_combine_pos]
        grad_routed_scores = (grad_weighted_routed_out.float() * combine_hidden_at_r.float()).sum(-1)
        grad_recv.reset()
        grad_recv.barrier()

        # Fused backward kernel: scatters grad_combine_hidden into this rank's dispatch-hidden
        # symmetric-memory buffer (mirroring forward's dispatch, role-swapped) and, per received
        # tile, recomputes up/gate from the saved hidden_shadow and computes BOTH the FFN's input
        # gradient (grad_dispatch_hidden) and its weight gradients (grad_up_proj/grad_down_proj/
        # grad_gate_proj, atomic-accumulated in fp32 since one expert's tokens span multiple
        # tiles) via real WMMA GEMMs -- replacing the old scatter_tiles+wait_tiles+
        # torch.autograd.grad-through-the-shadow-graph path entirely.
        n_producer_blocks = max(1, n_blocks // 8)
        n_consumer_blocks = max(1, n_blocks - n_producer_blocks)
        intermediate_dim = up_proj.shape[1]
        num_local_experts = up_proj.shape[0]

        up_scratch = torch.empty(n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device)
        gate_scratch = (
            torch.empty(n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device)
            if is_gated
            else None
        )
        grad_act_scratch = torch.empty(
            n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device
        )
        act_scratch = torch.empty(n_consumer_blocks, block_m, intermediate_dim, dtype=torch.bfloat16, device=device)
        grad_dispatch_hidden = torch.empty(recv_capacity, hidden_dim, dtype=torch.bfloat16, device=device)
        grad_up_proj_fp32 = torch.zeros(
            num_local_experts, intermediate_dim, hidden_dim, dtype=torch.float32, device=device
        )
        grad_down_proj_fp32 = torch.zeros(
            num_local_experts, hidden_dim, intermediate_dim, dtype=torch.float32, device=device
        )
        grad_gate_proj_fp32 = (
            torch.zeros(num_local_experts, intermediate_dim, hidden_dim, dtype=torch.float32, device=device)
            if is_gated
            else None
        )

        overlap_kernels.fused_grad_combine_ffn(
            grad_combine_hidden,
            grad_recv.dispatch_hidden.peer_ptrs,
            grad_recv.dispatch_flags.peer_ptrs,
            dispatch_tiles.peer_rank,
            dispatch_tiles.own_tile_ordinal * block_m,
            dispatch_tiles.peer_row_start,
            dispatch_tiles.valid_rows,
            dispatch_tiles.flag_index,
            grad_recv.dispatch_hidden.local,
            grad_recv.dispatch_flags.local,
            recv_tile_valid,
            recv_tile_to_local_expert,
            hidden_shadow,
            gate_proj,
            up_proj,
            down_proj,
            grad_dispatch_hidden,
            up_scratch,
            gate_scratch,
            grad_act_scratch,
            act_scratch,
            grad_up_proj_fp32,
            grad_down_proj_fp32,
            grad_gate_proj_fp32,
            block_m,
            n_producer_blocks,
            n_consumer_blocks,
        )
        grad_recv.barrier()

        grad_up_proj = grad_up_proj_fp32.to(up_proj.dtype)
        grad_down_proj = grad_down_proj_fp32.to(down_proj.dtype)
        grad_gate_proj = grad_gate_proj_fp32.to(gate_proj.dtype) if is_gated else None

        grad_combine.reset()
        grad_combine.barrier()
        overlap_kernels.scatter_tiles(
            grad_dispatch_hidden,
            grad_combine.combine_hidden.peer_ptrs,
            grad_combine.combine_flags.peer_ptrs,
            c_peer_rank,
            c_local_row_start,
            c_peer_row_start,
            c_valid_rows,
            c_flag_index,
            n_blocks,
        )
        overlap_kernels.wait_tiles(grad_combine.combine_flags.local[: combine_capacity // block_m], combine_tile_valid)
        grad_combine_bwd = grad_combine.combine_hidden.local[:combine_capacity]
        grad_routed_input = grad_combine_bwd[dispatch_row_to_combine_pos]
        grad_combine.barrier()
        grad_x = grad_routed_input[token_row_map].sum(dim=1)

        argsort_perm = torch.argsort(selected_experts_indices.reshape(-1), stable=True)
        grad_top_scores = torch.zeros(n_local_tokens * top_k, dtype=dtype, device=device)
        grad_top_scores[argsort_perm] = grad_routed_scores.to(dtype)
        grad_top_scores = grad_top_scores.view(n_local_tokens, top_k)
        return (
            grad_x,
            grad_top_scores,
            None,
            grad_up_proj,
            grad_down_proj,
            grad_gate_proj,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
