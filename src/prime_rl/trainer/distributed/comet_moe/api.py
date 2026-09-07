import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.comet_moe.buffers import CometMoEBuffers, init_comet_moe_buffers
from prime_rl.trainer.distributed.comet_moe.flash_moe_compute import FlashMoeScratch, run_flash_moe_expert_compute
from prime_rl.trainer.distributed.comet_moe.kernels import fused_combine_reduce, fused_dispatch_gemm
from prime_rl.trainer.distributed.comet_moe.metadata import build_schedule


def run_comet_moe_layer_with_buffers(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    weight: torch.Tensor,  # (num_local_experts, hidden_dim, out_dim)
    bufs: CometMoEBuffers,
    *,
    num_experts: int,
    top_k: int,
    group: dist.ProcessGroup,
    block_m: int,
    n_comm_ctas: int,
    n_compute_ctas: int,
) -> torch.Tensor:
    device = x.device
    out_dim = weight.shape[-1]

    # Both bounds are ordinary host ints -- `bufs.dispatch_capacity` is static config, and
    # `build_schedule`'s dispatch-side bound is computed from `x.shape[0]` (tensor-shape
    # metadata, not a device value) -- so these checks cost nothing (no `.item()`/sync). Only
    # the *receive* side depends on other ranks' real routing and can't be checked this way; see
    # `build_schedule`'s docstring for how that overflow case is handled (silent truncation).
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

    bufs.reset()
    # Buffers are shared, symmetric memory: a peer can start writing into mine as soon as *it*
    # reaches its dispatch kernel, regardless of whether *I've* finished zeroing my own copy of
    # these buffers yet. Without this barrier, a fast rank's write can land before a slow rank's
    # `reset()` runs and then get wiped by it -- silently dropping data and deadlocking that
    # rank's consumer, which polls a flag that was set once and then erased. `bufs.barrier()` is
    # a device-side symmetric-memory barrier, not `dist.barrier()`: both give the same ordering
    # guarantee, but `dist.barrier()` was measured to cost several milliseconds *per call* in some
    # environments (idle-GPU/driver-reinit overhead after any host-blocking NCCL sync) -- see
    # `CometMoEBuffers.barrier`'s docstring.
    bufs.barrier()
    expert_out = torch.zeros(schedule.recv_capacity, out_dim, dtype=x.dtype, device=device)
    n_local_routed = schedule.routed_input.shape[0]
    weighted_routed_out = torch.zeros(n_local_routed, out_dim, dtype=x.dtype, device=device)

    n_recv_tiles = schedule.recv_capacity // block_m
    fused_dispatch_gemm(
        schedule.routed_input,
        weight,
        expert_out,
        hidden_peer_ptrs=bufs.dispatch_hidden.peer_ptrs,
        flag_peer_ptrs=bufs.dispatch_flags.peer_ptrs,
        local_hidden=bufs.dispatch_hidden.local,
        local_flag=bufs.dispatch_flags.local,
        recv_tile_to_local_expert=schedule.recv_tile_to_local_expert,
        recv_tile_valid=schedule.recv_tile_valid,
        dispatch_tiles=schedule.dispatch_tiles,
        n_recv_tiles=n_recv_tiles,
        n_comm_ctas=n_comm_ctas,
        n_compute_ctas=n_compute_ctas,
        block_m=block_m,
    )

    fused_combine_reduce(
        expert_out,
        weighted_routed_out,
        schedule.routed_scores,
        hidden_peer_ptrs=bufs.combine_hidden.peer_ptrs,
        flag_peer_ptrs=bufs.combine_flags.peer_ptrs,
        local_hidden=bufs.combine_hidden.local,
        local_flag=bufs.combine_flags.local,
        combine_tiles=schedule.combine_tiles,
        dispatch_tiles=schedule.dispatch_tiles,
        n_comm_ctas=n_comm_ctas,
        n_compute_ctas=n_compute_ctas,
        block_m=block_m,
    )
    # Mirrors the barrier above: a fast rank must not race ahead into the *next* call's
    # `reset()` (which zeros these same shared buffers) while a slow peer is still reading or
    # writing them for *this* call.
    bufs.barrier()

    # Sum each original token's (up to `top_k`) routed-row contributions -- a plain gather, not
    # an atomic scatter-add inside the kernel; see `fused_combine_reduce`'s docstring for why.
    return weighted_routed_out[schedule.token_row_map].sum(dim=1)


def run_comet_moe_layer_with_buffers_flash_moe(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    gate_up_weight: torch.Tensor,  # (num_local_experts, 2*intermediate, hidden_dim), bf16
    down_weight: torch.Tensor,  # (num_local_experts, hidden_dim, intermediate), bf16
    bufs: CometMoEBuffers,
    scratch: FlashMoeScratch,  # from `flash_moe_compute.init_flash_moe_scratch`
    *,
    num_experts: int,
    top_k: int,
    group: dist.ProcessGroup,
    n_blocks: int = 132,
    flash_moe_block_n: int = 64,
    flash_moe_warp_n: int = 4,
    flash_moe_stages: int = 2,
) -> torch.Tensor:
    """Same dispatch/combine tile schedule as `run_comet_moe_layer_with_buffers`, but the expert
    compute step is `prime_kernels.flash_moe`'s real tcgen05/TMA fused MoE kernel (gate/up/SwiGLU/
    down + top-k reduce in one launch, plain bf16 -- see `flash_moe_compute.py` for why not mxfp8)
    instead of this package's own hand-written, un-tiled Triton GEMM, and the symmetric-memory
    scatter/wait/reduce steps run through `prime_kernels.comet_scatter`'s hand-written CUDA
    kernels instead of Triton.

    Triton's remote-store codegen for this access pattern (many ~512KB tile-granular symmetric-
    memory writes) was profiled well below real NVLink bandwidth; `comet_scatter` does the same
    scatter as a plain vectorized (128-bit) contiguous byte copy per tile, and the combine reduce
    step's plain (non-atomic; see `fused_combine_reduce`'s docstring) weighted store the same way.

    `scratch` must be sized for this rank's fixed `bufs.dispatch_capacity` and local routed-token
    count (see `init_flash_moe_scratch`) and reused across calls, not reallocated -- allocating
    its tensors fresh every call was measured fine in isolation but caused a ~10x regression under
    real N-way-concurrent (multi-rank) load, from `cudaMalloc`/`cudaFree`'s driver-level lock
    contention; see `FlashMoeScratch`'s docstring.

    `block_m` is fixed at flash_moe's required tile size (128), not a caller parameter like in
    `run_comet_moe_layer_with_buffers`. Compute becomes a separate, whole-buffer kernel launch
    rather than being CTA-interleaved with the dispatch scatter: `comet_scatter.wait_tiles` after
    the scatter returns only once every row this rank expects has arrived, since flash_moe needs
    the whole buffer populated before it can run -- there's no compute to interleave the wait
    with here regardless, unlike the CTA-specialized Triton kernels `kernels.py` still uses for
    the non-flash_moe naive-GEMM path.
    """
    import prime_kernels
    from prime_kernels.flash_moe import BLOCK_M as FLASH_MOE_BLOCK_M

    comet_scatter = prime_kernels.load("comet_scatter")

    max_recv_tiles = bufs.dispatch_capacity // FLASH_MOE_BLOCK_M
    schedule = build_schedule(
        x,
        top_scores,
        selected_experts_indices,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
        block_m=FLASH_MOE_BLOCK_M,
        max_recv_tiles=max_recv_tiles,
    )
    if schedule.combine_capacity > bufs.combine_capacity:
        raise RuntimeError(
            f"this rank's own routed tokens need combine_capacity={schedule.combine_capacity} "
            f"> buffer combine_capacity={bufs.combine_capacity}"
        )

    bufs.reset()
    bufs.barrier()  # see run_comet_moe_layer_with_buffers's docstring and CometMoEBuffers.barrier's

    dispatch_tiles = schedule.dispatch_tiles
    comet_scatter.scatter_tiles(
        schedule.routed_input,
        bufs.dispatch_hidden.peer_ptrs,
        bufs.dispatch_flags.peer_ptrs,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.peer_row_start,
        dispatch_tiles.valid_rows,
        dispatch_tiles.flag_index,
        n_blocks,
    )
    comet_scatter.wait_tiles(bufs.dispatch_flags.local, schedule.recv_tile_valid)

    expert_out = run_flash_moe_expert_compute(
        bufs.dispatch_hidden.local[: schedule.recv_capacity],
        schedule.recv_tile_to_local_expert,
        gate_up_weight,
        down_weight,
        scratch,
        block_m=FLASH_MOE_BLOCK_M,
        block_n=flash_moe_block_n,
        warp_n=flash_moe_warp_n,
        stages=flash_moe_stages,
    )
    weighted_routed_out = scratch.weighted_routed_out

    combine_tiles = schedule.combine_tiles
    comet_scatter.scatter_tiles(
        expert_out,
        bufs.combine_hidden.peer_ptrs,
        bufs.combine_flags.peer_ptrs,
        combine_tiles.peer_rank,
        combine_tiles.local_row_start,
        combine_tiles.peer_row_start,
        combine_tiles.valid_rows,
        combine_tiles.flag_index,
        n_blocks,
    )
    comet_scatter.wait_and_reduce(
        bufs.combine_hidden.local,
        bufs.combine_flags.local,
        schedule.routed_scores.float(),
        weighted_routed_out,
        dispatch_tiles.peer_rank,
        dispatch_tiles.local_row_start,
        dispatch_tiles.own_tile_ordinal,
        dispatch_tiles.valid_rows,
        FLASH_MOE_BLOCK_M,
        n_blocks,
    )
    bufs.barrier()

    # Sum each original token's (up to `top_k`) routed-row contributions -- a plain gather, not
    # an atomic scatter-add inside the kernel; see `fused_combine_reduce`'s docstring for why.
    return weighted_routed_out[schedule.token_row_map].sum(dim=1)


def run_comet_moe_layer(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    weight: torch.Tensor,  # (num_local_experts, hidden_dim, out_dim)
    *,
    num_experts: int,
    top_k: int,
    group: dist.ProcessGroup,
    block_m: int,
    n_comm_ctas: int,
    n_compute_ctas: int,
    max_dispatch_capacity: int,
    max_combine_capacity: int,
) -> torch.Tensor:
    """Convenience one-shot wrapper: allocate fresh buffers, run once, return the output.

    For a single call (e.g. a one-off correctness check) this is fine. For repeated calls (a
    real training loop, or a benchmark loop), allocate buffers once with `init_comet_moe_buffers`
    and call `run_comet_moe_layer_with_buffers` directly instead -- see its docstring for why.
    """
    bufs = init_comet_moe_buffers(
        group,
        hidden_dim=x.shape[1],
        dispatch_capacity=max_dispatch_capacity,
        combine_capacity=max_combine_capacity,
        block_m=block_m,
        dtype=x.dtype,
        device=x.device,
    )
    return run_comet_moe_layer_with_buffers(
        x,
        top_scores,
        selected_experts_indices,
        weight,
        bufs,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
        block_m=block_m,
        n_comm_ctas=n_comm_ctas,
        n_compute_ctas=n_compute_ctas,
    )
