"""Autograd-enabled comet_moe layer -- dispatch, differentiable expert FFN, combine, all with a
real backward pass, so this package can actually be used in training. Forward-only was never
going to move the needle on training step time: a real training-step profile (Qwen3-30B-A3B,
ep=8, `prime-rl-bench`'s `benchmarks/profiling/SYNC_PROFILE_2026-08-17.md`) shows EP all-to-all
at ~38% of top-kernel GPU time, and its *combine backward* (`ScatterAddBackward0`) as the single
largest named op (~528ms, bigger than the forward comm itself) -- backward communication, not
just forward FLOPs, is the thing worth making fast here.

`CometMoELayerFunction.backward` does not build a new schedule or write new communication
kernels: it reuses `dispatch_tiles`/`combine_tiles` from the forward pass with roles swapped
(dispatch's own producer schedule doubles as combine-direction backward's producer schedule, and
vice versa) and the same `comet_scatter` primitives (`scatter_tiles`/`wait_tiles`) the forward
pass already validated. It also reuses the same trick that let forward's combine avoid an atomic
scatter-add (`metadata.py`'s `token_row_map`, a token -> its top_k routed-row indices map) to
avoid one in backward too, for exactly the op (`ScatterAddBackward0`) the real profile flagged as
the single biggest cost: `grad_x = grad_routed_input[token_row_map].sum(dim=1)` is a plain
gather-sum, not a scatter-add, because `token_row_map` is a bijection over routed rows.

The expert FFN itself uses `differentiable_ffn.py` (`torch._grouped_mm`-based, real autograd),
not `flash_moe_compute.py`'s `flash_moe` kernel: `flash_moe` is forward-only, and writing it a
backward is a real CUDA undertaking, not something to bolt on here. See `differentiable_ffn.py`'s
docstring for why that trade is the right one given where the real cost is.
"""

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.comet_moe.buffers import CometMoEBuffers, init_comet_moe_buffers
from prime_rl.trainer.distributed.comet_moe.differentiable_ffn import differentiable_expert_ffn
from prime_rl.trainer.distributed.comet_moe.metadata import (
    TileList,
    build_schedule,
    compute_backward_aux,
)
from prime_rl.trainer.models.layers.activations import Activation


def init_comet_moe_grad_buffers(
    group: dist.ProcessGroup,
    *,
    hidden_dim: int,
    dispatch_capacity: int,
    combine_capacity: int,
    block_m: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[CometMoEBuffers, CometMoEBuffers]:
    """Allocate the two symmetric-memory buffer sets `CometMoELayerFunction`'s backward needs,
    with the *same* shape parameters as the forward pass's own `bufs` -- same `hidden_dim`,
    `dispatch_capacity`, `combine_capacity`, and **the same `block_m`**: backward addresses these
    buffers' flag arrays using `dispatch_tiles`/`combine_tiles`' own `flag_index` values, computed
    at schedule-build time as `peer_row_start // block_m` against the *forward* `block_m` -- a
    different `block_m` here would size the flags array to a different tile count and silently
    alias unrelated tiles onto the same flag.

    Kept fully separate from the forward pass's own buffers, not reused: backward for layer N can
    run long after forward for layer N (or N+1, N+2, ...) has already reset/reused its buffers
    for other calls, so sharing would be a real lifetime hazard, not just a style choice.

    Returns `(grad_recv, grad_combine)`: `grad_recv` mirrors `dispatch_hidden`/`dispatch_flags`
    (backward-of-combine's target -- grad w.r.t. this rank's own expert_out, received from every
    origin that sent tokens here); `grad_combine` mirrors `combine_hidden`/`combine_flags`
    (backward-of-dispatch's target -- grad w.r.t. this rank's own routed_input, in
    combine_hidden-shaped aligned space, gathered back into routed-row order via
    `metadata.compute_backward_aux`). Only `.dispatch_hidden`/`.dispatch_flags` of `grad_recv` and
    only `.combine_hidden`/`.combine_flags` of `grad_combine` are ever used.
    """
    grad_recv = init_comet_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=dispatch_capacity,
        combine_capacity=combine_capacity,
        block_m=block_m,
        dtype=dtype,
        device=device,
    )
    grad_combine = init_comet_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=dispatch_capacity,
        combine_capacity=combine_capacity,
        block_m=block_m,
        dtype=dtype,
        device=device,
    )
    return grad_recv, grad_combine


def _expert_aligned_chunk_bounds(recv_expert_row_offsets: torch.Tensor, recv_capacity: int, n_chunks: int) -> list[int]:
    """`n_chunks + 1` row-boundaries (host ints, ascending) splitting `[0, recv_capacity)` into
    roughly `n_chunks` pieces -- but snapped to real per-local-expert boundaries
    (`recv_expert_row_offsets`), so no local expert's rows are ever split across two chunks.

    An earlier version of this function split by raw tile count, agnostic to expert boundaries.
    That measurably regressed wallclock at real scale (0.82-0.88x vs the plain-NCCL reference,
    worse than the 1.03-1.29x this package had *before* chunking): with `num_local_experts`
    experts and a chunk boundary landing mid-expert, that expert's rows get split across two
    `torch._grouped_mm` calls instead of one, and grouped-GEMM's fixed per-group launch overhead
    is significant exactly at this shape (many experts, modest rows-per-expert). Snapping to real
    expert boundaries means each chunk's `torch._grouped_mm` call only ever sees *whole* experts,
    so the total number of (expert-group, GEMM-call) pairs across all chunks stays equal to
    `num_local_experts` (the same as the unchunked path), no matter how many chunks there are.

    This costs one small `.tolist()` host sync per forward call (reading back
    `num_local_experts + 1` int32s) that the rest of this package's hot path deliberately avoids
    elsewhere -- but by the time this runs, `recv_expert_row_offsets` was already computed (and
    its producing collective already awaited) to build `dispatch_tiles`/`combine_tiles` for the
    scatter this function's caller already issued, so the sync should find that work already done
    rather than adding a new wait.

    The final boundary is always `recv_capacity` (the static buffer capacity), not the real total
    row count: the last chunk absorbs the fixed alignment-padding tail beyond the real total, the
    same "attribute harmless zero padding rows to the last group" trick
    `metadata.padded_expert_offsets` uses globally.
    """
    offsets = recv_expert_row_offsets.tolist()
    total = offsets[-1]
    if total <= 0:
        return [0, recv_capacity]
    n = max(1, min(n_chunks, len(offsets) - 1))
    bounds = [0]
    for i in range(1, n):
        target = (total * i) // n
        pick = next((o for o in offsets if o >= target and o > bounds[-1]), total)
        if pick > bounds[-1]:
            bounds.append(pick)
    if bounds[-1] != total:
        bounds.append(total)
    bounds[-1] = recv_capacity
    return bounds


def _run_chunked_dispatch_wait_and_ffn(
    comet_scatter,
    bufs: CometMoEBuffers,
    schedule,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_proj: torch.Tensor | None,
    activation: type[Activation],
    *,
    block_m: int,
    n_chunks: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The actual overlap: split the receive side into `n_chunks` expert-aligned row ranges (see
    `_expert_aligned_chunk_bounds`) and pipeline them across two CUDA streams -- one that only ever spin-waits on arrival flags
    (few CTAs, mostly idle-spinning, exactly `comet_scatter.wait_tiles`'s existing role) and the
    main stream, which runs each chunk's expert FFN as soon as (and only as soon as) *that*
    chunk's own wait resolves. Chunk i+1's wait is queued on the comm stream immediately, without
    waiting for chunk i's FFN to finish, so as long as there's SM capacity to spare, the GPU can
    run chunk i's GEMMs and chunk i+1's flag-polling concurrently -- comm hidden behind compute,
    the actual thing COMET/MoK-style designs are for, achieved here via stream scheduling instead
    of intra-kernel CTA role specialization (this package's now-unused, forward-only `kernels.py`
    took the latter approach; see this package's own docstring for why the trainable path doesn't).

    Each chunk is `.clone()`d off the raw symmetric-memory buffer right after (not before) *its
    own* wait resolves, not the whole buffer at once after every chunk is ready -- cloning early
    is exactly the ordering bug this package already hit once (see `CometMoELayerFunction`'s own
    combine-side snapshot comment) applied to the dispatch side, and chunking makes the window
    for it real: a later chunk's data may not have arrived yet when an earlier chunk finishes.

    Each chunk's expert FFN also needs its own `offs` (`torch._grouped_mm`'s per-local-expert row
    boundaries) relative to the chunk's own start, not the whole buffer's -- computed by clamping
    the whole schedule's boundaries into the chunk's row range and shifting, with the same
    "attribute any trailing padding to the chunk's own last expert" trick
    `metadata.padded_expert_offsets` already uses globally (safe for the same reason: the padding
    rows are always zero, so it's harmless which expert's weights process them).

    Returns `(hidden_chunks, expert_out)` -- the `n_chunks` per-chunk clones as a *list* (each
    its own leaf, not concatenated into one tensor) plus `expert_out`, concatenated. Deliberately
    not concatenating `hidden_chunks` too: `expert_out`'s graph reaches each `hidden_chunks[i]`
    directly (through that chunk's own `differentiable_expert_ffn` call, not through a `cat` of
    the inputs), so `torch.autograd.grad(expert_out, hidden_chunks + [weights...], ...)` in
    `backward` correctly gets every chunk's own input gradient *and* the weight gradients summed
    across all chunks in one call -- concatenating the inputs first would build a `dispatch_hidden`
    that is a *sibling* of `expert_out` in the graph, not an ancestor, silently breaking that.

    Backward does not otherwise need a matching chunked implementation: `torch.autograd.grad`
    walks the graph above correctly regardless of how forward built it. It does not, however,
    *run* backward through this same chunk-overlapped schedule -- that's a separate,
    currently-unimplemented lever (see this package's docstring).
    """
    bounds = _expert_aligned_chunk_bounds(schedule.recv_expert_row_offsets, schedule.recv_capacity, n_chunks)
    n = len(bounds) - 1

    comm_stream = torch.cuda.Stream()
    ready = [torch.cuda.Event() for _ in range(n)]
    with torch.cuda.stream(comm_stream):
        for i in range(n):
            lo, hi = bounds[i] // block_m, bounds[i + 1] // block_m
            comet_scatter.wait_tiles(
                bufs.dispatch_flags.local[lo:hi],
                schedule.recv_tile_valid[lo:hi],
            )
            ready[i].record(comm_stream)

    compute_stream = torch.cuda.current_stream()
    offsets = schedule.recv_expert_row_offsets[1:]
    hidden_chunks = []
    expert_out_chunks = []
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        compute_stream.wait_event(ready[i])
        with torch.no_grad():
            hidden_chunk = bufs.dispatch_hidden.local[lo:hi].clone()
        hidden_chunk.requires_grad_(True)
        offs_chunk = (offsets.clamp(min=lo, max=hi) - lo).to(torch.int32)
        offs_chunk[-1] = hi - lo  # force: see this function's docstring
        with torch.enable_grad():
            expert_out_chunks.append(
                differentiable_expert_ffn(hidden_chunk, offs_chunk, up_proj, down_proj, activation, gate_proj)
            )
        hidden_chunks.append(hidden_chunk)
    # `torch.autograd.Function.forward` runs with grad globally disabled, so this `cat` must be
    # inside `enable_grad()` too -- outside it, `cat` would silently produce a `requires_grad=False`
    # tensor (regardless of its inputs), breaking `backward`'s `torch.autograd.grad(expert_out, ...)`.
    with torch.enable_grad():
        expert_out = torch.cat(expert_out_chunks, dim=0)
    return hidden_chunks, expert_out


class CometMoELayerFunction(torch.autograd.Function):
    """Forward: real routing -> `comet_scatter` dispatch -> differentiable expert FFN ->
    `comet_scatter` combine -> gather-sum. Backward: the exact mirror -- see module docstring.
    """

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
        bufs: CometMoEBuffers,
        grad_recv: CometMoEBuffers,
        grad_combine: CometMoEBuffers,
        group: dist.ProcessGroup,
        num_experts: int,
        top_k: int,
        block_m: int,
        n_blocks: int,
        n_chunks: int,
    ) -> torch.Tensor:
        import prime_kernels

        comet_scatter = prime_kernels.load("comet_scatter")

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

        with torch.no_grad():
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

        is_gated = gate_proj is not None
        # The overlap: each chunk's expert FFN starts as soon as (and only as soon as) *that*
        # chunk's own tiles have arrived, on a separate stream from the one still spin-waiting on
        # later chunks -- see `_run_chunked_dispatch_wait_and_ffn`'s docstring.
        hidden_chunks, expert_out = _run_chunked_dispatch_wait_and_ffn(
            comet_scatter,
            bufs,
            schedule,
            up_proj,
            down_proj,
            gate_proj if is_gated else None,
            activation,
            block_m=block_m,
            n_chunks=n_chunks,
        )

        with torch.no_grad():
            comet_scatter.scatter_tiles(
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
            comet_scatter.wait_and_reduce(
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
            # Snapshotted *after* `wait_and_reduce` returns, not right after issuing the scatter:
            # `combine_hidden` is symmetric memory *other* ranks write into, and `wait_and_reduce`
            # is what actually waits for those writes to land (per-tile flags) -- cloning any
            # earlier races the peers' writes and was measured to silently corrupt
            # `grad_routed_scores` at real scale (num_experts=128, top_k=8), while smaller
            # configurations happened not to expose it. Ordering after `wait_and_reduce` is
            # enough: both are queued on the same CUDA stream, so `.clone()` cannot start running
            # until that kernel's own internal waits have completed.
            combine_hidden_snapshot = bufs.combine_hidden.local[: schedule.combine_capacity].clone()
            output = weighted_routed_out[schedule.token_row_map].sum(dim=1)
        bufs.barrier()

        # `gate_proj` is saved unconditionally in a fixed slot -- `save_for_backward` wants real
        # tensors, not `None` -- but is a throwaway placeholder (`up_proj` again) when this
        # activation is ungated; `ctx.is_gated` (not a tensor, so stored directly) is what
        # `backward` actually checks before touching it. `hidden_chunks` is variable-length
        # (`n_chunks` of them), so it goes last and `ctx.n_chunks` records how many to slice back
        # off the end of `ctx.saved_tensors`.
        ctx.save_for_backward(
            expert_out,
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
            *hidden_chunks,
        )
        ctx.n_chunks = len(hidden_chunks)
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

        comet_scatter = prime_kernels.load("comet_scatter")

        n_chunks = ctx.n_chunks
        fixed_tensors, hidden_chunks = ctx.saved_tensors[:-n_chunks], list(ctx.saved_tensors[-n_chunks:])
        (
            expert_out,
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
        ) = fixed_tensors
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

        # --- backward of `output = weighted_routed_out[token_row_map].sum(dim=1)` ---
        # A plain scatter, not scatter-add: `token_row_map.reshape(-1)` is a permutation of
        # `[0, n_local_routed)` (every routed row belongs to exactly one (token, k) pair), so
        # there is no aliasing to accumulate.
        grad_weighted_routed_out = torch.zeros(n_local_routed, hidden_dim, dtype=dtype, device=device)
        grad_weighted_routed_out[token_row_map.reshape(-1).long()] = (
            grad_output.unsqueeze(1).expand(n_local_tokens, top_k, hidden_dim).reshape(-1, hidden_dim)
        )

        # --- backward of `weighted_routed_out[r] = combine_hidden[pos(r)] * routed_scores[r]` ---
        grad_combine_hidden = torch.zeros(combine_capacity, hidden_dim, dtype=dtype, device=device)
        grad_combine_hidden[dispatch_row_to_combine_pos] = (
            grad_weighted_routed_out.float() * routed_scores.unsqueeze(1)
        ).to(dtype)
        combine_hidden_at_r = combine_hidden_snapshot[dispatch_row_to_combine_pos]
        grad_routed_scores = (grad_weighted_routed_out.float() * combine_hidden_at_r.float()).sum(-1)

        # --- backward of combine's forward scatter: grad_expert_out, sent by *me* (the forward
        # combine's destination-of-the-forward-send / origin-of-the-gradient) to the rank that
        # computed expert_out, using *my own* `dispatch_tiles` as the producer schedule -- exactly
        # dispatch's forward schedule, since `dispatch_tiles.peer_row_start` already lines up with
        # the receiving rank's expert_out/dispatch_hidden row layout. ---
        # Full `.reset()` (zeros both data and flags), not just the flags: a stale, uninitialized,
        # or otherwise-garbage float in a padding row that ends up NaN (rather than merely finite
        # garbage) would survive `0 * NaN == NaN` in the FFN backward math below and corrupt real
        # weight gradients -- cheap insurance matching forward's own established `bufs.reset()`.
        grad_recv.reset()
        grad_recv.barrier()
        comet_scatter.scatter_tiles(
            grad_combine_hidden,
            grad_recv.dispatch_hidden.peer_ptrs,
            grad_recv.dispatch_flags.peer_ptrs,
            dispatch_tiles.peer_rank,
            dispatch_tiles.own_tile_ordinal * block_m,
            dispatch_tiles.peer_row_start,
            dispatch_tiles.valid_rows,
            dispatch_tiles.flag_index,
            n_blocks,
        )
        comet_scatter.wait_tiles(grad_recv.dispatch_flags.local, recv_tile_valid)
        grad_expert_out = grad_recv.dispatch_hidden.local[:recv_capacity]
        grad_recv.barrier()

        # --- backward of the differentiable expert FFN: real autograd, no manual gradient math.
        # `hidden_chunks` are each their own leaf in the graph (see
        # `_run_chunked_dispatch_wait_and_ffn`'s docstring for why they aren't one concatenated
        # tensor); passing all of them plus the shared weights as `inputs` in one call gets back
        # each chunk's own input gradient *and* the weight gradients already summed across every
        # chunk's use of them, in one pass over the graph. ---
        ffn_inputs = hidden_chunks + [up_proj, down_proj] + ([gate_proj] if is_gated else [])
        ffn_grads = torch.autograd.grad(
            expert_out,
            ffn_inputs,
            grad_outputs=grad_expert_out.to(expert_out.dtype),
        )
        grad_hidden_chunks, grad_up_proj, grad_down_proj = (
            ffn_grads[:n_chunks],
            ffn_grads[n_chunks],
            ffn_grads[n_chunks + 1],
        )
        grad_gate_proj = ffn_grads[n_chunks + 2] if is_gated else None
        grad_dispatch_hidden = torch.cat(grad_hidden_chunks, dim=0)

        # --- backward of dispatch's forward scatter: grad_routed_input, sent by *me* (the rank
        # that computed expert_out) to each origin, using *my own* `combine_tiles` as the producer
        # schedule -- exactly combine's forward schedule, landing at each origin's aligned
        # combine_hidden-shaped space; `compute_backward_aux`'s mapping un-aligns it back into
        # routed-row order below. ---
        grad_combine.reset()
        grad_combine.barrier()
        comet_scatter.scatter_tiles(
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
        # `combine_tile_valid`'s length is this schedule's `max_dispatch_tiles`, generally smaller
        # than `grad_combine`'s full buffer capacity (`bufs.combine_capacity`, a static, generous
        # bound) -- `wait_tiles` requires the two to match exactly, so slice.
        comet_scatter.wait_tiles(grad_combine.combine_flags.local[: combine_capacity // block_m], combine_tile_valid)
        grad_combine_bwd = grad_combine.combine_hidden.local[:combine_capacity]
        grad_routed_input = grad_combine_bwd[dispatch_row_to_combine_pos]
        grad_combine.barrier()

        # --- backward of `_local_reorder`'s `routed_input = x[token_indices_experts_sorted]` and
        # `top_scores_experts_sorted = top_scores.reshape(-1)[argsort_perm]` (the *pre*-`// top_k`
        # permutation `_local_reorder` computes internally but doesn't return -- recomputed here
        # from the saved `selected_experts_indices`, since it's cheap and this keeps forward's own
        # hot path exactly as it was). Both use `token_row_map`/`argsort_perm` as bijections, so
        # both are plain gathers, not the `scatter_add` the real profile flagged as the single
        # biggest named op in this exact spot. ---
        grad_x = grad_routed_input[token_row_map].sum(dim=1)

        argsort_perm = torch.argsort(selected_experts_indices.reshape(-1), stable=True)
        grad_top_scores = torch.zeros(n_local_tokens * top_k, dtype=dtype, device=device)
        grad_top_scores[argsort_perm] = grad_routed_scores.to(dtype)
        grad_top_scores = grad_top_scores.view(n_local_tokens, top_k)

        # (x, top_scores, selected_experts_indices, up_proj, down_proj, gate_proj, activation,
        #  bufs, grad_recv, grad_combine, group, num_experts, top_k, block_m, n_blocks, n_chunks)
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
            None,
        )
