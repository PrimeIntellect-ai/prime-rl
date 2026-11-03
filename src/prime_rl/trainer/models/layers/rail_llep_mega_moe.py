"""LLEP on Mega MoE for an EP group of N >= 2 nodes. Every rank sends its tokens to its rail peers (the same local GPU on
every other node) over GPU-initiated InfiniBand (`deep_gemm.rail`); each node then runs LLEP on its own experts over its
ranks' tokens and their rail peers' (an EP group of the node's ranks, NVLink only, so the plan balances each node and
weight copies never leave it), and every node's partial outputs come back the same way and are summed. The swaps run in chunks next
to the expert calls, with no host syncs:

- forward: top-k ids and weights, then rows chunk by chunk (MXFP8 with ``mxfp8_qat``, bf16 with ``bf16``); partial
  outputs back per chunk;
- backward: output gradients chunk by chunk (with ``mxfp8_qat`` also bf16 rows; with ``bf16`` the forward's rows are saved
  instead); input and top-k weight gradients back per chunk.

Rail chunk ``i`` carries a peer's rows ``chunk_rows[i]`` (the same pseudo-random dealing everywhere), and the node's
call ``i`` computes this rank's ``chunk_rows[i]`` and every peer's, so every call waits only for its own chunk. All ranks
pad their tokens to ``max_tokens_per_rank`` (padding routes nowhere), since packed micro batches differ in length and the
layout has to match without exchanging token counts.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from prime_rl.configs.trainer import LLEPConfig
from prime_rl.trainer.models.layers.expert_compute import to_local
from prime_rl.trainer.models.layers.llep_mega_moe import LLEPMegaMoEExpertCompute
from prime_rl.trainer.models.layers.mega_moe import (
    _activation_clamp,
    _dw_dtype,
    cached_qat_forward_weights,
    cached_qat_rounded_weights,
    mxfp8_rounding_args,
    qat_cast_tokens,
)

_RAILS: dict[int, "_Rail"] = {}


def _node_group(group: ProcessGroup, ranks_per_node: int) -> ProcessGroup:
    """The ranks of ``group`` on this rank's node, after checking the group is whole, rail-aligned nodes (node-major)."""
    ranks = dist.get_process_group_ranks(group)
    assert len(ranks) % ranks_per_node == 0 and len(ranks) >= 2 * ranks_per_node, f"EP group {ranks} is not whole nodes"
    nodes = [ranks[k : k + ranks_per_node] for k in range(0, len(ranks), ranks_per_node)]
    for node in nodes:
        assert len({rank // ranks_per_node for rank in node}) == 1, f"EP group {ranks} is not whole nodes"
    assert all(len({rank % ranks_per_node for rank in rail}) == 1 for rail in zip(*nodes)), (
        f"EP group {ranks} is not rail-aligned"
    )
    return dist.new_group(nodes[group.rank() // ranks_per_node], use_local_synchronization=True)


class _Rail:
    def __init__(
        self,
        group: ProcessGroup,
        ranks_per_node: int,
        num_experts: int,
        top_k: int,
        hidden: int,
        max_tokens_per_rank: int,
        num_chunks: int,
        node_compute: LLEPMegaMoEExpertCompute,
        precision: str,
    ) -> None:
        from deep_gemm.rail import RailTransport

        self.num_nodes = group.size() // ranks_per_node
        self.node = group.rank() // ranks_per_node
        self.experts_per_node = num_experts // self.num_nodes
        self.top_k, self.hidden, self.num_chunks, self.precision = top_k, hidden, num_chunks, precision
        self.max_tokens = max_tokens_per_rank
        # exp: RAIL_SCHEDULE=phased runs own rows first and last, peers' rows in between
        self.phased = os.environ.get("RAIL_SCHEDULE") == "phased"
        self.compute = node_compute
        self.state = node_compute.state
        chunk = -(-max_tokens_per_rank // num_chunks)
        self.regions = {
            # top-k ids (int64) and weights (fp32) of every token
            "meta": (max_tokens_per_rank * top_k * 12, 1),
            # E4M3 rows and their UE8M0 scales, or bf16 rows
            "x": (num_chunks * chunk * self.row_bytes, num_chunks),
            # partial outputs, one send area per peer
            "y": (num_chunks * chunk * hidden * 2, num_chunks, True),
            # bf16 rows and output gradients
            "bwd": (num_chunks * chunk * hidden * 4, num_chunks),
            # input gradients (bf16) and top-k weight gradients (fp32), one send area per peer
            "dx": (num_chunks * chunk * (hidden * 2 + top_k * 4), num_chunks, True),
        }
        self.rail = RailTransport(group, ranks_per_node, self.regions)
        self.num_peers = self.rail.num_peers
        self._rows: dict[int, list[torch.Tensor]] = {}

    @property
    def row_bytes(self) -> int:
        return self.hidden + self.hidden // 128 if self.precision == "mxfp8_qat" else self.hidden * 2

    def chunk_rows(self, num_tokens: int) -> list[torch.Tensor]:
        if num_tokens not in self._rows:
            generator = torch.Generator(device="cuda").manual_seed(0)
            order = torch.randperm(num_tokens, generator=generator, device="cuda")
            self._rows[num_tokens] = [order[i :: self.num_chunks].sort().values for i in range(self.num_chunks)]
        return self._rows[num_tokens]

    def layout(self, num_tokens: int) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], list, list[int]]:
        """Own rows of each chunk, their concatenation (the order peers' rows arrive in), the node calls' rows, each
        call's segments (``(peer, chunk)``, peer -1 for this rank's own rows; a peer's chunk sits at
        ``(1 + peer) * num_tokens + offset``) and the chunks' offsets.

        Call i is own chunk i then every peer's chunk i. Phased: calls of ``num_nodes`` segments, half of the own
        chunks first, then every peer's chunks, then the other own chunks, so the first rail chunks arrive behind own
        work and the last partial outputs travel back behind own work."""
        rows = self.chunk_rows(num_tokens)
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + r.numel())
        if self.phased:
            n = self.num_nodes
            own = [[(-1, c) for c in range(i, min(i + n, self.num_chunks))] for i in range(0, self.num_chunks, n)]
            peer_segments = [(p, c) for c in range(self.num_chunks) for p in range(self.num_peers)]
            peer = [peer_segments[i : i + n] for i in range(0, len(peer_segments), n)]
            half = -(-len(own) // 2)
            segments = own[:half] + peer + own[half:]
        else:
            segments = [[(-1, i)] + [(p, i) for p in range(self.num_peers)] for i in range(self.num_chunks)]

        def segment_rows(p: int, c: int) -> torch.Tensor:
            if p < 0:
                return rows[c]
            start = (1 + p) * num_tokens
            return torch.arange(start + offsets[c], start + offsets[c + 1], device=rows[c].device)

        calls = [torch.cat([segment_rows(p, c) for p, c in call]) for call in segments]
        return rows, torch.cat(rows), calls, segments, offsets

    def chunk(self, name: str, i: int, side: str, peer: int = 0) -> torch.Tensor:
        return self.rail.send_buffer(name, i, peer) if side == "send" else self.rail.recv_buffer(name, i, peer)

    def local_ids(self, topk: torch.Tensor) -> torch.Tensor:
        """Global expert ids to this node's (-1: another node's)."""
        local = topk - self.node * self.experts_per_node
        return torch.where((local >= 0) & (local < self.experts_per_node), local, -1)


def _pad(t: torch.Tensor, rows: int, value: float = 0) -> torch.Tensor:
    return torch.nn.functional.pad(t, (0, 0, 0, rows - t.shape[0]), value=value) if t.shape[0] < rows else t


def _split(buffer: torch.Tensor, n: int, *parts: tuple[int, torch.dtype]) -> list[torch.Tensor]:
    """Views of ``n`` rows of each ``(row_bytes, dtype)`` part, one after the other in ``buffer``."""
    out, offset = [], 0
    for row_bytes, dtype in parts:
        out.append(buffer[offset : offset + n * row_bytes].view(dtype).view(n, row_bytes // dtype.itemsize))
        offset += n * row_bytes
    return out


@torch.library.custom_op("prime_rl::rail_llep_forward", mutates_args=())
def rail_llep_forward_op(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    rail_key: int,
    activation_clamp: float | None = None,
    rounding_seed: torch.Tensor | None = None,
    rounding_salts: torch.Tensor | None = None,
    expert_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y, vids, foreign, plan_id, weights, xs)``: the output, and the node plan's routes and top-k weights over this
    rank's tokens followed by every peer's, and with ``bf16`` their rows (for the backward; empty otherwise)."""
    rail = _RAILS[rail_key]
    transport, state = rail.rail, rail.state
    num_tokens, T, H, K, P = x.shape[0], rail.max_tokens, rail.hidden, rail.top_k, rail.num_peers
    x, topk_idx, topk_weights = _pad(x, T), _pad(topk_idx, T, -1), _pad(topk_weights, T)
    rows, order, calls, segments, offsets = rail.layout(T)

    # Top-k ids and weights first: the node's plan needs them before any row
    transport.begin("meta")
    ids_send, weights_send = _split(rail.chunk("meta", 0, "send"), T, (K * 8, torch.int64), (K * 4, torch.float32))
    ids_send.copy_(topk_idx)
    weights_send.copy_(topk_weights)
    transport.send("meta", 0)

    qat = rail.precision == "mxfp8_qat"
    if qat:
        x_fp8, x_sf = qat_cast_tokens(x.to(torch.bfloat16).contiguous())
        x_sf = x_sf.view(torch.uint8)
    else:
        x16 = x.to(torch.bfloat16).contiguous()
    transport.begin("x")
    for i, r in enumerate(rows):
        if qat:
            fp8, sf = _split(rail.chunk("x", i, "send"), r.numel(), (H, torch.uint8), (H // 128, torch.uint8))
            torch.index_select(x_fp8.view(torch.uint8), 0, r, out=fp8)
            torch.index_select(x_sf, 0, r, out=sf)
        else:
            (rows16,) = _split(rail.chunk("x", i, "send"), r.numel(), (H * 2, torch.bfloat16))
            torch.index_select(x16, 0, r, out=rows16)
        transport.send("x", i, r.numel() * rail.row_bytes)

    transport.wait("meta")
    ids, weights = [rail.local_ids(topk_idx)], [topk_weights]
    for p in range(P):
        ids_recv, weights_recv = _split(rail.chunk("meta", 0, "recv", p), T, (K * 8, torch.int64), (K * 4, torch.float32))
        ids.append(rail.local_ids(ids_recv[order]))
        weights.append(weights_recv[order])
    ids, weights = torch.cat(ids), torch.cat(weights).to(torch.float32)
    vids, foreign, plan_id = state.plan(ids)

    if qat:
        xs_fp8 = torch.empty(((1 + P) * T, H), dtype=x_fp8.dtype, device=x.device)
        xs_sf = torch.empty(((1 + P) * T, H // 128), dtype=torch.uint8, device=x.device)
        xs_fp8[:T] = x_fp8
        xs_sf[:T] = x_sf
    else:
        xs16 = torch.empty(((1 + P) * T, H), dtype=torch.bfloat16, device=x.device)
        xs16[:T] = x16
    transport.begin("y")

    def before(i: int) -> None:
        for p, c in segments[i]:
            if p < 0:
                continue
            n = rows[c].numel()
            transport.wait("x", c, c + 1, peer=p)
            lo, hi = (1 + p) * T + offsets[c], (1 + p) * T + offsets[c + 1]
            if qat:
                fp8, sf = _split(rail.chunk("x", c, "recv", p), n, (H, torch.uint8), (H // 128, torch.uint8))
                xs_fp8[lo:hi].view(torch.uint8).copy_(fp8)
                xs_sf[lo:hi].copy_(sf)
            else:
                (rows16,) = _split(rail.chunk("x", c, "recv", p), n, (H * 2, torch.bfloat16))
                xs16[lo:hi] = rows16

    def after(i: int, y_chunk: torch.Tensor) -> None:
        # The call's outputs in segment order; each peer gets its rows' partial outputs back
        start = 0
        for p, c in segments[i]:
            n = rows[c].numel()
            if p >= 0:
                rail.chunk("y", c, "send", p)[: n * H * 2].view(torch.bfloat16).view(n, H).copy_(y_chunk[start : start + n])
                transport.send("y", c, n * H * 2, peer=p)
            start += n

    if qat:
        weights_fp8 = cached_qat_forward_weights(gate_up_proj, down_proj, rounding_seed, rounding_salts, expert_offset)
        y = state.forward_mxfp8(
            xs_fp8, xs_sf.view(torch.int32), vids, foreign, plan_id, weights, weights_fp8.l1, weights_fp8.l2,
            activation_clamp, chunks=calls, before_chunk=before, after_chunk=after,
        )[:T]
    else:
        y = state.forward(
            xs16, vids, foreign, plan_id, weights, gate_up_proj, down_proj, activation_clamp,
            chunks=calls, before_chunk=before, after_chunk=after,
        )[:T]
    transport.wait("y")
    for p in range(P):
        partial = [rail.chunk("y", i, "recv", p)[: r.numel() * H * 2].view(torch.bfloat16).view(-1, H) for i, r in enumerate(rows)]
        y = y.index_add(0, order, torch.cat(partial))
    xs = xs16 if not qat else x.new_empty((0, H), dtype=torch.bfloat16)
    return y[:num_tokens].to(x.dtype), vids, foreign, plan_id, weights, xs


@rail_llep_forward_op.register_fake
def _rail_llep_forward_fake(
    x, topk_idx, topk_weights, gate_up_proj, down_proj, rail_key, activation_clamp=None, rounding_seed=None,
    rounding_salts=None, expert_offset=0,
):
    rail = _RAILS[rail_key]
    rows, state = (1 + rail.num_peers) * rail.max_tokens, rail.state
    return (
        torch.empty_like(x),
        topk_idx.new_empty((rows, topk_idx.shape[1]), dtype=torch.int64),
        topk_idx.new_empty((state.group.size(), state.options.max_foreign), dtype=torch.int32),
        torch.empty(1, dtype=torch.int64),
        topk_weights.new_empty((rows, topk_weights.shape[1]), dtype=torch.float32),
        x.new_empty((rows if rail.precision == "bf16" else 0, x.shape[1]), dtype=torch.bfloat16),
    )


@torch.library.custom_op("prime_rl::rail_llep_backward", mutates_args=())
def rail_llep_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    xs: torch.Tensor,
    vids: torch.Tensor,
    foreign: torch.Tensor,
    plan_id: torch.Tensor,
    weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    rail_key: int,
    activation_clamp: float | None = None,
    rounding_seed: torch.Tensor | None = None,
    rounding_salts: torch.Tensor | None = None,
    expert_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rail = _RAILS[rail_key]
    transport, state = rail.rail, rail.state
    num_tokens, T, H, K, P = x.shape[0], rail.max_tokens, rail.hidden, rail.top_k, rail.num_peers
    x, dy = _pad(x, T), _pad(dy, T)
    rows, order, calls, segments, offsets = rail.layout(T)
    dy16 = dy.to(torch.bfloat16).contiguous()
    # bf16: the forward's node rows are saved (`xs`), so only output gradients cross; mxfp8_qat sends bf16 rows too
    send_rows = rail.precision != "bf16"
    parts = ((H * 2, torch.bfloat16),) * (2 if send_rows else 1)
    if send_rows:
        x16 = x.to(torch.bfloat16).contiguous()

    transport.begin("bwd")
    for i, r in enumerate(rows):
        views = _split(rail.chunk("bwd", i, "send"), r.numel(), *parts)
        torch.index_select(dy16, 0, r, out=views[0])
        if send_rows:
            torch.index_select(x16, 0, r, out=views[1])
        transport.send("bwd", i, r.numel() * H * 2 * len(parts))

    if send_rows:
        xs = torch.empty(((1 + P) * T, H), dtype=torch.bfloat16, device=x.device)
        xs[:T] = x16
    dys = torch.empty(((1 + P) * T, H), dtype=torch.bfloat16, device=x.device)
    dys[:T] = dy16
    transport.begin("dx")

    def before(i: int) -> None:
        for p, c in segments[i]:
            if p < 0:
                continue
            n = rows[c].numel()
            transport.wait("bwd", c, c + 1, peer=p)
            lo, hi = (1 + p) * T + offsets[c], (1 + p) * T + offsets[c + 1]
            views = _split(rail.chunk("bwd", c, "recv", p), n, *parts)
            dys[lo:hi] = views[0]
            if send_rows:
                xs[lo:hi] = views[1]

    def after(i: int, dx_chunk: torch.Tensor, dtopk_chunk: torch.Tensor) -> None:
        start = 0
        for p, c in segments[i]:
            n = rows[c].numel()
            if p >= 0:
                dxr, dtr = _split(rail.chunk("dx", c, "send", p), n, (H * 2, torch.bfloat16), (K * 4, torch.float32))
                dxr.copy_(dx_chunk[start : start + n])
                dtr.copy_(dtopk_chunk[start : start + n])
                transport.send("dx", c, n * (H * 2 + K * 4), peer=p)
            start += n

    if rail.precision == "mxfp8_qat":
        gate_up, down = cached_qat_rounded_weights(gate_up_proj, down_proj, rounding_seed, rounding_salts, expert_offset)
    else:
        gate_up, down = gate_up_proj, down_proj
    dx, dtopk, dw1, dw2 = state.backward(
        dys, xs, vids, foreign, plan_id, weights, gate_up, down, activation_clamp, _dw_dtype(gate_up_proj),
        chunks=calls, before_chunk=before, after_chunk=after,
    )
    transport.wait("dx")
    dx, dtopk = dx[:T], dtopk[:T]
    for p in range(P):
        dx_parts, dtopk_parts = [], []
        for i, r in enumerate(rows):
            dxr, dtr = _split(rail.chunk("dx", i, "recv", p), r.numel(), (H * 2, torch.bfloat16), (K * 4, torch.float32))
            dx_parts.append(dxr)
            dtopk_parts.append(dtr)
        dx = dx.index_add(0, order, torch.cat(dx_parts))
        dtopk = dtopk.index_add(0, order, torch.cat(dtopk_parts))
    dx, dtopk = dx[:num_tokens], dtopk[:num_tokens]
    return dx.to(x.dtype), dtopk, dw1, dw2


@rail_llep_backward_op.register_fake
def _rail_llep_backward_fake(
    dy, x, xs, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rail_key, activation_clamp=None,
    rounding_seed=None, rounding_salts=None, expert_offset=0,
):
    K = vids.shape[1]
    return (
        torch.empty_like(x),
        weights.new_empty((x.shape[0], K)),
        torch.empty_like(gate_up_proj),
        torch.empty_like(down_proj),
    )


def _setup_context(ctx, inputs, output) -> None:
    x, _, topk_weights, gate_up_proj, down_proj, rail_key, activation_clamp, rounding_seed, rounding_salts, offset = inputs
    _, vids, foreign, plan_id, weights, xs = output
    ctx.save_for_backward(x, xs, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rounding_seed, rounding_salts)
    ctx.rail_key, ctx.activation_clamp, ctx.expert_offset = rail_key, activation_clamp, offset
    ctx.topk_dtype = topk_weights.dtype


def _backward(ctx, grad_y, *_):
    x, xs, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rounding_seed, rounding_salts = ctx.saved_tensors
    dx, dtopk, dw1, dw2 = torch.ops.prime_rl.rail_llep_backward(
        grad_y, x, xs, vids, foreign, plan_id, weights, gate_up_proj, down_proj, ctx.rail_key, ctx.activation_clamp,
        rounding_seed, rounding_salts, ctx.expert_offset,
    )
    return dx, None, dtopk.to(ctx.topk_dtype), dw1, dw2, None, None, None, None, None


rail_llep_forward_op.register_autograd(_backward, setup_context=_setup_context)


class RailLLEPMegaMoEExpertCompute(LLEPMegaMoEExpertCompute):
    """LLEP Mega MoE (`LLEPMegaMoEExpertCompute`) over an EP group of N >= 2 nodes; see the module docstring.
    ``options.chunk_tokens`` counts the tokens of a node call (this rank's and every rail peer's)."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden: int,
        intermediate_hidden: int,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_sms: int,
        options: LLEPConfig,
        precision: str = "mxfp8_qat",
        stochastic_rounding: bool = False,
        ranks_per_node: int = 8,
    ) -> None:
        if precision not in ("bf16", "mxfp8_qat"):
            raise ValueError(f"LLEP across nodes supports precision 'bf16' and 'mxfp8_qat', got {precision!r}.")
        node_group = _node_group(group, ranks_per_node)
        N = group.size() // ranks_per_node
        num_chunks = -(-N * max_tokens_per_rank // (options.chunk_tokens or N * max_tokens_per_rank))
        super().__init__(
            num_experts // N, top_k, hidden, intermediate_hidden, node_group, N * max_tokens_per_rank, num_sms,
            options.model_copy(update={"chunk_tokens": N * -(-max_tokens_per_rank // num_chunks)}), precision,
            stochastic_rounding,
        )
        self.num_experts = num_experts
        self.max_tokens_per_rank = max_tokens_per_rank
        rail = _Rail(group, ranks_per_node, num_experts, top_k, hidden, max_tokens_per_rank, num_chunks, self, precision)
        self.rail_key = id(rail)
        _RAILS[self.rail_key] = rail

    @property
    def state(self):
        from prime_rl.trainer.models.layers.llep_mega_moe import _STATES

        return _STATES[self.state_key]

    def __call__(self, experts, x, top_scores, selected_experts_indices):
        if x.shape[0] > self.max_tokens_per_rank:
            raise RuntimeError(f"LLEP Mega MoE buffer holds {self.max_tokens_per_rank} tokens/rank, got {x.shape[0]}.")
        if experts.gate_up_proj is not None:
            gate_up_proj = to_local(experts.gate_up_proj)
        else:
            gate_up_proj = torch.cat([to_local(experts.gate_proj), to_local(experts.up_proj)], dim=1)
        y, *_ = torch.ops.prime_rl.rail_llep_forward(
            x,
            selected_experts_indices.to(torch.int64),
            top_scores.to(torch.float32),
            gate_up_proj,
            to_local(experts.down_proj),
            self.rail_key,
            _activation_clamp(experts.activation),
            *mxfp8_rounding_args(experts),
        )
        return y
