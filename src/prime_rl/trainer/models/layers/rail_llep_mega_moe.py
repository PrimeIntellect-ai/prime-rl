"""LLEP on Mega MoE for an EP group of two nodes. Every rank swaps its tokens with its rail peer (the same local GPU on
the other node) over GPU-initiated InfiniBand (`deep_gemm.rail`); each node then runs LLEP on its own experts over its
ranks' tokens and their rail peers' (an EP group of the node's ranks, NVLink only, so the plan balances each node and
weight copies never leave it), and the rail peers' partial outputs come back the same way. The swaps run in chunks next
to the expert calls, with no host syncs:

- forward: top-k ids and weights, then rows chunk by chunk (MXFP8 with ``mxfp8_qat``, bf16 with ``bf16``); partial
  outputs back per chunk;
- backward: bf16 rows and output gradients chunk by chunk; input and top-k weight gradients back per chunk.

Rail chunk ``i`` carries the peer's rows ``chunk_rows[i]`` (the same pseudo-random dealing on both sides), and the node's
call ``i`` computes this rank's ``chunk_rows[i]`` and the peer's, so every call waits only for its own chunk. Both peers
pad their tokens to ``max_tokens_per_rank`` (padding routes nowhere), since packed micro batches differ in length and the
layout has to match without exchanging token counts.
"""

from __future__ import annotations

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
    """The ranks of ``group`` on this rank's node, after checking the group is two rail-aligned nodes."""
    ranks = dist.get_process_group_ranks(group)
    nodes = [ranks[:ranks_per_node], ranks[ranks_per_node:]]
    assert len(ranks) == 2 * ranks_per_node, "the rail transport spans two nodes"
    for node in nodes:
        assert len({rank // ranks_per_node for rank in node}) == 1, f"EP group {ranks} is not two whole nodes"
    assert all(a % ranks_per_node == b % ranks_per_node for a, b in zip(*nodes)), f"EP group {ranks} is not rail-aligned"
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

        self.node = group.rank() // ranks_per_node
        self.experts_per_node = num_experts // 2
        self.top_k, self.hidden, self.num_chunks, self.precision = top_k, hidden, num_chunks, precision
        self.max_tokens = max_tokens_per_rank
        self.compute = node_compute
        self.state = node_compute.state
        chunk = -(-max_tokens_per_rank // num_chunks)
        self.regions = {
            # top-k ids (int64) and weights (fp32) of every token
            "meta": (max_tokens_per_rank * top_k * 12, 1),
            # E4M3 rows and their UE8M0 scales, or bf16 rows
            "x": (num_chunks * chunk * self.row_bytes, num_chunks),
            "y": (num_chunks * chunk * hidden * 2, num_chunks),
            # bf16 rows and output gradients
            "bwd": (num_chunks * chunk * hidden * 4, num_chunks),
            # input gradients (bf16) and top-k weight gradients (fp32)
            "dx": (num_chunks * chunk * (hidden * 2 + top_k * 4), num_chunks),
        }
        self.rail = RailTransport(group, ranks_per_node, self.regions)
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

    def layout(self, num_tokens: int) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], list[int]]:
        """Own rows of each chunk, their concatenation (the order rail rows arrive in), the node call's rows (own rows,
        then the rail chunk at ``num_tokens + offset``) and the chunks' offsets."""
        rows = self.chunk_rows(num_tokens)
        offsets = [0]
        for r in rows:
            offsets.append(offsets[-1] + r.numel())
        calls = [
            torch.cat([r, torch.arange(num_tokens + offsets[i], num_tokens + offsets[i + 1], device=r.device)])
            for i, r in enumerate(rows)
        ]
        return rows, torch.cat(rows), calls, offsets

    def chunk(self, name: str, i: int, side: str) -> torch.Tensor:
        return self.rail.send_buffer(name, i) if side == "send" else self.rail.recv_buffer(name, i)

    def local_ids(self, topk: torch.Tensor) -> torch.Tensor:
        """Global expert ids to this node's (-1: the other node's)."""
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y, vids, foreign, plan_id, weights)``: the output, and the node plan's routes and top-k weights over this
    rank's tokens followed by the rail peer's (for the backward)."""
    rail = _RAILS[rail_key]
    transport, state = rail.rail, rail.state
    num_tokens, T, H, K = x.shape[0], rail.max_tokens, rail.hidden, rail.top_k
    x, topk_idx, topk_weights = _pad(x, T), _pad(topk_idx, T, -1), _pad(topk_weights, T)
    rows, order, calls, offsets = rail.layout(T)

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
    ids_recv, weights_recv = _split(rail.chunk("meta", 0, "recv"), T, (K * 8, torch.int64), (K * 4, torch.float32))
    ids = torch.cat([rail.local_ids(topk_idx), rail.local_ids(ids_recv[order])])
    weights = torch.cat([topk_weights, weights_recv[order]]).to(torch.float32)
    vids, foreign, plan_id = state.plan(ids)

    if qat:
        xs_fp8 = torch.empty((2 * T, H), dtype=x_fp8.dtype, device=x.device)
        xs_sf = torch.empty((2 * T, H // 128), dtype=torch.uint8, device=x.device)
        xs_fp8[:T] = x_fp8
        xs_sf[:T] = x_sf
    else:
        xs16 = torch.empty((2 * T, H), dtype=torch.bfloat16, device=x.device)
        xs16[:T] = x16
    transport.begin("y")

    def before(i: int) -> None:
        n = rows[i].numel()
        transport.wait("x", i, i + 1)
        if qat:
            fp8, sf = _split(rail.chunk("x", i, "recv"), n, (H, torch.uint8), (H // 128, torch.uint8))
            xs_fp8[T + offsets[i] : T + offsets[i + 1]].view(torch.uint8).copy_(fp8)
            xs_sf[T + offsets[i] : T + offsets[i + 1]].copy_(sf)
        else:
            (rows16,) = _split(rail.chunk("x", i, "recv"), n, (H * 2, torch.bfloat16))
            xs16[T + offsets[i] : T + offsets[i + 1]] = rows16

    def after(i: int, y_chunk: torch.Tensor) -> None:
        n = rows[i].numel()
        rail.chunk("y", i, "send")[: n * H * 2].view(torch.bfloat16).view(n, H).copy_(y_chunk[n:])
        transport.send("y", i, n * H * 2)

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
    partial = [rail.chunk("y", i, "recv")[: r.numel() * H * 2].view(torch.bfloat16).view(-1, H) for i, r in enumerate(rows)]
    y = y.index_add(0, order, torch.cat(partial))
    return y[:num_tokens].to(x.dtype), vids, foreign, plan_id, weights


@rail_llep_forward_op.register_fake
def _rail_llep_forward_fake(
    x, topk_idx, topk_weights, gate_up_proj, down_proj, rail_key, activation_clamp=None, rounding_seed=None,
    rounding_salts=None, expert_offset=0,
):
    T = _RAILS[rail_key].max_tokens
    state = _RAILS[rail_key].state
    return (
        torch.empty_like(x),
        topk_idx.new_empty((2 * T, topk_idx.shape[1]), dtype=torch.int64),
        topk_idx.new_empty((state.group.size(), state.options.max_foreign), dtype=torch.int32),
        torch.empty(1, dtype=torch.int64),
        topk_weights.new_empty((2 * T, topk_weights.shape[1]), dtype=torch.float32),
    )


@torch.library.custom_op("prime_rl::rail_llep_backward", mutates_args=())
def rail_llep_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
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
    num_tokens, T, H, K = x.shape[0], rail.max_tokens, rail.hidden, rail.top_k
    x, dy = _pad(x, T), _pad(dy, T)
    rows, order, calls, offsets = rail.layout(T)
    x16 = x.to(torch.bfloat16).contiguous()
    dy16 = dy.to(torch.bfloat16).contiguous()

    transport.begin("bwd")
    for i, r in enumerate(rows):
        xr, dyr = _split(rail.chunk("bwd", i, "send"), r.numel(), (H * 2, torch.bfloat16), (H * 2, torch.bfloat16))
        torch.index_select(x16, 0, r, out=xr)
        torch.index_select(dy16, 0, r, out=dyr)
        transport.send("bwd", i, r.numel() * H * 4)

    xs = torch.empty((2 * T, H), dtype=torch.bfloat16, device=x.device)
    dys = torch.empty_like(xs)
    xs[:T] = x16
    dys[:T] = dy16
    transport.begin("dx")

    def before(i: int) -> None:
        n = rows[i].numel()
        transport.wait("bwd", i, i + 1)
        xr, dyr = _split(rail.chunk("bwd", i, "recv"), n, (H * 2, torch.bfloat16), (H * 2, torch.bfloat16))
        xs[T + offsets[i] : T + offsets[i + 1]] = xr
        dys[T + offsets[i] : T + offsets[i + 1]] = dyr

    def after(i: int, dx_chunk: torch.Tensor, dtopk_chunk: torch.Tensor) -> None:
        n = rows[i].numel()
        dxr, dtr = _split(rail.chunk("dx", i, "send"), n, (H * 2, torch.bfloat16), (K * 4, torch.float32))
        dxr.copy_(dx_chunk[n:])
        dtr.copy_(dtopk_chunk[n:])
        transport.send("dx", i, n * (H * 2 + K * 4))

    if rail.precision == "mxfp8_qat":
        gate_up, down = cached_qat_rounded_weights(gate_up_proj, down_proj, rounding_seed, rounding_salts, expert_offset)
    else:
        gate_up, down = gate_up_proj, down_proj
    dx, dtopk, dw1, dw2 = state.backward(
        dys, xs, vids, foreign, plan_id, weights, gate_up, down, activation_clamp, _dw_dtype(gate_up_proj),
        chunks=calls, before_chunk=before, after_chunk=after,
    )
    transport.wait("dx")
    dx_parts, dtopk_parts = [], []
    for i, r in enumerate(rows):
        dxr, dtr = _split(rail.chunk("dx", i, "recv"), r.numel(), (H * 2, torch.bfloat16), (K * 4, torch.float32))
        dx_parts.append(dxr)
        dtopk_parts.append(dtr)
    dx = dx[:T].index_add(0, order, torch.cat(dx_parts))[:num_tokens]
    dtopk = dtopk[:T].index_add(0, order, torch.cat(dtopk_parts))[:num_tokens]
    return dx.to(x.dtype), dtopk, dw1, dw2


@rail_llep_backward_op.register_fake
def _rail_llep_backward_fake(
    dy, x, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rail_key, activation_clamp=None,
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
    _, vids, foreign, plan_id, weights = output
    ctx.save_for_backward(x, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rounding_seed, rounding_salts)
    ctx.rail_key, ctx.activation_clamp, ctx.expert_offset = rail_key, activation_clamp, offset
    ctx.topk_dtype = topk_weights.dtype


def _backward(ctx, grad_y, *_):
    x, vids, foreign, plan_id, weights, gate_up_proj, down_proj, rounding_seed, rounding_salts = ctx.saved_tensors
    dx, dtopk, dw1, dw2 = torch.ops.prime_rl.rail_llep_backward(
        grad_y, x, vids, foreign, plan_id, weights, gate_up_proj, down_proj, ctx.rail_key, ctx.activation_clamp,
        rounding_seed, rounding_salts, ctx.expert_offset,
    )
    return dx, None, dtopk.to(ctx.topk_dtype), dw1, dw2, None, None, None, None, None


rail_llep_forward_op.register_autograd(_backward, setup_context=_setup_context)


class RailLLEPMegaMoEExpertCompute(LLEPMegaMoEExpertCompute):
    """LLEP Mega MoE (`LLEPMegaMoEExpertCompute`) over an EP group of two nodes; see the module docstring.
    ``options.chunk_tokens`` counts the tokens of a node call (this rank's and its rail peer's)."""

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
        num_chunks = -(-2 * max_tokens_per_rank // (options.chunk_tokens or 2 * max_tokens_per_rank))
        super().__init__(
            num_experts // 2, top_k, hidden, intermediate_hidden, node_group, 2 * max_tokens_per_rank, num_sms,
            options.model_copy(update={"chunk_tokens": 2 * -(-max_tokens_per_rank // num_chunks)}), precision,
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
