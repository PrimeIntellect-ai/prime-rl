"""MoE layers as prime-mega-moe's fused Mega MoE kernels (SM90, NVLink): BF16 (routed + shared experts, the
backward recomputes the forward) or FP8 (routed experts, the backward reads the forward's kept rows).

One kernel per direction runs the layer: it pulls each rank's tokens through NVLink symmetric memory, runs the
local experts' clamped SwiGLU and combines the weighted outputs back.

prime-mega-moe is a DeepGEMM fork whose package is also named `deep_gemm`. The trainer's FP8 linears
need the upstream `deep_gemm`, so the fork is imported as `prime_mega_moe` (its `deep_gemm` package
directory on the path under that name).
"""

from functools import partial

import torch
import triton
import triton.language as tl
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.moe import GroupedExperts

_dispatchers: dict[int, "MegaMoETokenDispatcher"] = {}
# Whether the coming backward is the last micro-batch's of the step (see `set_expert_wgrad_final_micro_batch`)
_final_micro_batch = True


def set_expert_wgrad_final_micro_batch(final: bool) -> None:
    """Call before each micro-batch's backward when FP8 Mega MoE layers may hold their expert weight gradients
    (`fused_wgrad_micro_batches > 1`): a layer holds a backward's operands only while this is false, and the backward
    of the final micro-batch adds any held gradient into the fp32 accumulators before FSDP reduces them. Holding is
    only valid while those accumulators persist across micro-batches (FSDP does not reduce them before the final
    micro-batch); a held gradient whose accumulator was reduced in between raises. Defaults to true (never hold)."""
    global _final_micro_batch
    _final_micro_batch = final


def flush_pending_expert_wgrads() -> int:
    """Add every held expert weight gradient into its accumulator now; returns how many layers held any. The
    trainer's step should find none (the final micro-batch's backward adds them)."""
    flushed = 0
    for dispatcher in _dispatchers.values():
        if isinstance(dispatcher, MegaMoEFP8TokenDispatcher) and dispatcher.held_wgrads:
            dispatcher.flush_held_wgrads()
            flushed += 1
    return flushed


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class MegaMoETokenDispatcher:
    """Runs a whole MoE layer, shared expert included, as one Mega MoE kernel per direction.

    Every layer shares one symmetric buffer: each kernel call first loads its own tokens and routing
    into it, and the backward reloads the forward's.
    """

    fuses_shared_expert = True

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int,
        activation_clamp: float,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        num_sms: int | None = None,
    ) -> None:
        import prime_mega_moe

        self.kernels = prime_mega_moe
        if num_sms is not None:
            prime_mega_moe.set_num_sms(num_sms)
        self.activation_clamp = activation_clamp
        self.max_tokens_per_rank = max_tokens_per_rank
        # Created up front: inside the first checkpointed forward, its collectives would desync recompute.
        self.buffer = _get_buffer(
            group, num_experts, max_tokens_per_rank, top_k, hidden_size, intermediate_size, num_shared_experts
        )
        self._id = id(self)
        _dispatchers[self._id] = self

    def synchronize(self) -> None:
        return None

    @torch.compiler.disable()
    def run_fused(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: GroupedExperts,
        shared_expert: nn.Module | None,
    ) -> torch.Tensor:
        assert x.shape[0] <= self.max_tokens_per_rank, (
            f"{x.shape[0]} tokens exceed the Mega MoE buffer's {self.max_tokens_per_rank} per rank"
        )
        assert experts.gate_up_proj is not None, "Mega MoE reads the packed [gate | up] expert weight"
        shared_l1 = shared_l2 = None
        if shared_expert is not None:
            shared_l1 = torch.cat((shared_expert.gate_proj.weight, shared_expert.up_proj.weight))
            shared_l2 = shared_expert.down_proj.weight
        return torch.ops.prime_rl.mega_moe(
            x.bfloat16().contiguous(),
            selected_experts_indices,
            top_scores.float(),
            _to_local(experts.gate_up_proj).bfloat16(),
            _to_local(experts.down_proj).bfloat16(),
            shared_l1,
            shared_l2,
            self._id,
        ).type_as(x)

    def load(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> None:
        num_tokens = x.shape[0]
        self.buffer.x[:num_tokens].copy_(x)
        self.buffer.topk_idx[:num_tokens].copy_(topk_idx)
        self.buffer.topk_weights[:num_tokens].copy_(topk_weights)


_buffers: dict[int, object] = {}


def _get_buffer(group: ProcessGroup, *shape: int):
    """A view for `shape` on the group's one symmetric buffer, allocated by the first layer."""
    import prime_mega_moe

    num_experts, max_tokens, top_k, hidden, intermediate, num_shared = shape
    buffer = prime_mega_moe.SymmBuffer(
        group,
        num_experts,
        max_tokens,
        top_k,
        hidden,
        intermediate,
        num_shared_experts=num_shared,
        mma_type="bf16xbf16",
        base=_buffers.get(id(group)),
    )
    _buffers.setdefault(id(group), buffer)
    return buffer


@torch.library.custom_op("prime_rl::mega_moe", mutates_args=())
def mega_moe(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    shared_l1: torch.Tensor | None,
    shared_l2: torch.Tensor | None,
    dispatcher_id: int,
) -> torch.Tensor:
    """One MoE layer's combined output: routed experts weighted by `topk_weights`, plus the shared expert."""
    dispatcher = _dispatchers[dispatcher_id]
    dispatcher.load(x, topk_idx, topk_weights)
    y = torch.empty_like(x)
    dispatcher.kernels.bf16_mega_moe(
        y,
        l1,
        l2,
        dispatcher.buffer,
        shared_l1_weights=shared_l1,
        shared_l2_weights=shared_l2,
        activation_clamp=dispatcher.activation_clamp,
    )
    return y


@mega_moe.register_fake
def _mega_moe_fake(x, topk_idx, topk_weights, l1, l2, shared_l1, shared_l2, dispatcher_id):
    return torch.empty_like(x)


def _mega_moe_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, topk_weights, l1, l2, shared_l1, shared_l2, dispatcher_id = inputs
    ctx.dispatcher_id = dispatcher_id
    ctx.has_shared = shared_l1 is not None
    ctx.save_for_backward(x, topk_idx, topk_weights, l1, l2, shared_l1, shared_l2)


def _mega_moe_backward(ctx, grad_y: torch.Tensor):
    x, topk_idx, topk_weights, l1, l2, shared_l1, shared_l2 = ctx.saved_tensors
    dispatcher = _dispatchers[ctx.dispatcher_id]
    dispatcher.load(x, topk_idx, topk_weights)
    grad_x = torch.empty_like(x)
    grad_l1, grad_l2 = torch.empty_like(l1), torch.empty_like(l2)
    grad_weights = torch.empty_like(topk_weights)
    grad_shared_l1 = torch.empty_like(shared_l1) if ctx.has_shared else None
    grad_shared_l2 = torch.empty_like(shared_l2) if ctx.has_shared else None
    dispatcher.kernels.bf16_mega_moe_backward(
        grad_x,
        grad_l1,
        grad_l2,
        grad_weights,
        grad_y.bfloat16().contiguous(),
        l1,
        l2,
        dispatcher.buffer,
        shared_l1_weights=shared_l1,
        shared_l2_weights=shared_l2,
        shared_dw1_weights=grad_shared_l1,
        shared_dw2_weights=grad_shared_l2,
        activation_clamp=dispatcher.activation_clamp,
    )
    return grad_x, None, grad_weights, grad_l1, grad_l2, grad_shared_l1, grad_shared_l2, None


mega_moe.register_autograd(_mega_moe_backward, setup_context=_mega_moe_setup_context)


class MegaMoEFP8TokenDispatcher:
    """Routed experts of a MoE layer as prime-mega-moe's SM90 FP8 Mega MoE kernels (the shared expert stays with
    the layer).

    Forward: one kernel dispatches FP8 tokens over NVLink, runs the experts' blockwise-FP8 clamped SwiGLU and combines
    the weighted outputs. It keeps each routed row's FP8 input, bf16 gate/up output, routing weight and source as the
    op's outputs, so activation checkpointing saves them. Backward: one kernel gathers `dy`, computes the data
    gradients from the kept rows (no forward recompute) and returns `dx` and the top-k weight gradients; DeepGEMM's
    K-grouped FP8 GEMMs then add the weight gradients into FSDP's fp32 accumulators.

    Pools hold `capacity_factor * tokens * top_k` routed rows per rank (plus a partial block per expert); a rank
    receiving more traps in the kernel.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation_clamp: float,
        group: ProcessGroup,
        max_tokens_per_rank: int,
        capacity_factor: float,
        num_sms: int | None = None,
        wgrad_tile_scales: bool = False,
        free_bf16_weights: bool = False,
        transposed_on_demand: bool = False,
        fused_wgrad_micro_batches: int = 1,
    ) -> None:
        import prime_mega_moe.mega.fp8 as fp8_kernels

        self.kernels = fp8_kernels
        self.activation_clamp = activation_clamp
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_local_experts = num_experts // group.size()
        self.capacity = fp8_kernels.pool_capacity(max_tokens_per_rank, top_k, self.num_local_experts, capacity_factor)
        self.num_sms = num_sms or torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.wgrad_tile_scales = wgrad_tile_scales
        self.free_bf16_weights = free_bf16_weights
        self.transposed_on_demand = transposed_on_demand
        assert 1 <= fused_wgrad_micro_batches <= (2 if wgrad_tile_scales else 4)
        self.fused_wgrad_micro_batches = fused_wgrad_micro_batches
        # Held backwards' weight-gradient operands (oldest first) and the accumulators they belong to
        self.held_wgrads: list[_PendingWgrad] = []
        # Created up front: inside the first checkpointed forward, its collectives would desync recompute.
        self.buffer = _get_fp8_buffer(group, num_experts, max_tokens_per_rank, top_k, hidden_size)
        self._experts: GroupedExperts | None = None
        self._id = id(self)
        _dispatchers[self._id] = self

    def synchronize(self) -> None:
        return None

    @torch.compiler.disable()
    def run(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        experts: GroupedExperts,
        *,
        score_before_experts: bool,
    ) -> torch.Tensor:
        assert not score_before_experts, "the FP8 Mega MoE kernel weights the experts' outputs"
        assert experts.gate_up_proj is not None, "the FP8 Mega MoE reads the packed [gate | up] expert weight"
        assert x.shape[0] <= self.max_tokens_per_rank, (
            f"{x.shape[0]} tokens exceed the Mega MoE buffer's {self.max_tokens_per_rank} per rank"
        )
        self._experts = experts
        topk_idx = selected_experts_indices.masked_fill(top_scores == 0, -1)
        outputs = torch.ops.prime_rl.mega_moe_fp8(x.bfloat16().contiguous(), topk_idx, top_scores.float(), self._id)
        return outputs[0].type_as(x)

    def quantized_weights(self) -> list[torch.Tensor]:
        """The experts' FP8 weights `[w13, w13 scales, w2, w2 scales, w13^T, its scales, w2^T, its scales]`,
        quantized once per optimizer step (shared with the prime-kernels expert path's cache)."""
        from prime_rl.trainer.models.layers.expert_compute import FusedSwigluExpertCompute, _quantized_expert_weights

        weights = _quantized_expert_weights(FusedSwigluExpertCompute, self._experts)
        if self.transposed_on_demand and weights[4] is not None:
            # The cached list is shared for the step: drop its transposed copies, `transposed_weights` rebuilds them.
            weights[4:] = [None] * 4
        if self.free_bf16_weights:
            # Only the quantization reads the bf16 copy. FSDP keeps the parameters unsharded for the rest of
            # the step, so its next reshard finds the storage already freed and its next unshard reallocates it.
            # Without FSDP the parameters are the master weights and stay.
            for fsdp_param in getattr(self._experts, "fsdp_params", {}).values():
                fsdp_param.free_unsharded_param()
        return weights

    def transposed_weights(self) -> list[torch.Tensor]:
        """`[w13^T, its scales, w2^T, its scales]` for the backward: from the cache, or with `transposed_on_demand`
        transposed from the forward copies (128 x 128 blocks, so the same values and scales as the cached copy)."""
        w13_q, w13_sf, w2_q, w2_sf, *transposed = self.quantized_weights()
        if transposed[0] is not None:
            return transposed
        return [_transpose_last2(w13_q), _transpose_last2(w13_sf), _transpose_last2(w2_q), _transpose_last2(w2_sf)]

    def load(self, x: torch.Tensor | None, topk_idx: torch.Tensor, topk_weights: torch.Tensor | None) -> None:
        from prime_rl.trainer.models.kernels.fp8_utils import per_token_cast_to_fp8_triton

        num_tokens = topk_idx.shape[0]
        if x is not None:
            # 1x128 FP8 with power-of-two scales, DeepEP's FP8 dispatch format
            x_q, x_sf = per_token_cast_to_fp8_triton(x, use_ue8m0=True)
            self.buffer.x[:num_tokens].copy_(x_q)
            self.buffer.x_sf[:num_tokens].copy_(x_sf)
        self.buffer.topk_idx[:num_tokens].copy_(topk_idx)
        if topk_weights is not None:
            self.buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def flush_held_wgrads(self) -> None:
        held, self.held_wgrads = self.held_wgrads, []
        for h in held:
            dw13, dw2 = h.accumulators()
            self.kernels.fp8_mega_moe_weight_grads_add(h.operands, dw13, dw2, self.wgrad_tile_scales)

    def l2_operand(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The L2 operand pool (FP8 `h` and its scales), shared by every layer of the group."""
        key = (id(self.buffer), self.capacity, self.intermediate_size)
        if key not in _l2_operands:
            I, C = self.intermediate_size, self.capacity
            _l2_operands[key] = (
                torch.empty(C, I, dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty(I // 64, C, dtype=torch.float32, device="cuda"),
            )
        return _l2_operands[key]


@triton.jit
def _transpose_bytes_kernel(x_ptr, y_ptr, R, C, BLOCK: tl.constexpr):
    """One BLOCK x BLOCK tile of ``x [E, R, C]`` (bytes) into ``y [E, C, R]``."""
    e = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + e * R * C + rows[:, None] * C + cols[None, :])
    tl.store(y_ptr + e * R * C + cols[:, None] * R + rows[None, :], tl.trans(v))


def _transpose_last2(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float8_e4m3fn:
        return t.transpose(1, 2).contiguous()
    E, R, C = t.shape
    block = 128
    assert R % block == 0 and C % block == 0 and t.is_contiguous()
    out = torch.empty(E, C, R, dtype=torch.uint8, device=t.device)
    _transpose_bytes_kernel[(E, R // block, C // block)](t.view(torch.uint8), out, R, C, BLOCK=block, num_warps=8)
    return out.view(torch.float8_e4m3fn)


class _PendingWgrad:
    """A held backward's weight-gradient operands, with the FSDP parameters whose fp32 accumulators take them."""

    def __init__(self, operands, fsdp_params: dict):
        self.operands, self.fsdp_params = operands, fsdp_params
        # FSDP's accumulator objects when held: a reduce in between replaces them
        self._raw = tuple(fsdp_params[name].unsharded_accumulated_grad for name in ("gate_up_proj", "down_proj"))

    def accumulators(self) -> tuple[torch.Tensor, torch.Tensor]:
        from prime_rl.trainer.models.layers.expert_compute import _fp32_grad_accumulator

        raw = tuple(self.fsdp_params[name].unsharded_accumulated_grad for name in ("gate_up_proj", "down_proj"))
        if any(a is not b for a, b in zip(raw, self._raw)):
            raise RuntimeError(
                "An expert weight gradient was held past an FSDP gradient reduction: mark the step's last "
                "micro-batch with set_expert_wgrad_final_micro_batch(True) before its backward."
            )
        return _fp32_grad_accumulator(self.fsdp_params["gate_up_proj"]), _fp32_grad_accumulator(
            self.fsdp_params["down_proj"]
        )


_fp8_buffers: dict[int, object] = {}
# Forward-side events: the weight-gradient operand of `x` is ready, by its data pointer
_x_t_ready: dict[int, torch.cuda.Event] = {}
_side_streams: dict[int, torch.cuda.Stream] = {}


def _side_stream() -> torch.cuda.Stream:
    """This device's stream for the column quantizations that overlap other work."""
    device = torch.cuda.current_device()
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream()
    return _side_streams[device]


_l2_operands: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_fp8_buffer(group: ProcessGroup, num_experts: int, max_tokens: int, top_k: int, hidden: int):
    """The group's one FP8 symmetric buffer, allocated by the first layer."""
    import prime_mega_moe.mega.fp8 as fp8_kernels

    key = id(group)
    if key not in _fp8_buffers:
        _fp8_buffers[key] = fp8_kernels.FP8SymmBuffer(group, num_experts, max_tokens, top_k, hidden)
    buffer = _fp8_buffers[key]
    assert (buffer.num_experts, buffer.num_max_tokens_per_rank, buffer.num_topk, buffer.hidden) == (
        num_experts,
        max_tokens,
        top_k,
        hidden,
    ), "every Mega MoE layer of a group must share one buffer shape"
    return buffer


@torch.library.custom_op("prime_rl::mega_moe_fp8", mutates_args=())
def mega_moe_fp8(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, dispatcher_id: int
) -> list[torch.Tensor]:
    """Routed experts of one MoE layer, combined and weighted by `topk_weights`, then what the backward reads: the
    FP8 inputs quantized per column for the gate/up weight gradient (and their scales), the bf16 gate/up, routing
    weights, sources and per-expert row counts of the pool rows."""
    dispatcher = _dispatchers[dispatcher_id]
    w13_q, w13_sf, w2_q, w2_sf, *_ = dispatcher.quantized_weights()
    h, h_sf = dispatcher.l2_operand()
    pools = dispatcher.kernels.FP8Pools.allocate(
        dispatcher.capacity, dispatcher.hidden_size, dispatcher.intermediate_size, dispatcher.num_local_experts, h, h_sf
    )
    dispatcher.load(x, topk_idx, topk_weights)
    y = torch.empty_like(x)
    dispatcher.kernels.fp8_mega_moe(
        y,
        (w13_q, w13_sf),
        (w2_q, w2_sf),
        pools,
        dispatcher.buffer,
        activation_clamp=dispatcher.activation_clamp,
        fast_math=True,
        num_sms=dispatcher.num_sms,
    )
    # The weight-gradient operand of `x`, on a side stream beside the next layers; the backward waits for it.
    # Allocated on the compute stream, so the backward's free returns it to the pool every activation draws from.
    C, H = pools.x.shape
    x_t = torch.empty(C * H, dtype=torch.float8_e4m3fn, device=x.device)
    x_t_sf = torch.empty(
        C // 128, H // 128 if dispatcher.wgrad_tile_scales else H, dtype=torch.float32, device=x.device
    )
    side_stream = _side_stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        dispatcher.kernels.column_quant(pools.x, pools.x_sf, pools.counts, x_t, x_t_sf)
        _x_t_ready[x_t.data_ptr()] = side_stream.record_event()
    for t in (pools.x, pools.x_sf, pools.counts, x_t, x_t_sf):
        t.record_stream(side_stream)
    return [y, x_t, x_t_sf, pools.z, pools.weights, pools.meta, pools.counts]


@mega_moe_fp8.register_fake
def _mega_moe_fp8_fake(x, topk_idx, topk_weights, dispatcher_id):
    dispatcher = _dispatchers[dispatcher_id]
    C, H, I = dispatcher.capacity, dispatcher.hidden_size, dispatcher.intermediate_size
    return [
        torch.empty_like(x),
        x.new_empty(C * H, dtype=torch.float8_e4m3fn),
        x.new_empty(C // 128, H // 128 if dispatcher.wgrad_tile_scales else H, dtype=torch.float32),
        x.new_empty(C, 2 * I),
        x.new_empty(C, dtype=torch.float32),
        x.new_empty(C, 3, dtype=torch.int32),
        x.new_empty(dispatcher.num_local_experts, dtype=torch.int32),
    ]


def _mega_moe_fp8_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, topk_weights, dispatcher_id = inputs
    ctx.dispatcher_id = dispatcher_id
    ctx.weights_dtype = topk_weights.dtype
    ctx.save_for_backward(topk_idx, *output[1:])
    # Only `y` carries gradient; the pools are returned only to be saved.
    ctx.set_materialize_grads(False)


def _mega_moe_fp8_backward(ctx, grads):
    from prime_rl.trainer.models.layers import expert_compute
    from prime_rl.trainer.models.layers.expert_compute import _fp32_grad_accumulator

    grad_y = grads[0]
    topk_idx, x_t, x_t_sf, z, weights, meta, counts = ctx.saved_tensors
    torch.cuda.current_stream().wait_event(_x_t_ready.pop(x_t.data_ptr()))
    dispatcher = _dispatchers[ctx.dispatcher_id]
    kernels = dispatcher.kernels
    w13_t, w13_t_sf, w2_t, w2_t_sf = dispatcher.transposed_weights()
    # The backward kernel reads every pool but `x`
    pools = kernels.FP8Pools(None, None, z, weights, meta, counts)
    bufs = kernels.FP8BackwardBuffers(dispatcher.capacity, dispatcher.hidden_size, dispatcher.intermediate_size)
    num_tokens = topk_idx.shape[0]
    dispatcher.buffer.dy[:num_tokens].copy_(grad_y)
    dispatcher.load(None, topk_idx, None)
    grad_x = torch.empty_like(grad_y, dtype=torch.bfloat16)
    grad_weights = torch.empty(topk_idx.shape, dtype=torch.float32, device=grad_y.device)
    kernels.fp8_mega_moe_backward(
        grad_x,
        grad_weights,
        (w13_t, w13_t_sf),
        (w2_t, w2_t_sf),
        pools,
        bufs,
        dispatcher.buffer,
        activation_clamp=dispatcher.activation_clamp,
        fast_math=True,
        num_sms=dispatcher.num_sms,
    )
    experts = dispatcher._experts
    fsdp_params = getattr(experts, "fsdp_params", None)
    if fsdp_params is not None:
        dw13 = _fp32_grad_accumulator(fsdp_params["gate_up_proj"])
        dw2 = _fp32_grad_accumulator(fsdp_params["down_proj"])
        weight_grads = partial(
            _expert_weight_grads, dispatcher, counts, bufs, (x_t, x_t_sf), fsdp_params, dw13, dw2, _final_micro_batch
        )
        if expert_compute._deferred_weight_grads is not None:
            expert_compute._deferred_weight_grads.append(weight_grads)
        else:
            weight_grads()
    else:
        # Without FSDP the experts' plain parameters take the gradients.
        params = (experts.gate_up_proj, experts.down_proj)
        accumulators = [torch.zeros(_to_local(p).shape, dtype=torch.float32, device=grad_y.device) for p in params]
        kernels.fp8_mega_moe_weight_grads(
            counts, bufs, (x_t, x_t_sf), *accumulators, _side_stream(), dispatcher.wgrad_tile_scales
        )
        for param, accumulator in zip(params, accumulators):
            grad = accumulator.to(param.dtype)
            param.grad = grad if param.grad is None else param.grad + grad
    return grad_x, None, grad_weights.to(ctx.weights_dtype), None


def _expert_weight_grads(dispatcher, counts, bufs, x_t, fsdp_params, dw13, dw2, final_micro_batch: bool) -> None:
    """Add one backward's expert weight gradients into the fp32 accumulators, or hold its operands (see
    `fused_wgrad_micro_batches`). Runs in the backward's micro-batch order, inside it or deferred after it."""
    held = dispatcher.held_wgrads
    for h in held:
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(h.accumulators(), (dw13, dw2)))
    # Hold this backward's operands until `fused_wgrad_micro_batches` are held or the step's last micro-batch: that
    # backward adds them all with one launch per weight
    hold = len(held) < dispatcher.fused_wgrad_micro_batches - 1 and not final_micro_batch
    # Only fusing needs prime-mega-moe's `pending` / `defer` arguments; without them any build works.
    fusion = (
        {}
        if dispatcher.fused_wgrad_micro_batches == 1
        else {"pending": None if hold else [h.operands for h in held], "defer": hold}
    )
    operands = dispatcher.kernels.fp8_mega_moe_weight_grads(
        counts, bufs, x_t, dw13, dw2, _side_stream(), dispatcher.wgrad_tile_scales, **fusion
    )
    if hold:
        held.append(_PendingWgrad(operands, fsdp_params))
    else:
        held.clear()


mega_moe_fp8.register_autograd(_mega_moe_fp8_backward, setup_context=_mega_moe_fp8_setup_context)
