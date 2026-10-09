"""MoE layers as prime-mega-moe's fused BF16 Mega MoE kernel (SM90, NVLink).

One kernel per direction runs the whole layer: it pulls each rank's tokens through NVLink symmetric
memory, runs the local experts' clamped SwiGLU and the shared expert, and combines the weighted
outputs back. The backward kernel recomputes the forward from the same inputs, so the forward keeps
nothing but its inputs.

prime-mega-moe is a DeepGEMM fork whose package is also named `deep_gemm`. The trainer's FP8 linears
need the upstream `deep_gemm`, so the fork is imported as `prime_mega_moe` (its `deep_gemm` package
directory on the path under that name).
"""

import torch
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.moe import GroupedExperts

_dispatchers: dict[int, "MegaMoETokenDispatcher"] = {}


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
