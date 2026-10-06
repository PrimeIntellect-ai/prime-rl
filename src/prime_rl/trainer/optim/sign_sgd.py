from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer


@triton.jit
def _sign_sgd_kernel(
    param_ptr, grad_ptr, sumsq_ptr, numel, lr, lr_weight_decay, BLOCK: tl.constexpr, ACCUMULATE_NORM: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    # Like torch.sign on CUDA: zero and NaN both map to 0.
    sign = tl.where(grad > 0, 1.0, tl.where(grad < 0, -1.0, 0.0))
    # `param * (1 - lr * wd)` would round the tiny decay away in fp32; subtract it instead.
    param = param - lr_weight_decay * param - lr * sign
    tl.store(param_ptr + offsets, param.to(param_ptr.dtype.element_ty), mask=mask)
    if ACCUMULATE_NORM:
        tl.atomic_add(sumsq_ptr, tl.sum(grad * grad))


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class SignSGD(Optimizer):
    """Sign-based SGD optimizer with minimal memory footprint.

    This optimizer uses the sign of gradients instead of storing momentum and variance,
    making it equivalent to AdamW with beta1=0 and beta2=0 (resetting optimizer state each step).

    Mathematical equivalence:
        AdamW: W = W - lr * m_t / sqrt(v_t + eps)
        With beta1=0, beta2=0: m_t = g_t, v_t = g_t^2
        Simplified: W = W - lr * g_t / sqrt(g_t^2 + eps)
        Ignoring eps: W = W - lr * sign(g_t)

    On CUDA the decay and update run as one fused pass over each parameter, which matters when a
    rank holds billions of expert parameters.
    """

    # Rescaling every gradient by a positive factor leaves the update unchanged, so gradient
    # clipping only needs the norm, not the rescale.
    scale_invariant = True

    BLOCK = 4096

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._armed = False
        self._sumsq: dict = {}

    def enable_step_in_backward(self) -> None:
        """Update each parameter as soon as its reduced gradient is ready during backward.

        The optimizer is stateless and scale invariant, so a parameter's update needs nothing but its
        own final gradient: it runs in a post-accumulate-grad hook, overlapping the rest of backward,
        and frees the gradient right away. Only backwards between `arm(True)` and `grad_norm()` step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(lambda p, group=group: self._step_in_backward(p, group))

    def arm(self, armed: bool) -> None:
        """Whether the next backward applies the update (its last micro-step, when not validating)."""
        self._armed = armed
        self._sumsq = {}

    def _step_in_backward(self, p: torch.Tensor, group: dict) -> None:
        if not self._armed:
            return
        param, grad = _local(p), _local(p.grad)
        assert param.is_cuda and param.is_contiguous() and grad.is_contiguous()
        mesh = p.device_mesh if isinstance(p, DTensor) else None
        if mesh is not None:
            assert all(isinstance(pl, Shard) for pl in p.placements), "the norm sums disjoint shards only"
        if mesh not in self._sumsq:
            self._sumsq[mesh] = torch.zeros((), dtype=torch.float32, device=param.device)
        if param.numel():
            lr, wd = group["lr"], group["weight_decay"]
            _sign_sgd_kernel[(triton.cdiv(param.numel(), self.BLOCK),)](
                param, grad, self._sumsq[mesh], param.numel(), lr, lr * wd, BLOCK=self.BLOCK, ACCUMULATE_NORM=True
            )
        p.grad = None

    def grad_norm(self) -> torch.Tensor:
        """Global norm of the gradients the armed backward stepped on; disarms."""
        by_group = defaultdict(list)
        for mesh, sumsq in self._sumsq.items():
            by_group[None if mesh is None else mesh.get_group()].append(sumsq)
        total = torch.zeros((), dtype=torch.float32, device="cuda")
        for pg, parts in by_group.items():
            part = torch.stack(parts).sum()
            if pg is not None:
                dist.all_reduce(part, group=pg)
            total += part
        self.arm(False)
        return total.sqrt()

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                param, grad = _local(p), _local(p.grad)
                if param.is_cuda and param.is_contiguous() and grad.is_contiguous():
                    numel = param.numel()
                    if numel:
                        _sign_sgd_kernel[(triton.cdiv(numel, self.BLOCK),)](
                            param, grad, grad, numel, lr, lr * weight_decay, BLOCK=self.BLOCK, ACCUMULATE_NORM=False
                        )
                    continue

                sign_grad = torch.sign(p.grad)

                if weight_decay > 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                p.add_(sign_grad, alpha=-lr)

        return loss
