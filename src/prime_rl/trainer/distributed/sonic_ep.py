import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

_managers: dict[tuple[str, int | None], object] = {}
_symm_mem_enabled_groups: set[str] = set()


def _get_group_name(group: ProcessGroup) -> str:
    group_name = getattr(group, "group_name", None)
    if group_name is None and group is dist.group.WORLD:
        group_name = "0"
    if group_name is None:
        raise RuntimeError("Cannot resolve symmetric-memory group name for sonic EP.")
    return group_name


def _enable_symmetric_memory(group_name: str) -> None:
    if group_name in _symm_mem_enabled_groups:
        return

    from torch.distributed import _symmetric_memory as symm_mem

    symm_mem.enable_symm_mem_for_group(group_name)
    _symm_mem_enabled_groups.add(group_name)


def _get_manager(group: ProcessGroup, device: torch.device):
    from sonicmoe.ep import SymmMemManager

    group_name = _get_group_name(group)
    _enable_symmetric_memory(group_name)

    key = (group_name, device.index)
    manager = _managers.get(key)
    if manager is None:
        manager = SymmMemManager(group, device)
        _managers[key] = manager
    return manager


def _to_sonic_ep_w1(w1w3: torch.Tensor) -> torch.Tensor:
    return w1w3.contiguous().permute(2, 1, 0)


def _to_sonic_ep_w2(w2: torch.Tensor) -> torch.Tensor:
    return w2.permute(2, 0, 1).contiguous().permute(0, 2, 1)


class _SonicEPForwardOnly(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
        w1w3: torch.Tensor,
        w2: torch.Tensor,
        num_experts: int,
        group: ProcessGroup,
    ) -> torch.Tensor:
        from sonicmoe.enums import ActivationType
        from sonicmoe.ep import moe_ep_general_routing_forward

        ctx.requires_backward = x.requires_grad or top_scores.requires_grad or w1w3.requires_grad or w2.requires_grad

        x_dtype = x.dtype
        manager = _get_manager(group, x.device)
        out = moe_ep_general_routing_forward(
            x.bfloat16(),
            selected_experts_indices,
            top_scores,
            _to_sonic_ep_w1(w1w3.bfloat16()),
            None,
            _to_sonic_ep_w2(w2.bfloat16()),
            None,
            E=num_experts,
            mgr=manager,
            activation_type=ActivationType.SWIGLU,
            is_inference_mode_enabled=False,
            concat_layout=False,
        )
        return out.to(dtype=x_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.requires_backward:
            raise RuntimeError(
                "sonic-moe EP only provides a forward kernel at the pinned revision; "
                "model.ep_comm_backend='sonic' is not supported for SFT/RL backward yet."
            )
        return None, None, None, None, None, None, None


def forward(
    x: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    top_scores: torch.Tensor,
    w1w3: torch.Tensor,
    w2: torch.Tensor,
    *,
    num_experts: int,
    group: ProcessGroup,
) -> torch.Tensor:
    return _SonicEPForwardOnly.apply(x, selected_experts_indices, top_scores, w1w3, w2, num_experts, group)


__all__ = ["forward"]
