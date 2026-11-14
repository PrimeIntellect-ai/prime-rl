from collections.abc import Callable, Sequence
from functools import cache, wraps
from pathlib import Path
from typing import Any


def local_expert_destinations(
    physical_to_logical: Sequence[int],
    global_to_local: Sequence[int],
    num_logical_experts: int,
) -> list[list[int]]:
    """Map each checkpoint expert to its current physical slots on this rank."""
    destinations: list[list[int]] = [[] for _ in range(num_logical_experts)]
    for physical_id, (logical_id, local_id) in enumerate(zip(physical_to_logical, global_to_local, strict=True)):
        if local_id < 0:
            continue
        if not 0 <= logical_id < num_logical_experts:
            raise ValueError(f"Local physical expert {physical_id} has invalid logical expert {logical_id}")
        destinations[logical_id].append(physical_id)
    return destinations


def remap_expert_weight_loader(loader: Callable, destinations: Sequence[Sequence[int]]) -> Callable:
    """Load each logical expert into all of its current local replicas."""

    # Preserve vLLM's online_process_loader marker to avoid another layerwise wrapper.
    @wraps(loader)
    def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        # Checkpoint loaders enumerate the initial redundant slots after the logical
        # experts. Their weights have already been sent to every current replica.
        if expert_id >= len(destinations):
            return False if return_success else None
        if expert_id < 0:
            raise ValueError(f"Invalid checkpoint expert {expert_id}")
        success = False
        for physical_id in destinations[expert_id]:
            loaded = loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_id,
                return_success=True,
            )
            success = loaded or success
        return success if return_success else None

    return load


def current_expert_placement(eplb_state: Any, model_config: Any) -> tuple[tuple, list[tuple[Any, list[list[int]]]]]:
    """EPLB's current placement: a fingerprint of it (every layer's physical-to-logical map) and, per MoE layer, its
    routed-experts module with each logical expert's current local slots."""
    state = eplb_state.model_states[model_config.compute_hash()]
    placements = state.physical_to_logical_map.cpu().tolist()
    layers = []
    for layer, placement in zip(state.model.moe_layers, placements, strict=True):
        experts = layer.routed_experts
        expert_map = experts._expert_map
        global_to_local = expert_map.cpu().tolist() if expert_map is not None else list(range(len(placement)))
        layers.append((experts, local_expert_destinations(placement, global_to_local, state.model.num_routed_experts)))
    return tuple(tuple(placement) for placement in placements), layers


EXPERT_LOAD_FILE = "expert_load.pt"
"""Per-layer expert token counts the trainer writes into a weight update's step dir (``[moe_layers, experts]``)."""


def trainer_placement(
    policy: Any, load: Any, num_physical_experts: int, num_groups: int, num_nodes: int, num_ranks: int
) -> Any:
    """The physical-to-logical map for ``load`` (``[moe_layers, logical_experts]`` token counts). It depends on the
    counts alone (not on the current placement), so every rank of an EP group, across servers and restarts, derives
    the same one from the same counts."""
    if num_ranks % num_nodes or num_groups % num_nodes:
        num_nodes = 1
    return policy.rebalance_experts(load, num_physical_experts, num_groups, num_nodes, num_ranks, None)


def install_trainer_placement(
    eplb_state: Any,
    model_config: Any,
    weight_dir: str | None,
    node_count: Callable[[], int] = lambda: ep_node_count(),
    all_ranks: Callable[[bool], bool] = lambda present: ep_all(present),
) -> int | None:
    """If the trainer published expert counts with this update (``EXPERT_LOAD_FILE`` in ``weight_dir``), place the
    experts by them: commit EPLB's maps (physical-to-logical, logical-to-physical, replica counts) without moving any
    weights, since the update about to run writes every expert into the slots the new maps give it. Returns the
    number of physical slots whose expert changed, or None when there was nothing to install. Call inside the
    weight-update pause, before the weights are written, on every worker of the engine (``node_count`` runs a
    collective over the EP group)."""
    import torch
    from vllm.distributed.eplb.eplb_state import _commit_eplb_maps
    from vllm.distributed.parallel_state import get_ep_group

    from prime_rl.transports.weights.base import marker_exists

    path = Path(weight_dir) / EXPERT_LOAD_FILE if weight_dir is not None else None
    # Every rank of the EP group installs the same maps or none does: a rank missing the file on a stale NFS lookup
    # keeps the current placement for everyone until the next update
    if not all_ranks(path is not None and marker_exists(path)):
        return None
    state = eplb_state.model_states[model_config.compute_hash()]
    model = state.model
    load = torch.load(path, map_location="cpu", weights_only=True)
    expected = (model.num_moe_layers, model.num_logical_experts)
    if tuple(load.shape) != expected:
        raise ValueError(f"Trainer expert counts have shape {tuple(load.shape)}, the served model needs {expected}")
    num_ranks = get_ep_group().world_size
    new_map = trainer_placement(
        eplb_state.policy, load, model.num_physical_experts, model.num_expert_groups, node_count(), num_ranks
    )
    old_map = state.physical_to_logical_map.cpu()
    moved = int((new_map != old_map).sum())
    _commit_eplb_maps(state, new_physical_to_logical_map=new_map.to(old_map.dtype))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return moved


def ep_all(flag: bool) -> bool:
    """Whether ``flag`` holds on every rank of this rank's EP group."""
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_ep_group

    value = torch.tensor([int(flag)])
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=get_ep_group().cpu_group)
    return bool(value.item())


@cache
def ep_node_count() -> int:
    """Hosts in this rank's EP group: the same on every rank, unlike vLLM's per-engine node count when one EP group
    spans several servers."""
    import socket

    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_ep_group

    group = get_ep_group()
    hosts: list[str | None] = [None] * group.world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group.cpu_group)
    return len(set(hosts))
