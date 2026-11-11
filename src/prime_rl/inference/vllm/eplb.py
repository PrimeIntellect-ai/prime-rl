from collections.abc import Callable, Sequence
from functools import wraps
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
    routed-experts module with each logical expert's current local slots. Drains async moves first, so the placement
    is the one the weights sit in."""
    eplb_state.drain_async()
    state = eplb_state.model_states[model_config.compute_hash()]
    placements = state.physical_to_logical_map.cpu().tolist()
    layers = []
    for layer, placement in zip(state.model.moe_layers, placements, strict=True):
        experts = layer.routed_experts
        expert_map = experts._expert_map
        global_to_local = expert_map.cpu().tolist() if expert_map is not None else list(range(len(placement)))
        layers.append((experts, local_expert_destinations(placement, global_to_local, state.model.num_routed_experts)))
    return tuple(tuple(placement) for placement in placements), layers


def moved_slots(fingerprint: tuple, num_routed_experts: int) -> int:
    """Physical slots holding another expert than at startup."""
    return sum(
        logical_id != physical_id % num_routed_experts
        for placement in fingerprint
        for physical_id, logical_id in enumerate(placement)
    )
