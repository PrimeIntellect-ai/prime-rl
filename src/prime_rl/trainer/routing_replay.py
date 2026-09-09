"""Immutable tensor inputs for direct routing replay.

The loader owns validation of captured rows, finite coefficients and expert-ID
ranges. These model-side helpers check tensor metadata without reading device
values or adding per-layer host synchronizations.
"""

from typing import NamedTuple

import torch


class RoutingReplay(NamedTuple):
    """Paired logical expert IDs and final FP32 coefficients, in recorded slot order.

    The tuple is immutable, not its tensor storage. Callers must keep both tensors
    unchanged until backward (including checkpoint recomputation) has finished.
    Weights already include inference normalization/scaling. They are detached
    when consumed by the router, not normalized or cast a second time.
    """

    ids: torch.Tensor
    weights: torch.Tensor

    def to(self, device: torch.device | str | int, *, non_blocking: bool = False) -> "RoutingReplay":
        """Move both tensors together without changing their dtypes."""
        return RoutingReplay(
            self.ids.to(device=device, non_blocking=non_blocking),
            self.weights.to(device=device, non_blocking=non_blocking),
        )


def validate_routing_replay(
    replay: RoutingReplay,
    *,
    expected_shape: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> None:
    """Check metadata only; data-dependent validation belongs at loader ingress."""
    if not isinstance(replay.ids, torch.Tensor) or not isinstance(replay.weights, torch.Tensor):
        raise TypeError("RoutingReplay.ids and RoutingReplay.weights must be tensors")
    if replay.ids.shape != replay.weights.shape:
        raise ValueError("RoutingReplay IDs and weights must have the same shape")
    if expected_shape is not None and replay.ids.shape != expected_shape:
        raise ValueError(f"RoutingReplay shape {tuple(replay.ids.shape)} does not match expected {expected_shape}")
    if replay.ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("RoutingReplay IDs must have dtype int32 or int64")
    if replay.weights.dtype != torch.float32:
        raise TypeError("RoutingReplay weights must have dtype float32")
    if replay.ids.device != replay.weights.device:
        raise ValueError("RoutingReplay IDs and weights must be on the same device")
    if device is not None and replay.ids.device != device:
        raise ValueError("RoutingReplay tensors must be on the same device as the hidden states")


def replay_routing(
    replay: RoutingReplay,
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use captured coefficients without entering a parameter-bearing router module.

    Bypassing the module call also avoids its FSDP pre-forward hooks. Parameters
    remain registered for checkpointing. The loader validates values and ranges;
    this hot path checks only metadata, preserves slot order and detaches weights.
    """
    validate_routing_replay(replay, expected_shape=(num_tokens, top_k), device=device)
    weights = replay.weights.detach()
    counts = torch.histc(replay.ids.reshape(-1).float(), bins=num_experts, min=0, max=num_experts).to(torch.int64)
    # The trainer gate was not evaluated; its confidence is unavailable.
    return weights, replay.ids, counts, weights.new_full((), float("nan"))


def select_routing_layer(
    routed_experts: torch.Tensor | RoutingReplay | None,
    layer_idx: int,
) -> torch.Tensor | RoutingReplay | None:
    """Slice the layer axis of model inputs shaped ``[batch, sequence, layer, top_k]``."""
    if routed_experts is None:
        return None
    if isinstance(routed_experts, RoutingReplay):
        validate_routing_replay(routed_experts)
        if routed_experts.ids.ndim != 4:
            raise ValueError("Model RoutingReplay must have shape [batch, sequence, layer, top_k]")
        return RoutingReplay(
            routed_experts.ids[:, :, layer_idx, :],
            routed_experts.weights[:, :, layer_idx, :],
        )
    if not isinstance(routed_experts, torch.Tensor):
        raise TypeError("routed_experts must be a Tensor, RoutingReplay, or None")
    return routed_experts[:, :, layer_idx, :]
