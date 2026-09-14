"""Exact startup restoration checks for MX refits, outside warm timing samples."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WeightSnapshot:
    values: dict[str, torch.Tensor]
    layouts: dict[str, tuple]
    aliases: dict[str, str]
    changed_tensors: int = 0


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters(remove_duplicate=False))
    for module_name, module in model.named_modules():
        if not hasattr(module, "kv_b_proj"):
            continue
        for leaf in ("W_UV", "W_UK_T"):
            tensor = getattr(module, leaf, None)
            if isinstance(tensor, torch.Tensor):
                name = f"{module_name}.{leaf}" if module_name else leaf
                if name in tensors:
                    raise ValueError(f"Duplicate verification tensor name: {name}")
                tensors[name] = tensor
    return tensors


def _layout(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr(),
        tensor.storage_offset(),
    )


def _aliases(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    canonical = {}
    return {name: canonical.setdefault(id(tensor), name) for name, tensor in tensors.items()}


def _chunks(tensor: torch.Tensor, max_bytes: int = 64 * 1024**2):
    if tensor.numel() * tensor.element_size() <= max_bytes:
        yield tensor
        return
    row_bytes = tensor[0].numel() * tensor.element_size()
    rows = max(1, max_bytes // row_bytes)
    for start in range(0, tensor.shape[0], rows):
        part = tensor[start : start + rows]
        if rows == 1 and row_bytes > max_bytes:
            yield from _chunks(part[0], max_bytes)
        else:
            yield part


@torch.no_grad()
def snapshot_weights(model: nn.Module, *, max_bytes: int = 64 * 1024**3) -> WeightSnapshot:
    """Copy all named parameters and MLA derived weights to bounded CPU storage.

    The budget is per rank and must be provisioned across every local worker.
    No GPU tensor references are retained. This is startup verification storage,
    separate from the transport's GPU staging budget.
    """
    tensors = _weights(model)
    aliases = _aliases(tensors)
    unique = {name: tensor for name, tensor in tensors.items() if aliases[name] == name}
    required = sum(t.numel() * t.element_size() for t in unique.values())
    if required > max_bytes:
        raise ValueError(f"Initial verification needs {required} CPU bytes per rank; budget={max_bytes}")
    if not unique or not any(t.numel() for t in unique.values()):
        raise ValueError("Initial verification has no nonempty weights")
    if any(not t.is_floating_point() or t.device.type == "meta" for t in unique.values()):
        raise ValueError("Initial restoration requires materialized floating-point model weights")
    values = {}
    for name, tensor in unique.items():
        value = tensor.detach().to(device="cpu", copy=True)
        if any(not torch.isfinite(chunk).all().item() for chunk in _chunks(value)):
            raise ValueError(f"Non-finite initial weights: {name}")
        values[name] = value
    return WeightSnapshot(values, {n: _layout(t) for n, t in tensors.items()}, aliases)


@torch.no_grad()
def perturb_weights(model: nn.Module, snapshot: WeightSnapshot) -> None:
    """Poison every nonempty selected tensor while generation is paused."""
    tensors = _weights(model)
    if snapshot.changed_tensors:
        raise ValueError("Initial snapshot was already perturbed")
    if {n: _layout(t) for n, t in tensors.items()} != snapshot.layouts or _aliases(tensors) != snapshot.aliases:
        raise ValueError("Weights changed between snapshot and perturbation")
    for name in snapshot.values:
        tensor = tensors[name]
        if tensor.numel():
            tensor.fill_(float("nan"))
            if not torch.isnan(tensor[tuple(0 for _ in tensor.shape)]).item():
                raise RuntimeError(f"Weight perturbation failed: {name}")
            snapshot.changed_tensors += 1


@torch.no_grad()
def verify_weights(model: nn.Module, snapshot: WeightSnapshot) -> dict:
    """Compare every perturbed weight exactly, including aliases and addresses."""
    tensors = _weights(model)
    layout_ok = {n: _layout(t) for n, t in tensors.items()} == snapshot.layouts
    aliases_ok = _aliases(tensors) == snapshot.aliases
    verified = fp32 = 0
    failed = []
    for name, expected in snapshot.values.items():
        if not expected.numel():
            continue
        actual = tensors.get(name)
        equal = actual is not None and _layout(actual) == snapshot.layouts[name]
        if equal:
            for observed_chunk, expected_chunk in zip(_chunks(actual), _chunks(expected), strict=True):
                if not torch.equal(observed_chunk.detach().cpu(), expected_chunk):
                    equal = False
                    break
        if equal:
            verified += 1
            fp32 += expected.dtype == torch.float32
        else:
            failed.append(name)
    return {
        "record": "mx-initial-refit-verification-v1",
        "passed": layout_ok and aliases_ok and not failed and verified == snapshot.changed_tensors > 0,
        "changed_tensors": snapshot.changed_tensors,
        "verified_tensors": verified,
        "verified_fp32_tensors": fp32,
        "addresses_preserved": layout_ok,
        "aliases_preserved": aliases_ok,
        "failed_tensors": failed[:20],
        "scope": "All named parameters and MLA derived weights; exact values, layouts and aliases",
    }
