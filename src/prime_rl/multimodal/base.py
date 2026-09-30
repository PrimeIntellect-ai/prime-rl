from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL.Image import Image


@dataclass(frozen=True)
class ForwardPolicy:
    pass_position_ids: bool = True
    requires_mm_token_type_ids: bool = False
    defer_context_parallelism: bool = False


@dataclass(frozen=True)
class MaterializedMM:
    kwargs: dict[str, torch.Tensor]
    forward_policy: ForwardPolicy


class MultimodalAdapter(Protocol):
    model_types: frozenset[str]
    forward_policy: ForwardPolicy

    def materialize(
        self,
        image_processor: Any,
        images: list[Image],
        placeholder_lengths: list[int],
    ) -> MaterializedMM: ...


def required_tensors(values: Any, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    data = dict(values)
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Image processor did not return {', '.join(missing)}")
    return {key: torch.as_tensor(data[key]).contiguous() for key in keys}
