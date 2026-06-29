from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image

    from prime_rl.multimodal.schema import RawMMItem


@dataclass(frozen=True)
class ForwardPolicy:
    pass_position_ids_with_mm: bool = True
    requires_mm_token_type_ids: bool = False


@dataclass(frozen=True)
class MaterializedMM:
    kwargs: dict[str, "torch.Tensor"]
    forward_policy: ForwardPolicy


class MultimodalAdapter(Protocol):
    family: str
    forward_policy: ForwardPolicy

    def validate_item(self, item: "RawMMItem") -> None: ...

    def processor_fingerprint(self, image_processor: Any) -> str: ...

    def materialize_for_trainer(
        self,
        image_processor: Any,
        items: list["RawMMItem"],
        images: list["Image"],
    ) -> MaterializedMM: ...

    def materialize_for_vllm(
        self,
        image_processor: Any,
        item: "RawMMItem",
        image: "Image",
        expected_placeholder_length: int,
    ) -> Any: ...

    def synthesize_placeholder(
        self,
        image_processor: Any,
        items: list["RawMMItem"],
    ) -> MaterializedMM | None: ...
