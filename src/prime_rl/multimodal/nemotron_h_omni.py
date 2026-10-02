from __future__ import annotations

from typing import Any

from PIL.Image import Image

from prime_rl.multimodal.base import ForwardPolicy, MaterializedMM, required_tensors


class NemotronHOmniAdapter:
    model_types = frozenset({"nemotron_h_omni"})
    forward_policy = ForwardPolicy(
        requires_mm_token_type_ids=True,
        defer_context_parallelism=True,
    )

    def materialize(
        self,
        image_processor: Any,
        images: list[Image],
        placeholder_lengths: list[int],
    ) -> MaterializedMM:
        values = image_processor(images=images, return_tensors="pt")
        pixel_values = values.get("pixel_values")
        if isinstance(pixel_values, list):
            raise ValueError("Nemotron images in one batch must have the same processed shape")
        kwargs = required_tensors(
            values,
            ("pixel_values", "imgs_sizes", "num_tokens", "num_patches"),
        )
        lengths = [int(length) for length in kwargs["num_tokens"].reshape(-1)]
        if lengths != placeholder_lengths:
            raise ValueError(
                f"Nemotron image placeholder lengths differ from vLLM: expected {placeholder_lengths}, got {lengths}"
            )
        return MaterializedMM(kwargs=kwargs, forward_policy=self.forward_policy)
