from __future__ import annotations

from typing import Any

from PIL.Image import Image

from prime_rl.multimodal.base import ForwardPolicy, MaterializedMM, required_tensors


class KimiK25Adapter:
    model_types = frozenset({"kimi_k25"})
    forward_policy = ForwardPolicy()

    def materialize(
        self,
        image_processor: Any,
        images: list[Image],
        placeholder_lengths: list[int],
    ) -> MaterializedMM:
        preprocess = getattr(image_processor, "preprocess", None)
        if preprocess is None:
            raise ValueError("Kimi image processor is missing preprocess")
        media = [{"type": "image", "image": image} for image in images]
        kwargs = required_tensors(
            preprocess(media, return_tensors="pt"),
            ("pixel_values", "grid_thws"),
        )
        lengths = [1] * len(kwargs["grid_thws"].reshape(-1, 3))
        if lengths != placeholder_lengths:
            raise ValueError(
                f"Kimi image placeholder lengths differ from vLLM: expected {placeholder_lengths}, got {lengths}"
            )
        return MaterializedMM(kwargs=kwargs, forward_policy=self.forward_policy)
