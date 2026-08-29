from __future__ import annotations

from typing import Any

from PIL.Image import Image

from prime_rl.multimodal.base import ForwardPolicy, MaterializedMM, required_tensors


class QwenVLAdapter:
    model_types = frozenset({"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"})
    forward_policy = ForwardPolicy(
        pass_position_ids=False,
        requires_mm_token_type_ids=True,
        defer_context_parallelism=True,
    )

    def materialize(
        self,
        image_processor: Any,
        images: list[Image],
        placeholder_lengths: list[int],
    ) -> MaterializedMM:
        kwargs = required_tensors(
            image_processor(images=images, return_tensors="pt"),
            ("pixel_values", "image_grid_thw"),
        )
        merge_size = int(image_processor.merge_size)
        # HF Qwen-VL / renderer pad count: T*H*W / merge_size^2.
        lengths = [int(grid.prod()) // (merge_size * merge_size) for grid in kwargs["image_grid_thw"].reshape(-1, 3)]
        if lengths != placeholder_lengths:
            raise ValueError(
                f"Qwen image placeholder lengths differ from vLLM: expected {placeholder_lengths}, got {lengths}"
            )
        return MaterializedMM(kwargs=kwargs, forward_policy=self.forward_policy)
