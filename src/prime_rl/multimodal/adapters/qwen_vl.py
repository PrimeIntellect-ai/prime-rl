from __future__ import annotations

from typing import Any

from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.multimodal.schema import RawMMItem


def _tensorize(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return torch.as_tensor(value).contiguous()


def _grid_payload(item: RawMMItem) -> list[int]:
    grid = item.payload.get("image_grid_thw")
    if grid is None:
        raise ValueError("Qwen raw descriptor payload is missing image_grid_thw")
    if not isinstance(grid, list | tuple):
        raise ValueError(f"Invalid Qwen image_grid_thw: {grid!r}")
    if len(grid) == 1 and isinstance(grid[0], list | tuple):
        grid = grid[0]
    if not isinstance(grid, list | tuple) or len(grid) != 3:
        raise ValueError(f"Invalid Qwen image_grid_thw: {grid!r}")
    out = [int(v) for v in grid]
    if any(v <= 0 for v in out):
        raise ValueError(f"Invalid Qwen image_grid_thw: {grid!r}")
    return out


class QwenVLAdapter:
    family = "qwen_vl"
    forward_policy = ForwardPolicy(
        pass_position_ids_with_mm=False,
        requires_mm_token_type_ids=True,
    )

    def validate_item(self, item: RawMMItem) -> None:
        if item.family != self.family:
            raise ValueError(f"Qwen adapter cannot handle family {item.family!r}")
        _grid_payload(item)

    def materialize_for_trainer(
        self,
        image_processor: Any,
        items: list[RawMMItem],
        images: list[Any],
    ) -> MaterializedMM:
        for item in items:
            self.validate_item(item)
        processed = image_processor(images=images, return_tensors="pt")
        tensors = {str(k): _tensorize(v) for k, v in dict(processed).items()}
        if "image_grid_thw" not in tensors:
            raise ValueError("Qwen processor did not return image_grid_thw")
        actual_grids = tensors["image_grid_thw"].tolist()
        for idx, (item, actual_grid) in enumerate(zip(items, actual_grids, strict=True)):
            expected = _grid_payload(item)
            if actual_grid != expected:
                raise ValueError(f"Image grid mismatch at index {idx}: expected {expected}, got {actual_grid}")
        return MaterializedMM(kwargs=tensors, forward_policy=self.forward_policy)

    def materialize_for_vllm(
        self,
        image_processor: Any,
        item: RawMMItem,
        image: Any,
        expected_placeholder_length: int,
    ) -> Any:
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems

        self.validate_item(item)
        hf_inputs = image_processor(images=[image], return_tensors="pt")
        merge_size = int(image_processor.merge_size)
        config_by_key = _create_qwen2vl_field_factory(merge_size)(hf_inputs)
        mm_item = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config_by_key)["image"][0]
        expected_grid = _grid_payload(item)
        actual_grid = mm_item["image_grid_thw"].data.tolist()
        if actual_grid != expected_grid:
            raise ValueError(f"Image grid mismatch: expected {expected_grid}, got {actual_grid}")
        num_image_tokens = int(expected_grid[0] * expected_grid[1] * expected_grid[2] // (merge_size * merge_size))
        if expected_placeholder_length != num_image_tokens:
            raise ValueError(
                f"Image placeholder length mismatch: expected {expected_placeholder_length}, got {num_image_tokens}"
            )
        return mm_item
