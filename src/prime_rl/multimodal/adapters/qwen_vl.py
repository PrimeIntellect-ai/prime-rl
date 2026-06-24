from __future__ import annotations

import math
from typing import Any

from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.multimodal.schema import RawMMItem


def _processor_value(processor: Any, name: str, *, size_key: str | None = None) -> int:
    value = getattr(processor, name, None)
    if value is None and size_key is not None:
        size = getattr(processor, "size", None)
        get_size_value = getattr(size, "get", None)
        if callable(get_size_value):
            value = get_size_value(size_key)
    if value is None:
        raise ValueError(f"Image processor is missing {name}")
    return int(value)


def _tensorize(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return torch.as_tensor(value).contiguous()


def _grid_payload(item: RawMMItem) -> list[int]:
    grid = item.payload.get("image_grid_thw")
    if grid is None:
        raise ValueError("Qwen raw descriptor payload is missing image_grid_thw")
    if len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    if not isinstance(grid, list | tuple) or len(grid) != 3:
        raise ValueError(f"Invalid Qwen image_grid_thw: {grid!r}")
    out = [int(v) for v in grid]
    if any(v <= 0 for v in out):
        raise ValueError(f"Invalid Qwen image_grid_thw: {grid!r}")
    return out


def _patch_area(patch_size: Any) -> int:
    if isinstance(patch_size, list | tuple):
        return math.prod(int(dim) for dim in patch_size)
    size = int(patch_size)
    return size * size


def _temporal_patch_extent(temporal_patch_size: Any) -> int:
    if isinstance(temporal_patch_size, list | tuple):
        return math.prod(int(dim) for dim in temporal_patch_size)
    return int(temporal_patch_size)


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

    def processor_fingerprint(self, image_processor: Any) -> str:
        from renderers.mm_store import image_layout_fingerprint

        return image_layout_fingerprint(
            family=self.family,
            patch_size=_processor_value(image_processor, "patch_size"),
            merge_size=_processor_value(image_processor, "merge_size"),
            temporal_patch_size=_processor_value(image_processor, "temporal_patch_size"),
            min_pixels=_processor_value(image_processor, "min_pixels", size_key="shortest_edge"),
            max_pixels=_processor_value(image_processor, "max_pixels", size_key="longest_edge"),
        )

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
        for idx, item in enumerate(items):
            expected = _grid_payload(item)
            if actual_grids[idx] != expected:
                raise ValueError(f"Image grid mismatch at index {idx}: expected {expected}, got {actual_grids[idx]}")
        return MaterializedMM(kwargs=tensors, forward_policy=self.forward_policy)

    def materialize_for_vllm(
        self,
        image_processor: Any,
        item: RawMMItem,
        image: Any,
        expected_placeholder_length: int | None,
    ) -> Any:
        from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
        from vllm.multimodal.inputs import MultiModalKwargsItems

        self.validate_item(item)
        actual_fingerprint = self.processor_fingerprint(image_processor)
        if actual_fingerprint != item.layout_fingerprint:
            raise ValueError(
                f"Image layout fingerprint mismatch: expected {item.layout_fingerprint}, got {actual_fingerprint}"
            )
        hf_inputs = image_processor(images=[image], return_tensors="pt")
        merge_size = _processor_value(image_processor, "merge_size")
        config_by_key = _create_qwen2vl_field_factory(merge_size)(hf_inputs)
        mm_item = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config_by_key)["image"][0]
        expected_grid = _grid_payload(item)
        actual_grid = mm_item["image_grid_thw"].data.tolist()
        if actual_grid != expected_grid:
            raise ValueError(f"Image grid mismatch: expected {expected_grid}, got {actual_grid}")
        num_image_tokens = int(expected_grid[0] * expected_grid[1] * expected_grid[2] // (merge_size * merge_size))
        if expected_placeholder_length is not None and expected_placeholder_length != num_image_tokens:
            raise ValueError(
                f"Image placeholder length mismatch: expected {expected_placeholder_length}, got {num_image_tokens}"
            )
        return mm_item

    def placeholder_feature_dim(self, image_processor: Any) -> int:
        patch_size = getattr(image_processor, "patch_size", None)
        temporal_patch_size = getattr(image_processor, "temporal_patch_size", None)
        image_mean = getattr(image_processor, "image_mean", None)
        channels = len(image_mean) if image_mean is not None else getattr(image_processor, "num_channels", 3)
        if patch_size is None or temporal_patch_size is None:
            raise ValueError(
                "Cannot synthesize raw image placeholders without image processor patch_size and temporal_patch_size"
            )
        return int(channels) * _temporal_patch_extent(temporal_patch_size) * _patch_area(patch_size)

    def synthesize_placeholder(
        self,
        image_processor: Any,
        items: list[RawMMItem],
    ) -> MaterializedMM | None:
        if not items:
            return None
        import torch

        feature_dim = self.placeholder_feature_dim(image_processor)
        pixel_values: list[torch.Tensor] = []
        image_grid_thw: list[list[int]] = []
        for idx, item in enumerate(items):
            self.validate_item(item)
            grid = _grid_payload(item)
            pixel_values.append(torch.zeros((math.prod(grid), feature_dim), dtype=torch.float32))
            image_grid_thw.append(grid)
        return MaterializedMM(
            kwargs={
                "pixel_values": torch.cat(pixel_values, dim=0).contiguous(),
                "image_grid_thw": torch.tensor(image_grid_thw, dtype=torch.long),
            },
            forward_policy=self.forward_policy,
        )
