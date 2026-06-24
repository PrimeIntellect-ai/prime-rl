from __future__ import annotations

import math
from typing import Any

from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.multimodal.schema import RawMMItem

KIMI_K25_DEFAULTS = {
    "patch_size": 14,
    "merge_kernel_size": 2,
    "in_patch_limit": 16384,
    "patch_limit_on_one_side": 512,
    "fixed_output_tokens": None,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def _tensorize(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.contiguous()
    return torch.as_tensor(value).contiguous()


def _cfg_value(image_processor: Any, name: str) -> Any:
    for source in (
        image_processor,
        getattr(image_processor, "media_proc_cfg", None),
        getattr(image_processor, "config", None),
    ):
        if source is None:
            continue
        if isinstance(source, dict) and name in source:
            return source[name]
        value = getattr(source, name, None)
        if value is not None:
            return value
    return KIMI_K25_DEFAULTS[name]


def _grid_payload(item: RawMMItem) -> list[int]:
    grid = item.payload.get("grid_thws")
    if grid is None:
        raise ValueError("Kimi raw descriptor payload is missing grid_thws")
    if len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    if not isinstance(grid, list | tuple) or len(grid) != 3:
        raise ValueError(f"Invalid Kimi grid_thws: {grid!r}")
    out = [int(v) for v in grid]
    if any(v <= 0 for v in out):
        raise ValueError(f"Invalid Kimi grid_thws: {grid!r}")
    return out


def _process_images(image_processor: Any, images: list[Any], *, return_tensors: str):
    medias = [{"type": "image", "image": image} for image in images]
    preprocess = getattr(image_processor, "preprocess", None)
    if preprocess is None:
        raise ValueError("Kimi image processor is missing preprocess")
    return preprocess(medias, return_tensors=return_tensors)


class KimiK25Adapter:
    family = "kimi_k25"
    forward_policy = ForwardPolicy(pass_position_ids_with_mm=True)

    def validate_item(self, item: RawMMItem) -> None:
        if item.family != self.family:
            raise ValueError(f"Kimi adapter cannot handle family {item.family!r}")
        _grid_payload(item)

    def processor_fingerprint(self, image_processor: Any) -> str:
        from renderers.mm_store import image_layout_fingerprint

        fixed_output_tokens = _cfg_value(image_processor, "fixed_output_tokens")
        return image_layout_fingerprint(
            family=self.family,
            patch_size=int(_cfg_value(image_processor, "patch_size")),
            merge_kernel_size=int(_cfg_value(image_processor, "merge_kernel_size")),
            in_patch_limit=int(_cfg_value(image_processor, "in_patch_limit")),
            patch_limit_on_one_side=int(_cfg_value(image_processor, "patch_limit_on_one_side")),
            fixed_output_tokens=None if fixed_output_tokens is None else int(fixed_output_tokens),
            image_mean=list(_cfg_value(image_processor, "image_mean")),
            image_std=list(_cfg_value(image_processor, "image_std")),
        )

    def materialize_for_trainer(
        self,
        image_processor: Any,
        items: list[RawMMItem],
        images: list[Any],
    ) -> MaterializedMM:
        for item in items:
            self.validate_item(item)
        processed = _process_images(image_processor, images, return_tensors="pt")
        tensors = {str(k): _tensorize(v) for k, v in dict(processed).items()}
        if "grid_thws" not in tensors:
            raise ValueError("Kimi processor did not return grid_thws")
        actual_grids = tensors["grid_thws"].reshape(-1, 3).tolist()
        for idx, item in enumerate(items):
            expected = _grid_payload(item)
            if actual_grids[idx] != expected:
                raise ValueError(f"Kimi grid mismatch at index {idx}: expected {expected}, got {actual_grids[idx]}")
        return MaterializedMM(kwargs=tensors, forward_policy=self.forward_policy)

    def materialize_for_vllm(
        self,
        image_processor: Any,
        item: RawMMItem,
        image: Any,
        expected_placeholder_length: int | None,
    ) -> Any:
        from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems

        self.validate_item(item)
        actual_fingerprint = self.processor_fingerprint(image_processor)
        if actual_fingerprint != item.layout_fingerprint:
            raise ValueError(
                f"Image layout fingerprint mismatch: expected {item.layout_fingerprint}, got {actual_fingerprint}"
            )
        hf_inputs = _process_images(image_processor, [image], return_tensors="pt")
        tensors = {str(k): _tensorize(v) for k, v in dict(hf_inputs).items()}
        expected_grid = _grid_payload(item)
        actual_grid = tensors["grid_thws"].reshape(-1, 3).tolist()[0]
        if actual_grid != expected_grid:
            raise ValueError(f"Kimi grid mismatch: expected {expected_grid}, got {actual_grid}")
        if expected_placeholder_length is not None and expected_placeholder_length != 1:
            raise ValueError(f"Kimi image placeholder length mismatch: expected {expected_placeholder_length}, got 1")
        grid_sizes = tensors["grid_thws"].reshape(-1, 3).prod(-1)
        config_by_key = {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("vision_chunk", grid_sizes),
            "grid_thws": MultiModalFieldConfig.batched("vision_chunk"),
        }
        return MultiModalKwargsItems.from_hf_inputs(tensors, config_by_key)["vision_chunk"][0]

    def synthesize_placeholder(
        self,
        image_processor: Any,
        items: list[RawMMItem],
    ) -> MaterializedMM | None:
        if not items:
            return None
        import torch

        patch_size = int(_cfg_value(image_processor, "patch_size"))
        grids: list[list[int]] = []
        pixel_values: list[torch.Tensor] = []
        for item in items:
            self.validate_item(item)
            grid = _grid_payload(item)
            grids.append(grid)
            pixel_values.append(torch.zeros((math.prod(grid), 3, patch_size, patch_size), dtype=torch.float32))
        return MaterializedMM(
            kwargs={
                "pixel_values": torch.cat(pixel_values, dim=0).contiguous(),
                "grid_thws": torch.tensor(grids, dtype=torch.long),
            },
            forward_policy=self.forward_policy,
        )
