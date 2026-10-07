import base64
import io
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.rl.data import prepare_micro_batch
from prime_rl.transports.batch import MMImageRef, MMRefs
from prime_rl.utils.worker_pool import WorkerPool

_IMAGE_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _refs(length: int) -> MMRefs:
    return MMRefs(images=[MMImageRef(url=_IMAGE_URL, offset=1, length=length)])


def test_qwen_adapter_materializes_and_validates_expansion():
    class ImageProcessor:
        merge_size = 1

        def __call__(self, *, images, return_tensors):
            assert len(images) == 1 and images[0].mode == "RGB" and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    adapter = get_multimodal_adapter("qwen3_vl")
    materialized = materialize_mm_refs(_refs(2), processor, adapter)

    assert set(materialized.kwargs) == {"pixel_values", "image_grid_thw"}
    assert materialized.forward_policy == QwenVLAdapter.forward_policy
    with pytest.raises(ValueError, match="placeholder lengths differ"):
        materialize_mm_refs(_refs(1), processor, adapter)


def test_kimi_adapter_materializes_sparse_image_position():
    class ImageProcessor:
        def preprocess(self, media, *, return_tensors):
            assert len(media) == 1 and media[0]["type"] == "image"
            assert media[0]["image"].mode == "RGB" and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(4, 3),
                "grid_thws": torch.tensor([[1, 2, 2]]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    materialized = materialize_mm_refs(_refs(1), processor, get_multimodal_adapter("kimi_k25"))

    assert set(materialized.kwargs) == {"pixel_values", "grid_thws"}
    assert materialized.forward_policy == KimiK25Adapter.forward_policy


class _PixelImageProcessor:
    merge_size = 1

    def __call__(self, *, images, return_tensors):
        pixels = torch.cat([torch.from_numpy(np.array(image)).reshape(-1, 3) for image in images]).float()
        return {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 1, 2]] * len(images))}


def _color_refs(color: tuple[int, int, int]) -> MMRefs:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return MMRefs(images=[MMImageRef(url=url, offset=1, length=2)])


@pytest.mark.parametrize("num_workers", [1, 2])
def test_prepare_micro_batch_in_worker_pool(num_workers: int):
    processor = SimpleNamespace(image_processor=_PixelImageProcessor())
    adapter = get_multimodal_adapter("qwen3_vl")
    prepare = partial(prepare_micro_batch, processor=processor, mm_adapter=adapter)

    def make_micro_batches():
        colors = [(255, 0, 0), None, (0, 255, 0), (0, 0, 255), (7, 8, 9)]
        return [
            {"mm_refs": None if color is None else _color_refs(color), "input_ids": torch.zeros(1, 4)}
            for color in colors
        ]

    micro_batches = make_micro_batches()
    with WorkerPool(num_workers) as workers:
        results = list(workers(prepare, micro_batches))
    with WorkerPool(0) as workers:
        inline_results = list(workers(prepare, make_micro_batches()))

    assert all(result is micro_batch for result, micro_batch in zip(results, micro_batches))
    assert results[1] == {
        "mm_refs": None,
        "input_ids": results[1]["input_ids"],
        "mm_kwargs": None,
        "mm_forward_policy": None,
    }
    for result, inline_result in zip(results, inline_results):
        assert result["mm_refs"] is None
        assert torch.equal(result["input_ids"], torch.zeros(1, 4))
        if result["mm_kwargs"] is None:
            assert inline_result["mm_kwargs"] is None
            continue
        assert result["mm_forward_policy"] == QwenVLAdapter.forward_policy
        assert set(result["mm_kwargs"]) == set(inline_result["mm_kwargs"]) == {"pixel_values", "image_grid_thw"}
        assert all(
            torch.equal(result["mm_kwargs"][key], inline_result["mm_kwargs"][key]) for key in result["mm_kwargs"]
        )
    with WorkerPool(num_workers) as workers:
        with pytest.raises(ValueError, match=r"\[model.vlm\] is not set"):
            list(workers(partial(prepare_micro_batch, processor=None, mm_adapter=None), [{"mm_refs": _refs(2)}]))
