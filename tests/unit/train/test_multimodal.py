import base64
import io
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


def _prepare(micro_batch: dict) -> dict:
    processor = SimpleNamespace(image_processor=_PixelImageProcessor())
    return prepare_micro_batch(micro_batch, processor=processor, mm_adapter=get_multimodal_adapter("qwen3_vl"))


def test_prepare_micro_batch_text_only():
    micro_batch = {"mm_refs": None, "input_ids": torch.zeros(1, 4)}

    prepared = _prepare(micro_batch)

    assert prepared is micro_batch
    assert prepared["mm_kwargs"] is None and prepared["mm_forward_policy"] is None


def test_prepare_micro_batch_materializes_images():
    micro_batch = {"mm_refs": _color_refs((255, 0, 0)), "input_ids": torch.zeros(1, 4)}

    prepared = _prepare(micro_batch)

    assert prepared is micro_batch
    assert prepared["mm_refs"] is None
    assert set(prepared["mm_kwargs"]) == {"pixel_values", "image_grid_thw"}
    assert prepared["mm_forward_policy"] == QwenVLAdapter.forward_policy
    assert torch.equal(prepared["input_ids"], torch.zeros(1, 4))


def test_prepare_micro_batch_requires_vlm_config():
    with pytest.raises(ValueError, match=r"\[model.vlm\] is not set"):
        prepare_micro_batch({"mm_refs": _refs(2)}, processor=None, mm_adapter=None)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_worker_pool_prepares_micro_batches_like_main_thread(num_workers: int):
    def make_micro_batches():
        colors = [(255, 0, 0), None, (0, 255, 0), (0, 0, 255), (7, 8, 9)]
        return [{"mm_refs": None if color is None else _color_refs(color)} for color in colors]

    with WorkerPool(num_workers) as workers:
        pool_results = list(workers(_prepare, make_micro_batches()))
    main_thread_results = [_prepare(micro_batch) for micro_batch in make_micro_batches()]

    for pool_result, main_thread_result in zip(pool_results, main_thread_results, strict=True):
        if main_thread_result["mm_kwargs"] is None:
            assert pool_result["mm_kwargs"] is None
            continue
        assert pool_result["mm_kwargs"].keys() == main_thread_result["mm_kwargs"].keys()
        for key, value in main_thread_result["mm_kwargs"].items():
            assert torch.equal(pool_result["mm_kwargs"][key], value)
