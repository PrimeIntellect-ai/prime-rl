from functools import partial
from types import SimpleNamespace

import pytest
import torch

from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.rl.data import materialize_micro_batch_mm
from prime_rl.transports.batch import MMImageRef, MMRefs
from prime_rl.utils.worker_map import WorkerMap, prepare

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


def test_materialize_micro_batch_mm_in_worker_map():
    class ImageProcessor:
        merge_size = 1

        def __call__(self, *, images, return_tensors):
            return {"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])}

    processor = SimpleNamespace(image_processor=ImageProcessor())
    adapter = get_multimodal_adapter("qwen3_vl")

    micro_batches = [{"mm_refs": _refs(2), "input_ids": torch.zeros(1, 4)}, {"mm_refs": None}, {"mm_refs": _refs(2)}]
    with WorkerMap(1, partial(materialize_micro_batch_mm, processor=processor, mm_adapter=adapter)) as worker_map:
        results = list(prepare(worker_map, micro_batches, ["mm_refs"]))

    assert all(result is micro_batch for result, micro_batch in zip(results, micro_batches))
    assert results[1] == {"mm_refs": None, "mm_kwargs": None, "mm_forward_policy": None}
    assert torch.equal(results[0]["input_ids"], torch.zeros(1, 4))
    for micro_batch in (results[0], results[2]):
        assert micro_batch["mm_refs"] is None
        assert set(micro_batch["mm_kwargs"]) == {"pixel_values", "image_grid_thw"}
        assert micro_batch["mm_forward_policy"] == QwenVLAdapter.forward_policy
    with WorkerMap(1, partial(materialize_micro_batch_mm, processor=None, mm_adapter=None)) as worker_map:
        with pytest.raises(ValueError, match=r"\[model.vlm\] is not set"):
            list(prepare(worker_map, [{"mm_refs": _refs(2)}], ["mm_refs"]))
