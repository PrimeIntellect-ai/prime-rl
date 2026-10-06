from types import SimpleNamespace

import pytest
import torch

from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.trainer.rl.data import init_micro_batch_worker, prepare_micro_batch
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


def test_prepare_micro_batch_in_worker_pool_materializes_in_order():
    class ImageProcessor:
        merge_size = 1

        def __call__(self, *, images, return_tensors):
            return {"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])}

    processor = SimpleNamespace(image_processor=ImageProcessor())
    adapter = get_multimodal_adapter("qwen3_vl")
    micro_batches = [{"id": 0, "mm_refs": _refs(2)}, {"id": 1, "mm_refs": None}, {"id": 2, "mm_refs": _refs(2)}]

    with WorkerPool(1, init_micro_batch_worker, (processor, adapter)) as pool:
        results = list(pool.imap(prepare_micro_batch, micro_batches))

    assert [micro_batch["id"] for micro_batch in results] == [0, 1, 2]
    assert results[1].get("materialized_mm") is None
    for micro_batch in (results[0], results[2]):
        assert micro_batch["mm_refs"] is None
        assert set(micro_batch["materialized_mm"].kwargs) == {"pixel_values", "image_grid_thw"}
    with WorkerPool(1, init_micro_batch_worker, (None, None)) as pool:
        with pytest.raises(ValueError, match=r"\[model.vlm\] is not set"):
            list(pool.imap(prepare_micro_batch, [{"mm_refs": _refs(2)}]))
