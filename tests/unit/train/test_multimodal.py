from types import SimpleNamespace

import pytest
import torch

from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs, release_staged_mm, stage_mm_refs
from prime_rl.transports.rollouts import MMImageRef, MMRefs

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


def test_stage_and_release_bounds_materialized_payload_lifetime():
    class ImageProcessor:
        merge_size = 1

        def __call__(self, *, images, return_tensors):
            assert len(images) == 1 and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    staged = stage_mm_refs(_refs(2), processor, get_multimodal_adapter("qwen3_vl"), "cpu")

    assert set(staged.kwargs) == {"pixel_values", "image_grid_thw"}
    release_staged_mm(staged)
    assert staged.kwargs == {}
