from types import SimpleNamespace

import pytest
import torch

from prime_rl.multimodal import get_multimodal_adapter
from prime_rl.multimodal.kimi_k25 import KimiK25Adapter
from prime_rl.multimodal.nemotron_h_omni import NemotronHOmniAdapter
from prime_rl.multimodal.qwen_vl import QwenVLAdapter
from prime_rl.trainer.multimodal import materialize_mm_refs
from prime_rl.transports.batch import MMImageRef, MMRefs

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


def test_nemotron_adapter_materializes_and_validates_expansion():
    class ImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert len(images) == 1 and images[0].mode == "RGB" and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(1, 3, 16, 16),
                "imgs_sizes": torch.tensor([[16, 16]]),
                "num_tokens": torch.tensor([4]),
                "num_patches": torch.tensor([1]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    adapter = get_multimodal_adapter("nemotron_h_omni")
    materialized = materialize_mm_refs(_refs(4), processor, adapter)

    assert set(materialized.kwargs) == {"pixel_values", "imgs_sizes", "num_tokens", "num_patches"}
    assert materialized.forward_policy == NemotronHOmniAdapter.forward_policy
    with pytest.raises(ValueError, match="placeholder lengths differ"):
        materialize_mm_refs(_refs(3), processor, adapter)


def test_nemotron_adapter_rejects_variable_image_shapes():
    class ImageProcessor:
        def __call__(self, *, images, return_tensors):
            return {
                "pixel_values": [torch.ones(3, 16, 16), torch.ones(3, 32, 16)],
                "imgs_sizes": [(16, 16), (32, 16)],
                "num_tokens": [4, 8],
                "num_patches": [1, 1],
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    refs = MMRefs(
        images=[
            MMImageRef(url=_IMAGE_URL, offset=1, length=4),
            MMImageRef(url=_IMAGE_URL, offset=5, length=8),
        ]
    )
    with pytest.raises(ValueError, match="same processed shape"):
        materialize_mm_refs(refs, processor, get_multimodal_adapter("nemotron_h_omni"))
