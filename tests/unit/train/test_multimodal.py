from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.vlm import materialize_images
from prime_rl.transports.batch import MMImageRef, MMRefs

_IMAGE_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _refs(length: int) -> MMRefs:
    return MMRefs(images=[MMImageRef(url=_IMAGE_URL, offset=1, length=length)])


def test_qwen_images_materialize_and_validate_expansion():
    class ImageProcessor:
        merge_size = 1

        def __call__(self, *, images, return_tensors):
            assert len(images) == 1 and images[0].mode == "RGB" and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    assert set(materialize_images(_refs(2), processor, "qwen3_5")) == {"pixel_values", "image_grid_thw"}
    with pytest.raises(ValueError, match="placeholder lengths differ"):
        materialize_images(_refs(1), processor, "qwen3_5")


def test_kimi_images_materialize_sparse_image_position():
    class ImageProcessor:
        def preprocess(self, media, *, return_tensors):
            assert len(media) == 1 and media[0]["type"] == "image"
            assert media[0]["image"].mode == "RGB" and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(4, 3),
                "grid_thws": torch.tensor([[1, 2, 2]]),
            }

    processor = SimpleNamespace(image_processor=ImageProcessor())
    assert set(materialize_images(_refs(1), processor, "kimi_k25")) == {"pixel_values", "grid_thws"}
