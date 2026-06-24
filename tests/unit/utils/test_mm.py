from types import SimpleNamespace

import pytest
import torch

from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.multimodal.schema import RAW_MM_ITEM_KIND, RAW_MM_ITEM_VERSION
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transport.types import MicroBatch, MMRefs
from prime_rl.utils.mm import RawImageMaterializer, build_mm_refs, missing_file_uris


class _ImageProcessor:
    patch_size = 2
    temporal_patch_size = 2
    image_mean = [0.5, 0.5, 0.5]


class _MissingMaterializer:
    def materialize(self, refs):
        raise FileNotFoundError("missing image")

    def synthesize_placeholder(self, refs):
        return MaterializedMM(
            kwargs={
                "pixel_values": torch.zeros((1, 24), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            },
            forward_policy=ForwardPolicy(pass_position_ids_with_mm=False),
        )


def _qwen_item(grid):
    return {
        "kind": RAW_MM_ITEM_KIND,
        "version": RAW_MM_ITEM_VERSION,
        "modality": "image",
        "family": "qwen_vl",
        "layout_fingerprint": "f" * 32,
        "payload": {"image_grid_thw": grid},
    }


def _refs(uri: str = "file:///tmp/missing-image.png") -> MMRefs:
    return MMRefs(
        descriptor={
            "mm_items": {"image": [_qwen_item([[1, 1, 1]])]},
            "mm_hashes": {"image": ["a" * 32]},
        },
        uris=[uri],
    )


def _loader(policy: str = "placeholder_zero_loss") -> DataLoader:
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)
    loader.mm_materializer = _MissingMaterializer()
    loader.missing_mm_image_policy = policy
    loader.last_mm_materialize_time = 0.0
    loader.last_mm_images_materialized = 0
    loader.last_mm_images_placeholdered = 0
    return loader


def _micro_batch() -> MicroBatch:
    return MicroBatch(
        input_ids=[10, 11, 12],
        loss_mask=[False, True, True],
        advantages=[1.5, 1.5, 1.5],
        inference_logprobs=[0.0, -0.1, -0.2],
        position_ids=[0, 1, 2],
        sequence_lengths=[3],
        temperatures=[1.0, 1.0, 1.0],
        env_names=["env", "env", "env"],
        lora_num_tokens=[3],
        mm_refs=_refs(),
        mm_token_type_ids=[0, 1, 0],
    )


def test_missing_file_uris_reports_missing_local_refs(tmp_path):
    existing = tmp_path / "image.png"
    existing.write_bytes(b"image")

    assert missing_file_uris(
        [existing.as_uri(), (tmp_path / "missing.png").as_uri(), "https://example.test/i.png"]
    ) == [(tmp_path / "missing.png").as_uri()]


def test_build_mm_refs_rejects_legacy_descriptor_without_raw_envelope(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    multi_modal_data = SimpleNamespace(
        mm_items={"image": [{"image_grid_thw": [[1, 1, 1]]}]},
        mm_hashes={"image": ["a" * 32]},
        mm_placeholders={"image": []},
    )
    messages = [
        {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_path.as_uri()},
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="common envelope"):
        build_mm_refs(multi_modal_data, messages)


def test_raw_image_materializer_synthesizes_qwen_placeholder_from_descriptor():
    materializer = RawImageMaterializer("unused", trust_remote_code=False)
    materializer._image_processor = _ImageProcessor()

    mm_kwargs = materializer.synthesize_placeholder(
        MMRefs(
            descriptor={
                "mm_items": {"image": [_qwen_item([[1, 2, 3]])]},
                "mm_hashes": {"image": ["b" * 32]},
            },
            uris=["file:///tmp/missing-image.png"],
        )
    )

    assert mm_kwargs is not None
    assert mm_kwargs.kwargs["pixel_values"].shape == (6, 24)
    assert not bool(mm_kwargs.kwargs["pixel_values"].any())
    assert mm_kwargs.kwargs["image_grid_thw"].tolist() == [[1, 2, 3]]


def test_dataloader_uses_zero_loss_placeholder_for_missing_raw_image():
    tensor_batch = _loader()._micro_batch_to_tensor(_micro_batch())

    assert tensor_batch["mm_kwargs"] is not None
    assert tensor_batch["mm_kwargs"]["pixel_values"].shape == (1, 24)
    assert tensor_batch["mm_forward_policy"] == ForwardPolicy(pass_position_ids_with_mm=False)
    assert tensor_batch["loss_mask"].tolist() == [[False, False, False]]
    assert tensor_batch["advantages"].tolist() == [[0.0, 0.0, 0.0]]
    assert tensor_batch["mm_token_type_ids"].tolist() == [[0, 1, 0]]


def test_dataloader_can_fail_fast_on_missing_raw_image():
    with pytest.raises(FileNotFoundError):
        _loader(policy="error")._micro_batch_to_tensor(_micro_batch())
