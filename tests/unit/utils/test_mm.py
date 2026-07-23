from types import SimpleNamespace

import torch

from prime_rl.multimodal.adapters.base import ForwardPolicy, MaterializedMM
from prime_rl.multimodal.schema import RAW_MM_ITEM_KIND
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transport.types import MicroBatch, MMImageRef, MMRefs
from prime_rl.utils.mm import build_mm_refs


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


def _qwen_item(grid, uri: str = "file:///tmp/missing-image.png"):
    return {
        "kind": RAW_MM_ITEM_KIND,
        "modality": "image",
        "family": "qwen_vl",
        "layout_fingerprint": "f" * 32,
        "raw_image_uri": uri,
        "payload": {"image_grid_thw": grid},
    }


def _refs(uri: str = "file:///tmp/missing-image.png") -> MMRefs:
    return MMRefs(
        images=[MMImageRef(item=_qwen_item([[1, 1, 1]], uri), hash="a" * 32, uri=uri, offset=1, length=1)],
    )


def _loader() -> DataLoader:
    loader = object.__new__(DataLoader)
    loader.multi_run_manager = SimpleNamespace(max_runs=1)
    loader.mm_materializer = _MissingMaterializer()
    loader.missing_mm_image_policy = "placeholder_zero_loss"
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
        seq_lens=[3],
        temperatures=[1.0, 1.0, 1.0],
        env_names=["env", "env", "env"],
        lora_num_tokens=[3],
        mm_refs=_refs(),
        mm_token_type_ids=[0, 1, 0],
    )


def test_build_mm_refs_accepts_offloaded_file_uri(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    multi_modal_data = SimpleNamespace(
        mm_items={"image": [_qwen_item([[1, 1, 1]], image_path.as_uri())]},
        mm_hashes={"image": ["a" * 32]},
        mm_placeholders={"image": [SimpleNamespace(offset=1, length=1)]},
    )

    refs = build_mm_refs(multi_modal_data)

    assert refs == _refs(image_path.as_uri())


def test_dataloader_uses_zero_loss_placeholder_for_missing_raw_image():
    tensor_batch = _loader()._micro_batch_to_tensor(_micro_batch())

    assert tensor_batch["mm_kwargs"] is not None
    assert tensor_batch["mm_kwargs"]["pixel_values"].shape == (1, 24)
    assert tensor_batch["mm_forward_policy"] == ForwardPolicy(pass_position_ids_with_mm=False)
    assert tensor_batch["loss_mask"].tolist() == [[False, False, False]]
    assert tensor_batch["advantages"].tolist() == [[0.0, 0.0, 0.0]]
    assert tensor_batch["mm_token_type_ids"].tolist() == [[0, 1, 0]]
