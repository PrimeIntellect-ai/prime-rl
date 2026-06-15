import base64
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from renderers.base import MultiModalData, PlaceholderRange

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import TrainRollout


@pytest.mark.asyncio
async def test_process_rollout_offloads_inline_images_before_building_mm_refs(tmp_path):
    raw_image = b"image-bytes"
    data_uri = "data:image/jpeg;base64," + base64.b64encode(raw_image).decode("ascii")
    raw = {
        "example_id": 1,
        "trajectory": [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
                    }
                ],
                "completion": [{"role": "assistant", "content": "ok"}],
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [False, False],
                    "completion_ids": [3],
                    "completion_mask": [True],
                    "completion_logprobs": [-0.1],
                    "multi_modal_data": MultiModalData(
                        mm_hashes={"image": ["image-hash"]},
                        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
                        mm_items={
                            "image": [
                                {
                                    "pixel_values": torch.tensor([[1.0]], dtype=torch.float32),
                                    "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                                }
                            ]
                        },
                    ),
                },
            }
        ],
        "sampling_args": {"temperature": 1.0},
        "error": None,
    }
    rollout = TrainRollout(
        raw=raw,
        env_name="test-env",
        example_id=1,
        group_id=uuid.uuid4(),
        policy_version=0,
        off_policy_steps=0,
    )
    sink = TrainSink.__new__(TrainSink)
    sink.config = SimpleNamespace(defer_mm_materialization=True)
    sink.renderer = None
    sink.mm_token_type_ids_mapping = {2: 1}
    sink._mm_asset_root = tmp_path

    await TrainSink.process_rollout(sink, rollout)

    assert len(rollout.samples) == 1
    refs = rollout.samples[0].mm_refs
    assert refs is not None
    assert refs.uris and refs.uris[0].startswith("file://")
    assert refs.uris[0] == raw["trajectory"][0]["prompt"][0]["content"][0]["image_url"]["url"]
    assert Path(refs.uris[0][len("file://") :]).read_bytes() == raw_image
